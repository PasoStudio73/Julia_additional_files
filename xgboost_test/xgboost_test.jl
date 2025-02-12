using CSV, DataFrames
using MLJ, MLJBase
using CategoricalArrays
using XGBoost, MLJXGBoostInterface
using SoleXplorer, SoleModels
import MLJModelInterface as MMI
using MLJDecisionTreeInterface
using AbstractTrees, StatsBase, Random

X, y       = SoleData.load_arff_dataset("NATOPS")
train_seed = 11
rng        = Random.Xoshiro(train_seed)
Random.seed!(train_seed)

# downsize dataset
num_cols_to_sample = 10
num_rows_to_sample = 10
chosen_cols = StatsBase.sample(rng, 1:size(X, 2), num_cols_to_sample; replace=false)
chosen_rows = StatsBase.sample(rng, 1:size(X, 1), num_rows_to_sample; replace=false)

X = X[chosen_rows, chosen_cols]
y = y[chosen_rows]

m = SoleXplorer.XGBoostClassifierModel()
ds = prepare_dataset(X, y, m)

# MLJXGBoostInterface
classifier = MLJXGBoostInterface.XGBoostClassifier(num_round=1, max_depth=3, objective="multi:softprob")
mach = MLJ.machine(classifier, ds.Xtrain, ds.ytrain)
fit!(mach, verbosity=0)
#

function get_condition(featidstr, featval, featurenames; test_operator)
    featid = parse(Int, featidstr[2:end]) + 1 # considering 0-based indexing in XGBoost feature ids
    feature = isnothing(featurenames) ? VariableValue(featid) : VariableValue(featid, featurenames[featid])
    return ScalarCondition(feature, test_operator, featval)
end

function satisfies_conditions(row::DataFrameRow, formula)
    # check_cond = true
    # for atom in formula
    #     if !atom.value.metacond.test_operator(row[atom.value.metacond.feature.i_variable], atom.value.threshold)
    #         check_cond = false
    #     end
    # end
    # return check_cond

    all(atom -> atom.value.metacond.test_operator(
                    row[atom.value.metacond.feature.i_variable],
                    atom.value.threshold), formula
                )
end

function bitmap_check_conditions(X, formula)
    BitVector([satisfies_conditions(row, formula) for row in eachrow(X)])
end

function pasomodel(
    model::Vector{<:XGBoost.Node},
    args...;
    weights::Union{AbstractVector{<:Number}, Nothing}=nothing,
    classlabels = nothing,
    featurenames = nothing,
    keep_condensed = false,
    kwargs...
)
    # TODO
    if keep_condensed && !isnothing(classlabels)
        # info = (;
        #     apply_preprocess=(y -> orig_O(findfirst(x -> x == y, classlabels))),
        #     apply_postprocess=(y -> classlabels[y]),
        # )
        info = (;
            apply_preprocess=(y -> findfirst(x -> x == y, classlabels)),
            apply_postprocess=(y -> classlabels[y]),
        )
        keep_condensed = !keep_condensed
        # O = eltype(classlabels)
    else
        info = (;)
        # O = orig_O
    end

    trees = filter(!isnothing, map(t -> pasomodel(t, args...; classlabels, featurenames, keep_condensed, kwargs...), model))


    if !isnothing(featurenames)
        info = merge(info, (; featurenames=featurenames, ))
    end

    info = merge(info, (;
            leaf_values=vcat([t.info[:leaf_values] for t in trees]...),
            supporting_predictions=vcat([t.info[:supporting_predictions] for t in trees]...),
            supporting_labels=vcat([t.info[:supporting_labels] for t in trees]...),
        )
    )

    if isnothing(weights)
        m = DecisionEnsemble(trees, info)
    else
        m = DecisionEnsemble(trees, weights, info)
    end
    return m
end

"""
    solemodel(tree::XGBoost.Node; fl=Formula[], fr=Formula[], classlabels=nothing, featurenames=nothing, keep_condensed=false)

Traverses a learned XGBoost tree, collecting the path conditions for each branch. 
Left paths (<) store conditions in `fl`, right paths (≥) store conditions in `fr`. 
When reaching a leaf, calls `xgbleaf` with the path's collected conditions.
"""
function pasomodel(
    tree::XGBoost.Node,
    X::AbstractDataFrame,
    y::AbstractVector;
    path_conditions = Formula[],
    classlabels=nothing,
    featurenames=nothing,
    keep_condensed=false
)
    keep_condensed && error("Cannot keep condensed XGBoost.Node.")

    # xgboost trees could be composed of only one leaf, without any split
    isnothing(tree.split) && return nothing

    antecedent = Atom(get_condition(tree.split, tree.split_condition, featurenames; test_operator=(<)))
    
    # Create a new path for the left branch
    left_path = copy(path_conditions)
    push!(left_path, Atom(get_condition(tree.split, tree.split_condition, featurenames; test_operator=(<))))
    
    # Create a new path for the right branch
    right_path = copy(path_conditions)
    push!(right_path, Atom(get_condition(tree.split, tree.split_condition, featurenames; test_operator=(≥))))
    
    lefttree = if isnothing(tree.children[1].split)
        # @show SoleModels.join_antecedents(left_path)
        xgbleaf(tree.children[1], left_path, X, y; classlabels, featurenames)
    else
        pasomodel(tree.children[1], X, y; path_conditions=left_path, classlabels=classlabels, featurenames=featurenames)
    end
    
    righttree = if isnothing(tree.children[2].split)
        # @show SoleModels.join_antecedents(right_path)
        xgbleaf(tree.children[2], right_path, X, y; classlabels, featurenames)
    else
        pasomodel(tree.children[2], X, y; path_conditions=right_path, classlabels=classlabels, featurenames=featurenames)
    end

    info = (;
        leaf_values = [lefttree.info[:leaf_values]..., righttree.info[:leaf_values]...],
        supporting_predictions = [lefttree.info[:supporting_predictions]..., righttree.info[:supporting_predictions]...],
        supporting_labels = [lefttree.info[:supporting_labels]..., righttree.info[:supporting_labels]...],
    )
    return Branch(antecedent, lefttree, righttree, info)
end

function xgbleaf(
    leaf::XGBoost.Node,
    formula::Vector{<:Formula},
    X::AbstractDataFrame,
    y::AbstractVector;
    classlabels=nothing,
    featurenames=nothing,
    keep_condensed=false
)
    keep_condensed && error("Cannot keep condensed XGBoost.Node.")

    bitX = bitmap_check_conditions(X, formula)
    prediction = SoleModels.bestguess(y[bitX])
    labels = unique(y)

    # if !isnothing(classlabels)
    #     prediction = classlabels[prediction]
    #     labels = classlabels[labels]
    # end

    info = (;
        leaf_values = leaf.leaf,
        supporting_predictions = fill(prediction, length(labels)),
        supporting_labels = labels,
    )
    return SoleModels.ConstantModel(prediction, info)
end

##### learn_method
get_encoding(classes_seen) = Dict(MMI.int(c) => c for c in MMI.classes(classes_seen))
get_classlabels(encoding) = [string(encoding[i]) for i in sort(keys(encoding) |> collect)]

trees = XGBoost.trees(mach.fitresult[1])
encoding = get_encoding(mach.fitresult[2])
classlabels = get_classlabels(encoding)
featurenames = mach.report.vals[1][1]
dt = pasomodel(trees, ds.Xtrain, ds.ytrain; classlabels, featurenames)
apply!(dt, ds.Xtest, ds.ytest)

##### performance measures
preds = MLJ.predict(mach, ds.Xtest)
yhat = MLJ.mode.(preds)
kp = MLJ.kappa(yhat, ds.ytest)
acc = MLJ.accuracy(yhat, ds.ytest)

#####
feature_names = names(ds.Xtest)
y_coded_train = @. CategoricalArrays.levelcode(ds.ytrain) - 1 # convert to 0-based indexing
y_coded_test  = @. CategoricalArrays.levelcode(ds.ytest)  - 1 # convert to 0-based indexing
dstrain = XGBoost.DMatrix((ds.Xtrain, y_coded_train); feature_names)
dstest  = XGBoost.DMatrix((ds.Xtest, y_coded_test); feature_names)
nlabels = 6
xgbmodel = XGBoost.xgboost(dstrain, num_round=1, max_depth=3, objective="multi:softprob", num_class=nlabels)
xgbtree = XGBoost.trees(xgbmodel)
