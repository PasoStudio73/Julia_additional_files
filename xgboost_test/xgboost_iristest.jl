using CSV, DataFrames
using MLJ, MLJBase
using CategoricalArrays
using XGBoost, MLJXGBoostInterface
using Sole, SoleXplorer
import MLJModelInterface as MMI
using MLJDecisionTreeInterface

x, y = @load_iris
X = DataFrame(x)

fixed_indices = [83, 12, 145, 22, 98, 125, 44, 3, 150, 67, 89, 31, 122, 15, 78, 99, 45, 11, 134, 52, 23, 120, 48, 88, 28, 4, 65, 17, 127, 49]
train_indices = setdiff(1:150, fixed_indices)

feature_names = names(X)
labels       = unique(y)
nlabels      = length(labels)

X_train = X[train_indices, :]
X_test  = X[fixed_indices, :]
y_train = y[train_indices]
y_test  = y[fixed_indices]

y_coded_train = @. CategoricalArrays.levelcode(y_train) - 1 # convert to 0-based indexing
y_coded_test  = @. CategoricalArrays.levelcode(y_test)  - 1 # convert to 0-based indexing

dstrain = XGBoost.DMatrix((X_train, y_coded_train); feature_names)
dstest  = XGBoost.DMatrix((X_test, y_coded_test);   feature_names)

bst = XGBoost.xgboost(dstrain, num_round=10, max_depth=3, num_class=nlabels, objective="multi:softmax")

# return AbstractTrees.jl compatible tree objects describing the model
bst_tree = XGBoost.trees(bst)

# create and train a gradient boosted tree model of 5 trees
bst = XGBoost.xgboost(dtrain, num_round=5, max_depth=6, objective="reg:squarederror")
y_predict = XGBoost.predict(bst, dtest)

# early stopping
bst = XGBoost.xgboost(dtrain, 
    num_round = 100, 
    eval_metric = "rmse", 
    watchlist = OrderedDict(["train" => dtrain, "eval" => dtest]), 
    early_stopping_rounds = 5, 
    max_depth=6, 
    η=0.3
)
# get the best iteration and use it for prediction
y_predict = XGBoost.predict(bst, dtest, ntree_limit = bst.best_iteration)

bst = XGBoost.xgboost(dtrain, num_round=1; XGBoost.randomforest()...)

# we can also retain / use the best score (based on eval_metric) which is stored in the booster
println("Best RMSE from model training $(round((bst.best_score), digits = 8)).")

prediction_rounded = round.(Int, y_predict)

MLBase.errorrate(y_code[tt_pairs.test], prediction_rounded)
MLBase.confusmat(length(levels(y_code)), Array(y_code[tt_pairs.test] .+ 1), Array(prediction_rounded) .+ 1)

XGBoost.importancetable(bst)

# return AbstractTrees.jl compatible tree objects describing the model
bst_tree = XGBoost.trees(bst)

AbstractTrees.repr_tree(bst_tree)
AbstractTrees.print_tree(bst_tree)

############à#######
# function xgboost #
####################
params = (;
    test                        = 1, 
    num_round                   = 10, 
    booster                     = "gbtree", 
    disable_default_eval_metric = 0, 
    eta                         = 0.3, 
    num_parallel_tree           = 1, 
    gamma                       = 0.0, 
    max_depth                   = 3, 
    min_child_weight            = 1.0, 
    max_delta_step              = 0.0, 
    subsample                   = 1.0, 
    colsample_bytree            = 1.0, 
    colsample_bylevel           = 1.0, 
    colsample_bynode            = 1.0, 
    lambda                      = 1.0, 
    alpha                       = 0.0, 
    tree_method                 = "auto", 
    sketch_eps                  = 0.03, 
    scale_pos_weight            = 1.0, 
    updater                     = nothing, 
    refresh_leaf                = 1, 
    process_type                = "default", 
    grow_policy                 = "depthwise", 
    max_leaves                  = 0, 
    max_bin                     = 256, 
    predictor                   = "cpu_predictor", 
    sample_type                 = "uniform", 
    normalize_type              = "tree", 
    rate_drop                   = 0.0, 
    one_drop                    = 0, 
    skip_drop                   = 0.0, 
    feature_selector            = "cyclic", 
    top_k                       = 0, 
    tweedie_variance_power      = 1.5, 
    objective                   = "automatic", 
    base_score                  = 0.5, 
    early_stopping_rounds       = 0, 
    watchlist                   = nothing, 
    nthread                     = 1, 
    importance_type             = "gain", 
    seed                        = nothing, 
    validate_parameters         = false, 
    eval_metric                 = String[]
)

classifier = MLJXGBoostInterface.XGBoostClassifier(; params...)
mach = MLJ.machine(classifier, X_train, y_train)
fit!(mach, verbosity=0)

# learn_method = (
#     (mach, X, y) -> begin
        fitresult = SoleModels.fitted_params(mach)

        solemodel(fitresult.trees, fitresult.encoding)


plain_classifier = XGBoostClassifier(num_round=10, max_depth=3)
m = machine(plain_classifier, X_train, y_train)
fit!(m,verbosity = 0)

fitresult, cache, report = MLJBase.fit(plain_classifier, 0, X_train, y_train;)

yhat = mode.(MLJ.predict(plain_classifier, fitresult, x_test))
classification_rate = 1- sum(yhat .!= y_test)/length(y_test)



classifier = MLJXGBoostInterface.XGBoostClassifier(; params...)
mach = MLJ.machine(classifier, x_train, y_train)
fit!(mach, verbosity=0)

# learn_method = (
#     (mach, X, y) -> begin
        fitresult = SoleModels.fitted_params(mach)
        dt = SoleModels.solemodel(fitresult.stumps, fitresult.encoding)

# testset = modelset.learn_method(mach, Xtest[i], ytest[i])
weights = mach.fitresult[2]
classlabels = sort(mach.fitresult[3])
featurenames = MLJ.report(mach).features
dt = solemodel(MLJ.fitted_params(mach).stumps; weights, classlabels, featurenames)
apply!(dt, X, y)


#########################################

# function DecisionTreeModel
type = MLJDecisionTreeInterface.DecisionTreeClassifier
config  = (; algo=:classification, type=DecisionTree, treatment=:aggregate)

params = (;
    max_depth              = -1,
    min_samples_leaf       = 1, 
    min_samples_split      = 2, 
    min_purity_increase    = 0.0, 
    n_subfeatures          = 0,
    post_prune             = false,
    merge_purity_threshold = 1.0,
    display_depth          = 5,
    feature_importance     = :impurity,
    rng                    = Random.TaskLocalRNG()
)

winparams = (type=SoleBase.wholewindow,)

# learn_method = (
#     (mach, X, y) -> (dt = solemodel(MLJ.fitted_params(mach).tree); apply!(dt, X, y); dt),
#     (mach, X, y) -> (dt = solemodel(MLJ.fitted_params(mach).best_fitted_params.tree); apply!(dt, X, y); dt)
# )

# tuning = (
#     tuning        = false,
#     method        = (type = latinhypercube, ntour = 20),
#     params        = TUNING_PARAMS,
#     ranges        = [
#         model -> MLJ.range(model, :merge_purity_threshold, lower=0, upper=1),
#         model -> MLJ.range(model, :feature_importance, values=[:impurity, :split])
#     ]
# )

# rules_method = SoleModels.PlainRuleExtractor()

# classifier = getmodel(m)
classifier = MLJDecisionTreeInterface.DecisionTreeClassifier(; params...)

# mach = fitmodel(m, classifier, ds);
# model = testmodel(m, mach, ds);
mach = MLJ.machine(classifier, x_train, y_train)
fit!(mach, verbosity=0)

fitresult = MMI.fitted_params(mach)

