using CSV, DataFrames
using MLJ, MLJBase
using CategoricalArrays
using XGBoost, MLJXGBoostInterface
using Sole, SoleXplorer
import MLJModelInterface as MMI
using MLJDecisionTreeInterface, Random

# Use joinpath for proper path handling
train_path = joinpath(@__DIR__, "train.csv")
test_path = joinpath(@__DIR__, "test.csv")

dtrain = CSV.read(train_path, DataFrame)
dtest = CSV.read(test_path, DataFrame)

x_train = dtrain[:, 1:end-1]
y_train = CategoricalArray(dtrain[:, end])

x_test = dtest[:, 1:end-1]
y_test = CategoricalArray(dtest[:, end])

plain_classifier = XGBoostClassifier(num_round=100, max_depth=3, seed=123)
m = machine(plain_classifier, x_train, y_train)
fit!(m,verbosity = 0)

fitresult, cache, report = MLJBase.fit(plain_classifier, 0, x_train, y_train;)

yhat = mode.(MLJ.predict(plain_classifier, fitresult, x_test))
classification_rate = 1- sum(yhat .!= y_test)/length(y_test)

#########################################

# function xgboost
type   = MLJXGBoostInterface.XGBoostClassifier
config = (; algo=:classification, type=SoleXplorer.DecisionEnsemble, treatment=:aggregate)

params = (;
    test                        = 1, 
    num_round                   = 100, 
    booster                     = "gbtree", 
    disable_default_eval_metric = 0, 
    eta                         = 0.3, 
    num_parallel_tree           = 1, 
    gamma                       = 0.0, 
    max_depth                   = 6, 
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

tuning = (
    tuning        = true,
    method        = (type = latinhypercube, ntour = 20),
    params        = SoleXplorer.TUNING_PARAMS,
    ranges        = [
        model -> MLJ.range(model, :max_depth, lower=3, upper=6),
        model -> MLJ.range(model, :sample_type, values=["uniform", "weighted"])
    ]
)

classifier = MLJXGBoostInterface.XGBoostClassifier(; params...)
mach = MLJ.machine(classifier, x_train, y_train)
fit!(mach, verbosity=0)

# learn_method = (
#     (mach, X, y) -> begin
        fitresult = SoleModels.fitted_params(mach)
        dt = SoleModels.solemodel(fitresult...)

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

