using MLJ
using DataFrames
using MLJXGBoostInterface
using SoleModels
import XGBoost as XGB
using CategoricalArrays
using Random
using PyCall
xgb = pyimport("xgboost")
st = pyimport("supertree")

X, y = @load_iris
X = DataFrame(X)

seed, num_round, eta = 3, 10, 0.1
rng = Xoshiro(seed)
train, test = partition(eachindex(y), train_ratio; shuffle=true, rng)
X_train, y_train = X[train, :], y[train]
X_test, y_test = X[test, :], y[test]

# MLJ
XGTrees = MLJ.@load XGBoostClassifier pkg=XGBoost
model = XGTrees(; num_round, eta, objective="multi:softmax")
mach = machine(model, X_train, y_train)
fit!(mach)
trees = XGB.trees(mach.fitresult[1])
solem = solemodel(trees, Matrix(X_train), y_train; classlabels, featurenames)
preds = apply(solem, X_test)
predsl = CategoricalArrays.levelcode.(categorical(preds)) .- 1

# XGBoost
yl_train = CategoricalArrays.levelcode.(categorical(y_train)) .- 1
bst = XGB.xgboost(
    (X_train, yl_train);
    eta,
    num_round,
    num_class=3,
    objective="multi:softmax"
)
xtrs = XGB.trees(bst)
xpreds = XGB.predict(bst, X_test)

# Python
xgb_classifier = xgb.XGBClassifier(
    objective="multi:softmax",
    num_class=3,
    eta=eta,
    n_estimators=num_round,
)
xgb_classifier.fit(Matrix(X_train), yl_train)
# st = st.SuperTree(xgb_classifier, Matrix(X_train), yl_train)
# st.save_html(which_tree=5)
y_pred = xgb_classifier.predict(Matrix(X_test))
