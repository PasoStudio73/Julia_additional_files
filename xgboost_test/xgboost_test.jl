using CSV, DataFrames
using MLJ, MLJBase
using CategoricalArrays
using XGBoost, MLJXGBoostInterface

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
