using Test
using MLJ, SoleXplorer
using DataFrames, Random
# using SoleData, SoleModels
using XGBoost
const SX = SoleXplorer
const XGB = XGBoost


Xr, yr = @load_boston
Xr = DataFrame(Xr)

Xc, yc = @load_iris
Xc = DataFrame(Xc)

# ---------------------------------------------------------------------------- #
#                        measures on solemodel models                          #
# ---------------------------------------------------------------------------- #

# ---------------------------------------------------------------------------- #
#                   decision tree solemodel regression path                    #
# ---------------------------------------------------------------------------- #
modelc, _, _ = train_test(Xc, yc);
modelr, _, _ = train_test(Xr, yr);

# DecisionTreeExt
# entrambi passano da solemodel 109 DT.InfoNode

# Verifico l'albero originale di DecisionTree regression
_, mach, _ = train_test(Xr, yr; model=(;type=:decisiontree, params=(;max_depth=2)));
r = MLJ.fitted_params(mach).tree

# julia> r.info
# (featurenames = [:Crim, :Zn, :Indus, :NOx, :Rm, :Age, :Dis, :Rad, :Tax, :PTRatio, :Black, :LStat],)

# julia> r.node.left
# Decision Tree
# Leaves: 2
# Depth:  1

# julia> r.node.left.left
# Decision Leaf
# Majority: 28.688043478260866
# Samples:  27

# Verifico l'albero originale di DecisionTree classification
_, mach, _ = train_test(Xc, yc; model=(;type=:decisiontree, params=(;max_depth=2)));
c = MLJ.fitted_params(mach).tree

# julia> c.info
# (featurenames = [:sepal_length, :sepal_width, :petal_length, :petal_width],
#  classlabels = ["setosa", "versicolor", "virginica"],)

# julia> c.node.right
# Decision Tree
# Leaves: 2
# Depth:  1

# julia> c.node.right.right
# Decision Leaf
# Majority: 3
# Samples:  40

# ---------------------------------------------------------------------------- #
#                                    xgboost                                   #
# ---------------------------------------------------------------------------- #
_, mach, _ = train_test(Xc, yc; model=(;type=:xgboost, params=(num_round=1, max_depth=2)));
c = XGB.trees(mach.fitresult[1])
# 3-element Vector{XGBoost.Node}: 3 alberi per le 3 classi
#  XGBoost.Node(split_feature=f2)
#  XGBoost.Node(split_feature=f2)
#  XGBoost.Node(split_feature=f3)

# julia> c[1].children
# 2-element Vector{XGBoost.Node}:
#  XGBoost.Node(leaf=0.423529446)
#  XGBoost.Node(leaf=-0.217894763)

_, ds = prepare_dataset(Xr, yr)
Tree = @load XGBoostRegressor pkg=XGBoost
tree = Tree(;num_round=1, max_depth=2, objective="reg:squarederror")
mach = MLJ.machine(tree, MLJ.table(@views ds.X; names=ds.info.vnames), @views ds.y)
train = ds.tt[1].train
MLJ.fit!(mach, rows=train, verbosity=0)
r = XGB.trees(mach.fitresult[1])
# 1-element Vector{XGBoost.Node}: # un solo albero per la regressione
#  XGBoost.Node(split_feature=f11)

# julia> r[1].children[1].children
# 2-element Vector{XGBoost.Node}:
#  XGBoost.Node(leaf=7.52650547)
#  XGBoost.Node(leaf=11.2097559)


# modelr, _, _ = train_test(
#     Xr, yr; model=(;type=:xgboost, params=(;num_round=1)),
#     preprocess=(;train_ratio=0.7, rng=Xoshiro(11))
# );

# # XGBoost model
# bst = XGBoost.xgboost(
#     (Xr, yr);
#     num_round=1,
#     objective="reg:squarederror"
# )
# xgbt = XGBoost.trees(bst)[1]
# # xpreds = XGBoost.predict(bst, X_test)

# # SoleXplorer model
# model.model = Vector{SX.AbstractModel}(undef, 1)
# predictor = SX.get_predictor!(model.setup)

# @test predictor == mljmach.model

# mach = MLJ.machine(predictor, MLJ.table(@views ds.X; names=ds.info.vnames), @views ds.y)

# train = ds.tt[1].train
# test  = ds.tt[1].test
# X_test  = DataFrame((@views ds.X[test, :]), ds.info.vnames)
# y_test  = @views ds.y[test]

# MLJ.fit!(mach, rows=train, verbosity=0)

# # learn_method
# trees        = XGBoost.trees(mach.fitresult[1])
# featurenames = mach.report.vals[1].features
# solem        = solemodel(trees, @views(Matrix(X_test)), @views(y_test); featurenames)
# # solem        = solemodel(trees, @views(Matrix(X)), @views(y); classlabels, featurenames)
# apply!(solem, mapcols(col -> Float32.(col), X), @views(y))
# return solem

# ---------------------------------------------------------------------------- #
#                          classification crash test                           #
# ---------------------------------------------------------------------------- #
@testset "data validation" begin
    for train_ratio in 0.5:0.1:0.9
        for seed in 1:40
            _, ds = prepare_dataset(Xc, yc; preprocess=(;train_ratio, rng=Xoshiro(seed)))
            X_train, y_train = ds.X[ds.tt[1].train, :], ds.y[ds.tt[1].train]
            X_test, y_test = ds.X[ds.tt[1].test, :], ds.y[ds.tt[1].test]

            for num_round in 10:10:50
                for eta in 0.1:0.1:0.6
                    # XGBoost model
                    yl_train = MLJ.levelcode.(MLJ.categorical(y_train)) .- 1
                    bst = XGB.xgboost((X_train, yl_train); num_round, eta, num_class=3, objective="multi:softmax", verbosity=0)
                    xg_preds = XGB.predict(bst, X_test)
                    yl_test = MLJ.levelcode.(MLJ.categorical(y_test)) .- 1
                    xg_accuracy = sum(xg_preds .== yl_test) / length(yl_test)

                    # SoleXplorer model
                    model = symbolic_analysis(
                        Xc, yc;
                        model=(type=:xgboost, params=(;num_round, eta)),
                        preprocess=(;train_ratio, rng=Xoshiro(seed)),
                        measures=(accuracy,)
                    )

                    @test model.measures.measures_values[1] == xg_accuracy
                end
            end
        end
    end
end

train_ratio = 0.5
seed = 1
_, ds = prepare_dataset(Xc, yc; preprocess=(;train_ratio, rng=Xoshiro(seed)))
X_train, y_train = ds.X[ds.tt[1].train, :], ds.y[ds.tt[1].train]
X_test, y_test = ds.X[ds.tt[1].test, :], ds.y[ds.tt[1].test]
num_round = 10
eta = 0.1
# XGBoost model
yl_train = MLJ.levelcode.(MLJ.categorical(y_train)) .- 1
bst = XGB.xgboost((X_train, yl_train); num_round, eta, num_class=3, objective="multi:softmax", verbosity=0)
xg_preds = XGB.predict(bst, X_test)
yl_test = MLJ.levelcode.(MLJ.categorical(y_test)) .- 1
xg_accuracy = sum(xg_preds .== yl_test) / length(yl_test)

# SoleXplorer model
mok = symbolic_analysis(
    Xc, yc;
    model=(type=:xgboost, params=(;num_round, eta)),
    preprocess=(;train_ratio, rng=Xoshiro(seed)),
    measures=(accuracy,)
)

mno = symbolic_analysis(
    Xc, yc;
    model=(type=:xgboost, params=(;num_round, eta)),
    preprocess=(;train_ratio, rng=Xoshiro(seed)),
    measures=(accuracy,)
)
@test model.measures.measures_values[1] == xg_preds


