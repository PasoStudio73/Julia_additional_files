using Test
using Sole, ModalDecisionTrees
import SoleXplorer as SX
using SoleXplorer
using Random, StatsBase, JLD2, DataFrames
using RDatasets
using MLJ, MLJDecisionTreeInterface
using DecisionTree

using MLJModelInterface

# ---------------------------------------------------------------------------- #
X, y = SoleData.load_arff_dataset("NATOPS")
train_ratio = 0.8
shuffle = true
train_seed = 11

# preprocessamento dataset
rng = Random.Xoshiro(train_seed)
Random.seed!(train_seed)

m1 = SX.AdaBoostModel()
m1.learn_method = m1.learn_method[1]
ds1 = preprocess_dataset(X, y, m1)
c1 = get_model(m1, ds1)
mc1 = modelfit(m1, c1, ds1)
fp1 = MLJ.fitted_params(mc1)
r1 = modeltest(m1, mc1, ds1)
f1 = SX.get_predict(mc1, ds1)


m2 = SX.ModalAdaBoostModel()
m2.learn_method = m2.learn_method[1]
ds2 = preprocess_dataset(X, y, m2)
c2 = get_model(m2, ds2)
mc2 = modelfit(m2, c2, ds2)
fp2 = MLJ.fitted_params(mc2)
rp2 = MLJ.report(mc2)
r2 = modeltest(m2, mc2, ds2)
f2 = SX.get_predict(mc2, ds2)
# (Xnew2, ynew2, var_grouping2, classes_seen2, w2) = MLJModelInterface.reformat(c2, ds2.X[ds2.tt.test, :], y[ds2.tt.test], rp2.weights)
# preds2, sprinkledmodel2 = ModalDecisionTrees.sprinkle(mc2.fitresult.model, Xnew2, ynew2, tree_weights=rp2.weights)

m3 = SX.ModalRandomForestModel()
m3.learn_method = m3.learn_method[1]
ds3 = preprocess_dataset(X, y, m3)
c3 = get_model(m3, ds3)
mc3 = modelfit(m3, c3, ds3)
fp3 = MLJ.fitted_params(mc3)
rp3 = MLJ.report(mc3)
r3 = modeltest(m3, mc3, ds3)
f3 = SX.get_predict(mc3, ds3)
(Xnew3, ynew3, var_grouping3, classes_seen3, w3) = MLJModelInterface.reformat(c3, ds3.X[ds3.tt.test, :], y[ds3.tt.test])
preds3, sprinkledmodel3 = ModalDecisionTrees.sprinkle(mc3.fitresult.model, Xnew3, ynew3)

