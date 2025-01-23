using Sole
using SoleXplorer
using Random, StatsBase, JLD2, DataFrames
using RDatasets

# ---------------------------------------------------------------------------- #
X, y = SoleData.load_arff_dataset("NATOPS")
train_seed = 11;

# Select first 3 columns and convert to DataFrame
X_subset = DataFrame(X[:, 1:3], [:A, :B, :C])

# Find indices for each class
i_command = findall(y .== "I have command")[1:5]
i_wings = findall(y .== "Lock wings")[1:5]

# Combine indices
selected_indices = vcat(i_command, i_wings)

# Select the instances from X and y
X_selected = X_subset[selected_indices, :]
y_selected = y[selected_indices]

##############à DEBUGGING ############################
model_name = :modal_adaboost
features = [mean,]
rng = Random.Xoshiro(train_seed)
Random.seed!(train_seed)

m1 = SoleXplorer.get_model(:modal_adaboost; relations=:IA7, features, set=X)
ds1 = SoleXplorer.prepare_dataset(X_selected, y_selected, m1, features, train_ratio=0.99, treatment_params=(nwindows=3,))


# m1 = SoleXplorer.get_model(:adaboost)

# m3 = SoleXplorer.get_model(:modal_adaboost; relations=:IA7, features, set=X)
# ds2 = SoleXplorer.prepare_dataset(X, y, m2; features, treatment_params=(nwindows=10,))

SoleXplorer.modelfit!(m1, ds1; features, rng)
# SoleXplorer.modelfit!(m2, ds1; features, rng)
# SoleXplorer.modelfit!(m3, ds1; features, rng)

n_y = length(ds1.y)
weights = ones(n_y) / n_y
x = ModalDecisionTrees.wrapdataset(ds1.X, m2.classifier, nothing)

n1 = ModalDecisionTrees.build_stump(x[1], string.(ds1.y), weights)
n2 = DecisionTree.build_stump(string.(ds1.y), Array(ds1.X), weights)

t1 = ModalDecisionTrees.build_tree(x[1], string.(ds1.y), weights, max_depth=1)

