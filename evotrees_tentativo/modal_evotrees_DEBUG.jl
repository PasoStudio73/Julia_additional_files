using Sole
using SoleXplorer
using Random, StatsBase, JLD2, DataFrames
using EvoTrees
using MLJ, MLJBase
using Base.Threads
using BenchmarkTools
using Tables
using Statistics


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

model = SoleXplorer.get_model(:modal_evotree_classifier; relations=:IA7, features, set=X)
mtest = SoleXplorer.get_model(:decision_tree)
ds = SoleXplorer.preprocess_dataset(X_selected, y_selected, model, features=features, train_ratio=0.99, treatment_params=(nwindows=3,))
test = SoleXplorer.preprocess_dataset(X_selected, y_selected, mtest, features=features, train_ratio=0.99)
mach = MLJ.machine(model.classifier, ds.X, ds.y)

# fit!(mach, verbosity=0)
# function MMI.fit(model::ModalEvoTypes, verbosity::Int, X, y, w=nothing)
X = ds.X
Xtest = test.X
y = ds.y
w = nothing

nobs = nrow(X)
fnames = names(X)
w = isnothing(w) ? EvoTrees.device_ones(EvoTrees.CPU, Float32, nobs) : Vector{Float32}(w)
m, cache, bias = SoleXplorer.ModalEvoTrees.modal_init_core(model.classifier, EvoTrees.CPU, X, fnames, y, w, nothing)

