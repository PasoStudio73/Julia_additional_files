using Sole
using SoleXplorer
using MLJ
# using StatsBase, JLD2, DataFrames

models = (; type=:decisiontree)
X, y = SoleData.load_arff_dataset("NATOPS")

modelset = validate_modelset(models, typeof(y))[1]
classifier = getmodel(modelset)
ds = prepare_dataset(X, y, modelset)

featuresnames = [VariableValue(i, val) for (i, val) in enumerate(names(ds.Xtrain))]
lstrain = scalarlogiset(ds.Xtrain, featuresnames)
lstest  = scalarlogiset(ds.Xtest,  featuresnames)

mach = [MLJ.machine(classifier, x, y) |> m -> fit!(m, verbosity=0) for (x, y) in zip([ds.Xtrain], [ds.ytrain])]

lsmach = [MLJ.machine(classifier, x, y) |> m -> fit!(m, verbosity=0) for (x, y) in zip([lstrain.base.featstruct], [ds.ytrain])]