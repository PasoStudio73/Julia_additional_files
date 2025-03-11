using SoleModels

phi = parseformula("p∧q∨r")

const_integer = 1

cmodel_number = ConstantModel{Number}(const_integer)
cmodel_integer = ConstantModel{Int}(const_integer)

rmodel_number = Rule(phi, cmodel_number)
rmodel_integer = Rule(phi, cmodel_integer)

# taken from types/model.jl

# Interface
# - `apply(m::AbstractModel, i::AbstractInterpretation; kwargs...)`

iscomplete(rmodel_number)
outcometype(rmodel_number)
outputtype(rmodel_number)

immediatesubmodels(rmodel_number)
nimmediatesubmodels(rmodel_number)
listimmediaterules(rmodel_number)

# info(rmodel_number, [key, [defaultval]])
# info!(rmodel_number, key, value)
# hasinfo(rmodel_number, key)

# Utility functions
# - `apply(rmodel_number, i::AbstractInterpretationSet; kwargs...)`

submodels(rmodel_number)
nsubmodels(rmodel_number)
leafmodels(rmodel_number)
nleafmodels(rmodel_number)

subtreeheight(rmodel_number)
listrules(
        rmodel_number;
        use_shortforms=true,
        use_leftmostlinearform=nothing,
        normalize=false,
        force_syntaxtree=false,
    )
joinrules(rmodel_number, silent=false)


[compat]
AbstractTrees = "0.4"
BenchmarkTools = "1"
CSV = "0.10"
CategoricalArrays = "0.10"
DataFrames = "1"
DataStructures = "0.18"
DecisionTree = "0.12"
FillArrays = "1"
FunctionWrappers = "1"
Graphs = "1"
HTTP = "1"
IterTools = "1"
Lazy = "0.15"
MLJ = "0.19 - 0.20"
MLJBase = "1.6 - 1.7"
MLJDecisionTreeInterface = "0.4"
MLJModelInterface = "1"
PrettyTables = "2"
ProgressMeter = "1"
Random = "1"
Reexport = "1"
Revise = "3"
SoleBase = "0.13"
SoleData = "0.15, 0.16"
SoleLogics = "0.11 - 0.12"
StatsBase = "0.30 - 0.34"
Suppressor = "0.2"
Tables = "1"
ThreadSafeDicts = "0.1"
XGBoost = "2"
ZipFile = "0.10"
julia = "1"