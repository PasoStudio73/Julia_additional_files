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