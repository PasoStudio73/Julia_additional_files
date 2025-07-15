using Test
using MLJ, SoleXplorer
using DataFrames, Random
using SoleData
const SX = SoleXplorer
using SolePostHoc

Xc, yc = @load_iris
Xc = DataFrame(Xc)

Xr, yr = @load_boston
Xr = DataFrame(Xr)

Xts, yts = SoleData.load_arff_dataset("NATOPS")

# ---------------------------------------------------------------------------- #
#                       Sole vs MLJ machine & fit setup                        #
# ---------------------------------------------------------------------------- #
dsc = setup_dataset(
    Xc, yc,
    model=RandomForestClassifier(),
    resample=(type=Holdout(shuffle=true), train_ratio=0.7, rng=Xoshiro(1))
)
solemc = train_test(dsc)

# ds=dsc
# solem = solemc
# solemodels(solem::SX.ModelSet) = solem.sole
# extractor = InTreesRuleExtractor(max_rules=10)

# rules = reduce(vcat, map(enumerate(solemodels(solem))) do (i, model)
#     test = SX.get_test(ds.pidxs[i])
#     X_test, y_test = SX.get_X(ds)[test, :], SX.get_y(ds)[test]
#     RuleExtraction.modalextractrules(extractor, model, X_test, y_test)
# end)

modelc = symbolic_analysis(dsc, solemc, rules=InTreesRuleExtractor(), measures=(accuracy,))


# # ---------------------------------------------------------------------------- #
# #                              rules extraction                                #
# # ---------------------------------------------------------------------------- #
# function rules_extraction!(model::Modelset, ds::Dataset, mach::MLJ.Machine)
#     model.rules = EXTRACT_RULES[model.setup.rulesparams.type](model, ds, mach)
# end