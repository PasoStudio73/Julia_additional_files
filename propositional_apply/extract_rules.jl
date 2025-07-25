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
Xcl = Xc[40:60, :]
ycl = yc[40:60]
dsc = setup_dataset(
    Xcl, ycl,
    model=RandomForestClassifier(),
    resample=(type=Holdout(shuffle=true), train_ratio=0.7, rng=Xoshiro(1))
)
solemc = train_test(dsc)

ds=dsc
solem = solemc
solemodels(solem::SX.ModelSet) = solem.sole
extractor = InTreesRuleExtractor(max_rules=-1, min_coverage=0.01)
get_rawmodel(ds::SX.EitherDataSet) = ds.mach.fitresult[1]

# soleposthoc
extractor = InTreesRuleExtractor()
rules = reduce(vcat, map(enumerate(solemodels(solem))) do (i, model)
    test = SX.get_test(ds.pidxs[i])
    X_test, y_test = SX.get_X(ds)[test, :], SX.get_y(ds)[test]
    RuleExtraction.modalextractrules(extractor, model, X_test, y_test)
end)

#solexplorer
modelc = symbolic_analysis(dsc, solemc, rules=InTreesRuleExtractor(), measures=(accuracy,))

# soleposthoc
extractor = LumenRuleExtractor(;ott_mode=false)
rules = reduce(vcat, map(enumerate(solemodels(solem))) do (i, model)
    # rawmodel = get_rawmodel(ds)
    RuleExtraction.modalextractrules(extractor, model)
end)

# # ---------------------------------------------------------------------------- #
# #                              rules extraction                                #
# # ---------------------------------------------------------------------------- #
# function rules_extraction!(model::Modelset, ds::Dataset, mach::MLJ.Machine)
#     model.rules = EXTRACT_RULES[model.setup.rulesparams.type](model, ds, mach)
# end