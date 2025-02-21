import XGBoost as XGB
using Sole
using SoleXplorer, SoleModels
using Random, StatsBase, JLD2, DataFrames
using CategoricalArrays
using PyCall

# PyCall.Conda.add("scipy")
# PyCall.Conda.add("scikit-learn")
# PyCall.Conda.add("xgboost")
# PyCall.Conda.add("supertree")

xgb = pyimport("xgboost")
stree = pyimport("supertree")

X, y = Sole.load_arff_dataset("NATOPS")

############ JULIA CODE ############
models=(type=:xgboost_classifier,
    params=(
        num_round=1,
        max_depth=10,
        objective="multi:softprob",
        seed=11),
        winparams=(; type=adaptivewindow, nwindows=5),
        features=catch9
    )
modelsets = validate_modelset(models, typeof(y), nothing, preprocess)
m = modelsets[1]
ds = prepare_dataset(X, y, m)
classifier = getmodel(m)
mach = fitmodel(m, classifier, ds)
trees        = XGB.trees(mach.fitresult[1])
encoding     = SoleXplorer.get_encoding(mach.fitresult[2])
classlabels  = SoleXplorer.get_classlabels(encoding)
featurenames = mach.report.vals[1][1]
ds_safetest = vcat(ds.ytest, "nothing")
dt           = solemodel(trees, @views(Matrix(ds.Xtest)), @views(ds_safetest); classlabels, featurenames)

################# PYTHON CODE #################
xgbmodel = xgb.XGBClassifier(
    n_estimators=1,
    max_depth=10,
    objective="multi:softprob", 
    seed=11
)
y_coded_train = @. CategoricalArrays.levelcode(ds.ytrain) - 1 # convert to 0-based indexing
y_coded_test  = @. CategoricalArrays.levelcode(ds.ytest)  - 1 # convert to 0-based indexing
dstrain = XGB.DMatrix((ds.Xtrain, y_coded_train); feature_names=string.(featurenames))
dstest  = XGB.DMatrix((ds.Xtest, y_coded_test); feature_names=string.(featurenames))
pdt = xgbmodel.fit(dstrain, y_coded_train)
dump_list=xgbmodel.get_booster().get_dump()

# julia> dump_list[1]
# "0:[f347<1.3520143] yes=1,no=2,missing=2
#       [f941<2.09464097] yes=3,no=4,missing=4
#                       leaf=-0.175549462
#                       leaf=0.189473689
#       [f412<0.0284870006] yes=5,no=6,missing=6
#           [f9<-1.67472303] yes=7,no=8,missing=8
#               [f414<-0.847468972] yes=11,no=12,missing=12
#                       leaf=0.0209302269
#                       [f938<1.64899158] yes=17,no=18,missing=18
#                           leaf=0.224999994
#                           leaf=0.786713243
#                       [f523<0.00508883223] yes=13,no=14,missing=14
#                           leaf=0.33157894
#                           leaf=-0.140963852

# julia> dt
# ▣ Ensemble{Union{Nothing, CategoricalValue{String, UInt32}}} of 6 models of type Branch
# ├[1/6]┐ ([mean(X[Thumb r])w3] < 1.3520143)
# │     ├✔ ([entropy_pairs(Z[Thumb l])w2] < 2.09464097)
# │     │ ├✔ Lock wings : (ninstances = 6, ncovered = 6, confidence = 0.17, lift = 1.0)
# │     │ └✘ I have command : (ninstances = 6, ncovered = 6, confidence = 0.17, lift = 1.0)
# │     └✘ ([median(Y[Elbow r])w3] < 0.0284870006)
# │       ├✔ ([maximum(Y[Hand tip l])w5] < -1.67472303)
# │       │ ├✔ ([median(Y[Elbow r])w5] < -0.847468972)
# │       │ │ ├✔ Not clear : (ninstances = 6, ncovered = 6, confidence = 0.17, lift = 1.0)
# │       │ │ └✘ ([entropy_pairs(Y[Thumb l])w4] < 1.64899158)
# │       │ │   ├✔ Not clear : (ninstances = 6, ncovered = 6, confidence = 0.17, lift = 1.0)
# │       │ │   └✘ Not clear : (ninstances = 6, ncovered = 6, confidence = 0.17, lift = 1.0)
# │       │ └✘ ([std(Z[Elbow l])w4] < 0.00508883223)
# │       │   ├✔ nothingError showing value of 

feature_names_class = ["f$i" for i in 1:size(dstrain, 2)]

st = stree.SuperTree(
    xgbmodel, 
    dstrain, 
    y_coded_train, 
    feature_names_class
)
# Visualize the tree
st.show_tree(which_tree=0)
st.save_html() 

