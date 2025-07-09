using Test
using MLJ
using SoleXplorer
using DataFrames, Random
using SoleData
const SX = SoleXplorer

Xc, yc = @load_iris
Xc = DataFrame(Xc)

Xr, yr = @load_boston
Xr = DataFrame(Xr)

Xts, yts = SoleData.load_arff_dataset("NATOPS")

a = prepare_dataset(
        Xc, yc;
        model=DecisionTreeClassifier(),
        resample=(type=Holdout(shuffle=true), train_ratio=0.7, rng=Xoshiro(1))
)

resample=(type=Holdout(shuffle=true), train_ratio=0.7, valid_ratio=0.1, rng=Xoshiro(1))
a=SX.partition(yc; resample...)
