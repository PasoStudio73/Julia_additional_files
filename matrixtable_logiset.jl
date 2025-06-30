using Test
using DataFrames, Random

using SoleModels
using SoleLogics
using SoleData
using SoleBase
using MultiData
using SoleXplorer
const SX = SoleXplorer

using MLJ
using MLJXGBoostInterface
using DecisionTree
const DT = DecisionTree
using XGBoost
const XGB = XGBoost

Xr, yr = @load_boston
Xr = DataFrame(Xr)

Xc, yc = @load_iris
Xc = DataFrame(Xc)

Xts, yts = SoleData.load_arff_dataset("NATOPS")

# ---------------------------------------------------------------------------- #
#                        get rid of negations: benchmarks                      #
# ---------------------------------------------------------------------------- #
a = false
@btime a === false ? 1 : 0
# 2.815 ns (0 allocations: 0 bytes)
@btime a ? 1 : 0
# 3.265 ns (0 allocations: 0 bytes)
@btime !a ? 1 : 0
# 14.892 ns (0 allocations: 0 bytes)

# ---------------------------------------------------------------------------- #
#                    matrixtable based logiset: benchmarks                     #
# ---------------------------------------------------------------------------- #
model, ds = prepare_dataset(
    Xr, yr;
    model=(;type=:xgboost),
    resample = (type=Holdout, params=(shuffle=true, rng=Xoshiro(1))),
    preprocess=(;train_ratio=0.7, rng=Xoshiro(1)),
    measures=(rms,)
)

dX = DataFrame(ds.X, ds.info.vnames)
mX = Tables.table(ds.X; header=ds.info.vnames)
nX = NamedTuple{Tuple(Symbol.(ds.info.vnames))}(eachcol(ds.X))

@btime dX = DataFrame(ds.X, ds.info.vnames)
# 10.110 μs (61 allocations: 51.47 KiB)
@btime mX = Tables.table(ds.X; header=ds.info.vnames)
# 1.238 μs (12 allocations: 1.14 KiB)
@btime nX = NamedTuple{Tuple(Symbol.(ds.info.vnames))}(eachcol(ds.X))
# 4.411 μs (39 allocations: 3.33 KiB)

dP = PropositionalLogiset(dX)
mP = PropositionalLogiset(mX)
nP = PropositionalLogiset(nX)

@btime dP = PropositionalLogiset(dX);
# 2.196 μs (12 allocations: 608 bytes)
@btime mP = PropositionalLogiset(mX);
# 2.270 μs (24 allocations: 1.17 KiB)
@btime nP = PropositionalLogiset(nX);
# 708.906 ns (4 allocations: 1.17 KiB)


dS = SoleData.scalarlogiset(dX; silent=true, allow_propositional = true);
mS = SoleData.scalarlogiset(mX; silent=true, allow_propositional = true);
nS = SoleData.scalarlogiset(nX; silent=true, allow_propositional = true);

@btime SoleData.scalarlogiset(dX; silent=true, allow_propositional = true);
# 5.397 ms (24826 allocations: 26.35 MiB)
@btime SoleData.scalarlogiset(mX; silent=true, allow_propositional = true);
# 3.655 ms (20284 allocations: 24.25 MiB)
@btime SoleData.scalarlogiset(nX; silent=true, allow_propositional = true);
# 15.115 μs (14 allocations: 1.80 KiB)


# ---------------------------------------------------------------------------- #
model, ds = prepare_dataset(
    Xts, yts;
    model=(;type=:modaldecisiontree),
    resample = (type=Holdout, params=(shuffle=true, rng=Xoshiro(1))),
    preprocess=(;train_ratio=0.7, rng=Xoshiro(1)),
    measures=(rms,)
)

dX = DataFrame(ds.X, ds.info.vnames)
mX = Tables.table(ds.X; header=ds.info.vnames)
nX = NamedTuple{Tuple(Symbol.(ds.info.vnames))}(eachcol(ds.X))

@btime dX = DataFrame(ds.X, ds.info.vnames);
# 24.565 μs (97 allocations: 72.72 KiB)
@btime mX = Tables.table(ds.X; header=ds.info.vnames);
# 2.429 μs (12 allocations: 1.77 KiB)
@btime nX = NamedTuple{Tuple(Symbol.(ds.info.vnames))}(eachcol(ds.X));
# 7.314 μs (63 allocations: 6.09 KiB)

@btime dataframe2dimensional(dX);
# 12.911 ms (96213 allocations: 4.81 MiB)
@btime matrix2dimensional(mX);
# 11.021 ms (83946 allocations: 4.43 MiB)

dS = SoleData.scalarlogiset(dX);
mS = SoleData.scalarlogiset(mX);
nS = SoleData.scalarlogiset(nX);

@btime dS = SoleData.scalarlogiset(dX);
# 70.051 ms (1435747 allocations: 69.10 MiB)
@btime mS = SoleData.scalarlogiset(mX);
# 50.770 ms (1161041 allocations: 63.69 MiB)
@btime nS = SoleData.scalarlogiset(nX);
# 80.911 ms (1050448 allocations: 276.53 MiB)
