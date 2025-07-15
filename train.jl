using Test
using MLJ, SoleXplorer
using DataFrames, Random
using SoleData
const SX = SoleXplorer

Xc, yc = @load_iris
Xc = DataFrame(Xc)

Xr, yr = @load_boston
Xr = DataFrame(Xr)

Xts, yts = SoleData.load_arff_dataset("NATOPS")

modelc  = setup_dataset(Xc, yc)
modelr  = setup_dataset(Xr, yr; model=XGBoostRegressor())
modelts = setup_dataset(Xts, yts; model=ModalDecisionTree())

model = modelc

########################### TRAIN ################################

get_X(model::SX.AbstractDataSet)::DataFrame = model.mach.args[1].data
get_y(model::SX.AbstractDataSet)::Vector = model.mach.args[2].data

n_folds     = length(model.pidxs)
# model.model = Vector{AbstractModel}(undef, n_folds)

i = 1
train   = ds.tt[i].train
test    = ds.tt[i].test
X_test  = DataFrame((@views ds.X[test, :]), ds.info.vnames)
y_test  = @views ds.y[test]

@btime begin
    X = get_X(model)
    y = get_y(model)
    train = SX.get_train(model.pidxs[1])
    test  = SX.get_test(model.pidxs[1])
    X_train = X[train, :]
    y_train = y[train]
    X_test = X[test, :]
    y_test = y[test]
end
# 7.071 μs (220 allocations: 15.39 KiB)

@btime begin
    X = get_X(model)
    y = get_y(model)
    train = SX.get_train(model.pidxs[1])
    test  = SX.get_test(model.pidxs[1])
    X_train = @views X[train, :]
    y_train = @views y[train]
    X_test = @views X[test, :]
    y_test = @views y[test]
end
# 2.384 μs (156 allocations: 6.12 KiB)

@btime begin
    @inbounds @views for i in 1:n_folds
        X, y = SX.get_X(model), SX.get_y(model)
        train, test = SX.get_train(model.pidxs[i]), SX.get_test(model.pidxs[i])
        # X_train , y_train = X[train, :], y[train]
        X_test, y_test = X[test, :], y[test]
        # MLJ.fit!(mach, rows=train, verbosity=0)
        # model.model[i] = apply(mach, X_test, y_test)
    end
end
# 2.364 μs (156 allocations: 6.09 KiB)

@btime begin
    @views for i in 1:n_folds
        # X, y = get_X(model), get_y(model)
        train, test = SX.get_train(model.pidxs[i]), SX.get_test(model.pidxs[i])
        # X_train , y_train = X[train, :], y[train]
        X_test, y_test = get_X(model)[test, :], get_y(model)[test]
        MLJ.fit!(model.mach, rows=train, verbosity=0)
        # model.model[i] = apply(mach, X_test, y_test)
    end
end

@btime SX.apply(model.mach, X_test, y_test)
# 256.331 μs (2252 allocations: 181.03 KiB)

@btime SX.apply(model, X_test, y_test)
# 257.186 μs (2253 allocations: 181.09 KiB)

@btime model.mach

@btime begin
    mach = model.mach
    mach
end

a = train_test(modelc)

@btime train_test(modelc)
# 197.799 μs (1931 allocations: 166.45 KiB)

@btime train_test(modelc)

# TODO this can be parallelizable
@inbounds for i in 1:n_folds
    train   = ds.tt[i].train
    test    = ds.tt[i].test
    X_test  = DataFrame((@views ds.X[test, :]), ds.info.vnames)
    y_test  = @views ds.y[test]

    # xgboost reg:squarederror default base_score is mean(y_train)
    if model.setup.type == MLJXGBoostInterface.XGBoostRegressor 
        base_score = get_base_score(model) == -Inf ? mean(ds.y[train]) : 0.5
        get_tuning(model) === false ?
            (mach.model.base_score = base_score) :
            (mach.model.model.base_score = base_score)
        MLJ.fit!(mach, rows=train, verbosity=0)
        model.model[i] = apply(mach, X_test, y_test, base_score)
    else
        MLJ.fit!(mach, rows=train, verbosity=0)
        model.model[i] = apply(mach, X_test, y_test)
    end
end


# ---------------------------------------------------------------------------- #
#                                   get model                                  #
# ---------------------------------------------------------------------------- #
function get_predictor(model::AbstractModelSetup)::MLJ.Model
    predictor = model.type(;model.params...)

    model.tuning === false || begin
        ranges = [r(predictor) for r in model.tuning.ranges]

        predictor = MLJ.TunedModel(; 
            model=predictor, 
            tuning=model.tuning.method.type(;model.tuning.method.params...),
            range=ranges, 
            model.tuning.params...
        )
    end

    return predictor
end

# ---------------------------------------------------------------------------- #
#                                     train                                    #
# ---------------------------------------------------------------------------- #
function _train_machine!(model::AbstractModelset, ds::AbstractDataset)::MLJ.Machine
    # Early stopping is a regularization technique in XGBoost that prevents overfitting by monitoring model performance 
    # on a validation dataset and stopping training when performance no longer improves.
    if haskey(model.setup.params, :watchlist) && model.setup.params.watchlist == makewatchlist
        model.setup.params = merge(model.setup.params, (watchlist = makewatchlist(ds),))
    end

    model.type = get_predictor(model.setup)

    MLJ.machine(
        model.type,
        MLJ.table(ds.X; names=ds.info.vnames),
        ds.y
    )
end

# ---------------------------------------------------------------------------- #
#                                     test                                     #
# ---------------------------------------------------------------------------- #
function _test_model!(model::DataSet)
    n_folds     = length(ds.tt)
    model.model = Vector{AbstractModel}(undef, n_folds)

    # TODO this can be parallelizable
    @inbounds for i in 1:n_folds
        train   = ds.tt[i].train
        test    = ds.tt[i].test
        X_test  = DataFrame((@views ds.X[test, :]), ds.info.vnames)
        y_test  = @views ds.y[test]

        # xgboost reg:squarederror default base_score is mean(y_train)
        if model.setup.type == MLJXGBoostInterface.XGBoostRegressor 
            base_score = get_base_score(model) == -Inf ? mean(ds.y[train]) : 0.5
            get_tuning(model) === false ?
                (mach.model.base_score = base_score) :
                (mach.model.model.base_score = base_score)
            MLJ.fit!(mach, rows=train, verbosity=0)
            model.model[i] = apply(mach, X_test, y_test, base_score)
        else
            MLJ.fit!(mach, rows=train, verbosity=0)
            model.model[i] = apply(mach, X_test, y_test)
        end
    end
end

