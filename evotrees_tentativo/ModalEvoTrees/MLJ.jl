function MMI.fit(model::ModalEvoTypes, verbosity::Int, X, y, w=nothing)
    # X = isa(X, AbstractMatrix) ? Tables.columntable(Tables.table(X)) : Tables.columntable(X)
    # nobs = Tables.DataAPI.nrow(X)
    # fnames = Tables.schema(X).names

    fnames = names(X)
    w = isnothing(w) ? Evo.device_ones(CPU, Float32, nrow(X)) : Vector{Float32}(w)
    fitresult, cache = modal_init_core(model, CPU, X, fnames, y, w, nothing)
  
    while cache[:info][:nrounds] < model.nrounds
        Evo.grow_evotree!(fitresult, cache, model)
    end
    report = (features=cache[:fnames],)
    return fitresult, cache, report
end