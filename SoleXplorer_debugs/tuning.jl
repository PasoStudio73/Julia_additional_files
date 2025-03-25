using Sole
using SoleXplorer
using StatsBase
using Random
using MLJ
using MLJBase: Probabilistic, ParamRange
using MLJTuning
using MLJDecisionTreeInterface
using MLJParticleSwarmOptimization: ParticleSwarm, AdaptiveParticleSwarm
using Distributions
using TreeParzen

# ---------------------------------------------------------------------------- #
#                           available tuning methods                           #
# ---------------------------------------------------------------------------- #
const AVAIL_TUNINGS = Dict(
    :grid => (
        method = MLJ.Grid,
        params = (;
            goal=nothing,
            resolution=10,
            shuffle=true,
            rng=Random.TaskLocalRNG()
        )
    ),

    :randomsearch => (
        method = MLJ.RandomSearch,
        params = (;
            bounded=Distributions.Uniform,
            positive_unbounded=Distributions.Gamma,
            other=Distributions.Normal,
            rng=Random.TaskLocalRNG()
        )
    ),

    :latinhypercube => (
        method = MLJ.LatinHypercube,
        params = (;
            gens=2,
            popsize=100,
            ntour=2,
            ptour=0.8,
            interSampleWeight=1.0,
            ae_power=2,
            periodic_ae=false,
            rng=Random.TaskLocalRNG()
        )
    ),

    :treeparzen => (
        method = MLJTreeParzenTuning,
        params = (;
            config=TreeParzen.Configuration.Config(0.25, 25, 24, 20, 1.0), 
            max_simultaneous_draws=1
        )
    ),

    :particleswarm => (
        method = ParticleSwarm,
        params = (;
            n_particles=3, 
            w=1.0, 
            c1=2.0, 
            c2=2.0, 
            prob_shift=0.25, 
            rng=Random.TaskLocalRNG()
        )
    ),

    :adaptiveparticleswarm => (
        method = AdaptiveParticleSwarm,
        params = (;
            n_particles=3, 
            c1=2.0, 
            c2=2.0, 
            prob_shift=0.25, 
            rng=Random.TaskLocalRNG()
        )
    ),
)

const TUNEDMODEL_PARAMS = (;
    resampling=Holdout(),
    measure=LogLoss(tol = 2.22045e-16),
    weights=nothing,
    gino="ulo",
    class_weights=nothing,
    repeats=1,
    operation=nothing,
    selection_heuristic= MLJTuning.NaiveSelection(nothing),
    n=nothing,
    train_best=true,
    acceleration=default_resource(),
    acceleration_resampling=CPU1(),
    check_measure=true,
    cache=true
)

# ---------------------------------------------------------------------------- #
#                               get tuning model                               #
# ---------------------------------------------------------------------------- #
function get_tuning(tuning_method::Symbol; kwargs...)
    !haskey(AVAIL_TUNINGS, tuning_method) && throw(ArgumentError("Method $tuning_method not found in tuning models. Valid options are: $(keys(AVAIL_TUNINGS))"))

    params = AVAIL_TUNINGS[tuning_method].params
    valid_kwargs = filter(kv -> kv.first in keys(params), kwargs)
    
    AVAIL_TUNINGS[tuning_method].method(; merge(params, valid_kwargs)...)
end

function get_tuning(model::T, tuning::S; 
    ranges::Union{Nothing, S, AbstractVector{S}}=nothing,
    kwargs...
) where {T<:SoleXplorer.Modelset, S<:MLJTuning.TuningStrategy}
    _ranges = isnothing(ranges) ? model.ranges : ranges
    valid_ranges = [f(model.classifier) for f in _ranges]
    valid_kwargs = filter(kv -> kv.first in keys(TUNEDMODEL_PARAMS), kwargs)
println(valid_kwargs)
    MLJ.TunedModel(; model=model.classifier, tuning=tuning, ranges=valid_ranges, valid_kwargs...)
end

function get_tuning(model::T, tuning_method::Symbol;
    ranges::Union{Nothing, S, AbstractVector{S}}=nothing,
    kwargs...
) where {T<:SoleXplorer.Modelset, S<:Function}
println("ciao")
    tuning = get_tuning(tuning_method; kwargs...)
    get_tuning(model, tuning; ranges=ranges, kwargs...)
end


# X, y = SoleData.load_arff_dataset("NATOPS");
rng = Random.Xoshiro(1)
model_name = :decision_tree
tuning_method = :latinhypercube

@info "Tuning Example"
model = SoleXplorer.get_model(model_name)
tuning = get_tuning(tuning_method)
tuning = get_tuning(model, tuning_method)