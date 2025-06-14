using Catch22

isdefined(Main, :Catch22) && (Base.nameof(f::SuperFeature) = getname(f)) # wrap for Catch22

using Test
using Sole
using SoleXplorer, Catch22
using Random, StatsBase, JLD2, DataFrames

X, y       = SoleData.load_arff_dataset("NATOPS")
train_seed = 11
rng        = Random.Xoshiro(train_seed)
Random.seed!(train_seed)

# downsize dataset
num_cols_to_sample = 10
num_rows_to_sample = 50
chosen_cols = StatsBase.sample(rng, 1:size(X, 2), num_cols_to_sample; replace=false)
chosen_rows = StatsBase.sample(rng, 1:size(X, 1), num_rows_to_sample; replace=false)

X = X[chosen_rows, chosen_cols]
y = y[chosen_rows]

# ================== PREPARE VARIABLES, WINDOWS, MEASURES ==================
@info "PREPARE VARIABLES, WINDOWS, MEASURES"

struct FixedNumMovingWindows
    nwindows::Int
    reloverlap::Float64

    function FixedNumMovingWindows(nwindows::Integer, reloverlap::AbstractFloat)
        nwindows <= 0 && throw(DomainError(nwindows, "Must be greater than 0"))
        !(0.0 <= reloverlap <= 1.0) &&
            throw(DomainError(reloverlap, "Must be within 0.0 and 1.0"))
        return new(nwindows, reloverlap)
    end
end

# prepare awmds
vars = Symbol.(names(X))
fnmw = FixedNumMovingWindows(3, 0.25)
measures = [catch22..., minimum, maximum, StatsBase.mean]

function build_awmds(
    vars::AbstractVector{Symbol},
    fnmw,
    measures::AbstractVector{<:Function}
)
    return Iterators.product(vars, fnmw, measures)
end

# function build_awmds(
#     vars::AbstractVector{Symbol},
#     mw,
#     measures::AbstractVector{<:Function}
# )
#     return build_awmds(vars, [mw...], measures)
# end


awmds = build_awmds(vars, fnmw, measures)

################################################################################
############################### FEATURE SELECTION ##############################
################################################################################

# feature_selection_method = :feature
feature_selection_method = :variable
# feature_selection_method = :none

feature_selectors = [
    ( # STEP 1: unsupervised variance-based filter
        selector = SoleFeatures.VarianceFilter(SoleFeatures.IdentityLimiter()),
        limiter = PercentageLimiter(0.5),
    ),
    ( # STEP 2: supervised Mutual Information filter
        selector = SoleFeatures.MutualInformationClassif(SoleFeatures.IdentityLimiter()),
        limiter = PercentageLimiter(0.1),
    ),
    ( # STEP 3: group results by variable
        selector = IdentityFilter(),
        limiter = SoleFeatures.IdentityLimiter(),
    ),
]

feature_selection_aggrby = nothing

feature_selection_validation_n_test = 10
feature_selection_validation_seed = 5