using DataFrames
using SoleFeatures
using DataStructures
using HypothesisTests
using Random
using Catch22
using StatsBase
using SimpleCaching
using Dates
using ConfigEnv
using Distributions
using CSV
using Statistics
using PyCall
using MLBase
using SoleData

include("../lib.jl")
include("../datasets/feature-selection.jl")
include("../docs/all.jl")
include("../casile-results/conversion_variable.jl")
include("window.jl")
include("../docs/validate-feature-table.jl")
include("occ.jl")
include("random-select-for-validate.jl")
include("aggr-names-by-windows.jl")

# === Dataset name === #
const LOCAL_PATH = "/home/datasets/PatternRecognition/all_electrodes_and_freqs_ratio"
const LABELS_FILE = joinpath(LOCAL_PATH, "Labels.csv")
const ELECTRODE_NAMES_CSV = joinpath(LOCAL_PATH, "Electrode_names.csv")
const ELECTRODE_NAMES_TXT = joinpath(LOCAL_PATH, "Electrode_names.txt")

# === Cache directory === #

const cache_dir = "./EEG_Casile/cache"

# === FUNCTION === #

Base.nameof(f::SuperFeature) = getname(f) # wrap for Catch22

# === Load electrode names ===
function load_electrode_names()
	electrode_names = CSV.read(ELECTRODE_NAMES_CSV, DataFrame, header = 1)
	electrode_names = electrode_names[!, "Electrode_name"]
	return electrode_names
end

# === Load electrode indices ===
function load_electrode_indices()
	electrode_indices = CSV.read(ELECTRODE_NAMES_CSV, DataFrame, header = 1)
	electrode_indices = electrode_indices[!, "Index"]
	return electrode_indices
end

# ================== PREPARE DATASET ================== #
@info "PREPARE DATASET"

combined_df = @scache "ds" cache_dir loaddataset(LOCAL_PATH; onlywithlabels = [["trial_type" => ["1"]]])
df = combined_df[1]
labels_df = CSV.read(LABELS_FILE, DataFrame)
trial_type_1_filter = labels_df.trial_type .== 1
ylabels = labels_df[trial_type_1_filter, :]
y = (ylabels.condition .== "avatar") .| (ylabels.condition .== "observation")

#y = labels_df[!, "condition"] .== "action"
#df = df[condition_filter, :]
#y = labels_df[condition_filter, "trial_type"]


# ================== FEATURE SELECTION ================== #

ws = [SoleFeatures.Experimental.FixedNumMovingWindows(6, 0.05)...]
ms = [catch22..., minimum, maximum, mean]
fs_methods = [
	( # STEP 1: unsupervised variance-based filter
		selector = SoleFeatures.VarianceFilter(SoleFeatures.IdentityLimiter()),
		limiter = PercentageLimiter(0.025),
	),
	( # STEP 2: supervised Mutual Information filter
		selector = SoleFeatures.MutualInformationClassif(SoleFeatures.IdentityLimiter()),
		limiter = PercentageLimiter(0.01),
	),
	( # STEP 3: group results by variable
		selector = IdentityFilter(),
		limiter = SoleFeatures.IdentityLimiter(),
	),
]

@info "FEATURE SELECTION"
X, fs_mid_results = feature_selection(df, y, ex_windows = ws, ex_measures = ms, fs_methods = fs_methods, normalize = true)

unsup_score = fs_mid_results[2][1].score
unsup_score_mv = sort(moving_window_filter(unsup_score, f = mean, nwindows = 116, relative_overlap = 0.0), rev = true)
unsup_names = collect(1:116)
sup_score = fs_mid_results[2][2].score
sup_score_mv = sort(moving_window_filter(sup_score, f = mean, nwindows = 87, relative_overlap = 0.0), rev = true)
sup_names = collect(1:87)
nsX = [string.(split(i, "@@@")[1]) for i in names(X)]
wsX = [string.(split(i, "@@@")[2]) for i in names(X)]
wsX = string.([split(s, '(')[1] for s in wsX])
msX = [string.(split(i, "@@@")[3]) for i in names(X)]
aggr_names = aggr_names_by_windows(X)
electrode_names = load_electrode_names()
selected_el = replace.(conversion_electrode_frequency_variable(collect(1:40), electrode_names; variables = parse.(Int, nsX)), r"-.*" => "")
selected_freq = parse.(Int, replace.(conversion_electrode_frequency_variable(collect(1:40), electrode_names; variables = parse.(Int, nsX)), r".*-" => ""))
frequencies_keys = string.(collect(1:40))
for i in 1:length(aggr_names)
	aggr_names[i] = replace.(conversion_electrode_frequency_variable(collect(1:40), electrode_names; variables = parse.(Int, aggr_names[i])), r"-.*" => "")
end
selected_indices = fs_mid_results[2][2].indices
sort_selected_indicies = sortperm(sup_score[selected_indices], rev = true)
selected_names = names(X[:, sort_selected_indicies])
indicies_4_validate = random_select_for_validate(selected_names, 4)
validate_score = validate_features(X[:, vcat(indicies_4_validate...)], y)
validate_score = random_select_for_validate(validate_score[1], 4)
wrap_eeg_feature_selection("avatar_vs_observation.tex", "unsupfile", "supfile", unsup_score_mv, unsup_names, sup_score_mv, sup_names, selected_el, selected_freq, wsX, msX, aggr_names, validate_score; frequencies_keys = frequencies_keys)