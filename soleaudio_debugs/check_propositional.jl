using DataFrames, JLD2
using SoleAudio, Random, Audio911
using StatsBase, Catch22
# using Plots

# -------------------------------------------------------------------------- #
#                       experiment specific parameters                       #
# -------------------------------------------------------------------------- #
wav_path ="/home/paso/Documents/Aclai/Datasets/health_recognition/Respiratory_Sound_Database/audio_partitioned"
csv_path = "/home/paso/Documents/Aclai/Datasets/health_recognition/Respiratory_Sound_Database"
csv_file = csv_path * "/" * "patient_diagnosis_partitioned.csv"

classes = :Pneumonia

jld2_file = string("respiratory_", classes)

if classes == :Pneumonia
    classes_dict = Dict{String,String}(
        "Pneumonia" => "sick",
        "Healthy" => "healthy",
    )
end

header = true
id_labels = :filename
label_labels = :diagnosis

featset = (:mel, :mfcc, :spectrals)

audioparams = let sr = 8000
    (
        sr = sr,
        norm = true,
        speech_detect = false,
        nfft = 256,
        mel_scale = :mel_htk, # :mel_htk, :mel_slaney, :erb, :bark, :semitones, :tuned_semitones
        mel_nbands = 26,
        mfcc_ncoeffs = 13,
        mel_freqrange = (250, round(Int, sr / 2)),
    )
end

analysisparams = (propositional=true, modal=false,)

min_length = 17500
min_samples = 132

features = :catch9
# features = :minmax
# features = :custom

# modal analysis
nwindows = 20
relative_overlap = 0.05

# partitioning
train_ratio = 0.8
train_seed = 1

rng = Random.MersenneTwister(train_seed)
Random.seed!(train_seed)

# -------------------------------------------------------------------------- #
#                         interface: get_raw_audio                           #
# -------------------------------------------------------------------------- #
sort_before_merge = true
id_df = :filename
label_df = :label

df = collect_audio_from_folder(wav_path; audioparams=audioparams)
labels = isnothing(csv_file) ? 
    collect_classes(df, classes_dict; classes_func=classes_func) : 
    collect_classes(csv_file, classes_dict; id_labels=id_labels, label_labels=label_labels, header=header)
merge_df_labels!(df, labels; sort_before_merge=sort_before_merge, id_df=id_df, id_labels=id_labels, label_df=label_df, label_labels=label_labels)


# -------------------------------------------------------------------------- #
#                           interface: get_rules                             #
# -------------------------------------------------------------------------- #
splitlabel = :label
lengthlabel = :length
audiolabel = :audio
source_label = :audio 
r_select = r"\e\[\d+m(.*?)\e\[0m"
r_split = r"(\e\[[\d;]*m)(.*?)(\e\[0m)"

sort_df!(df, :length; rev=true)
df = trimlength_df(df, splitlabel, lengthlabel, audiolabel; min_length=min_length, min_samples=min_samples, sr=audioparams.sr)

# audio test
# for i in eachrow(df)
#     audio_test_path = "/home/paso/Documents/Aclai/Julia_additional_files/soleaudio_debugs/audio_test"
#     audio_test_path = string(audio_test_path, "/", i.filename, ".wav")
#     # println(audio_test_path)
#     save_audio(Audio(i.audio, audioparams.sr), audio_test_path)
# end

# X, y, variable_names = afe(df, featset, audioparams)

function set_audio_length!(df::DataFrame, audiolabel::Symbol, lengthlabel::Symbol)
    insertcols!(df, lengthlabel => size.(df[:, audiolabel], 1))
end


freq = round.(Int, audio_features(df[1, source_label], audioparams.sr; featset=(:get_only_freqs), audioparams...))
variable_names = vnames_builder(featset, audioparams; freq=freq)

X = DataFrame([name => Vector{Float64}[] for name in [match(r_select, v)[1] for v in variable_names]])

row = df[1, :]
audiofeats = collect(eachcol(audio_features(row[source_label], audioparams.sr; featset=featset, audioparams...)))
push!(X, audiofeats)

# test audio features
println(source_label)
println(length(row[source_label]))
mel = audio_features(row[source_label], audioparams.sr; featset=(:mel), audioparams...)

# return X, CategoricalArray(df[!, label]), variable_names

# -------------------------------------------------------------------------- #
#                                propositional                               #
# -------------------------------------------------------------------------- #
propositional_feature_dict = Dict(
    :catch9 => [
        maximum,
        minimum,
        StatsBase.mean,
        median,
        std,
        Catch22.SB_BinaryStats_mean_longstretch1,
        Catch22.SB_BinaryStats_diff_longstretch0,
        Catch22.SB_MotifThree_quantile_hh,
        Catch22.SB_TransitionMatrix_3ac_sumdiagcov,
    ],
)

f_dict_string = Dict(
    maximum => "max",
    minimum => "min",
    StatsBase.mean => "mean",
    median => "med",
    std => "std",
    Catch22.SB_BinaryStats_mean_longstretch1 => "mean_ls",
    Catch22.SB_BinaryStats_diff_longstretch0 => "diff_ls",
    Catch22.SB_MotifThree_quantile_hh => "qnt",
    Catch22.SB_TransitionMatrix_3ac_sumdiagcov => "sdiag",
    Catch22.DN_HistogramMode_5 => "hist",
    Catch22.CO_f1ecac => "f1ecac",
    Catch22.CO_HistogramAMI_even_2_5 => "hist_even",
)

metaconditions = get(propositional_feature_dict, features) do
    error("Unknown set of features: $features.")
end

p_variable_names = [
    string(m[1], f_dict_string[j], "(", m[2], ")", m[3])
    for j in metaconditions
    for m in [match(r_split, i) for i in variable_names]
]

X_propos = DataFrame([name => Float64[] for name in [match(r_select, v)[1] for v in p_variable_names]])
push!(X_propos, vcat([vcat([map(func, Array(row)) for func in metaconditions]...) for row in eachrow(X)])...)

# X_train, y_train, X_test, y_test = partitioning(X_propos, y; train_ratio=train_ratio, rng=rng)

# @info("Propositional analysis: train model...")
# learned_dt_tree = begin
#     model = Tree(; max_depth=-1, )
#     mach = machine(model, X_train, y_train) |> fit!
#     fitted_params(mach).tree
# end

# sole_dt = solemodel(learned_dt_tree)
# apply!(sole_dt, X_test, y_test)
# # printmodel(sole_dt; show_metrics = true, variable_names_map = variable_names)

# sole_dt
