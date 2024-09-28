using SoleAudio
using DataFrames
using JLD2
using StatsBase

using CSV
using Tables
# using Statistics
# using Clustering
# using MLJ

#--------------------------------------------------------------------------------------#
#                              program workflow variables                              #
#--------------------------------------------------------------------------------------#
build_dataset = false
dataset_verification = false
create_csv = false

#--------------------------------------------------------------------------------------#
#                              audio features extractions                              #
#--------------------------------------------------------------------------------------#
function audio_features_collect(;
    X::DataFrame,
    source_path::String,
    afe_cols::Int64,
    sr::Int64,
    stft_length::Int64,
    mel_bands::Int64
)
    cd(source_path)
    for i in readdir()
        # load wav file
        x, sr = load_audio(i, sr=sr)

        # cast
        x = Float64.(x)
        # normalize
        x = normalize_audio(x)
        # detect speech
        x = speech_detector(x, sr)

        if size(x, 1) > 0 # if speech is present
            # audio feature extraction
            afe = audio_features_extractor(
                x,
                dataset=:speaker_recognition,
                sr=sr,

                # fft
                stft_length=stft_length,
                window_type=[:hann, :periodic],
                window_length=stft_length,
                overlap_length=Int(round(stft_length * 0.500)),
                window_norm=:true,

                # spectrum
                freq_range=Int[0, sr/2],
                spectrum_type=:power,

                # mel
                mel_style=:htk,
                mel_bands=mel_bands,
                filterbank_design_domain=:linear,
                filterbank_normalization=:bandwidth,
                frequency_scale=:mel,

                # mfcc
                mfcc_coeffs=13,
                normalization_type=:standard,
                rectification=:log,
                log_energy_source=:standard,
                log_energy_pos=:none,
                delta_window_length=9,
                delta_matrix=:standard,

                # spectral
                spectral_spectrum=:linear
            )
            push!(X, (vcat.(afe[j, :] for j in 1:afe_cols)..., split(i, "_")[1]))
        end
    end
end

#--------------------------------------------------------------------------------------#
#                                  dataset building                                    #
#--------------------------------------------------------------------------------------#
function create_dataset()
    # dataset:
    # Warden P. "Speech Commands: A public dataset for single-word speech recognition", 2017. 
    # Available from https://storage.googleapis.com/download.tensorflow.org/data/speech_commands_v0.01.tar.gz. 
    # Copyright Google 2017. The Speech Commands Dataset is licensed under the Creative Commons Attribution 4.0 license, 
    # available here: https://creativecommons.org/licenses/by/4.0/legalcode.
    source_path = [
        "/home/riccardopasini/Documents/Aclai/Datasets/google_speech/train/bed",
        "/home/riccardopasini/Documents/Aclai/Datasets/google_speech/train/bird",
        "/home/riccardopasini/Documents/Aclai/Datasets/google_speech/train/cat",
        "/home/riccardopasini/Documents/Aclai/Datasets/google_speech/train/dog",
        "/home/riccardopasini/Documents/Aclai/Datasets/google_speech/train/down",
        "/home/riccardopasini/Documents/Aclai/Datasets/google_speech/train/bed",
        "/home/riccardopasini/Documents/Aclai/Datasets/google_speech/train/eight",
        "/home/riccardopasini/Documents/Aclai/Datasets/google_speech/train/five",
        "/home/riccardopasini/Documents/Aclai/Datasets/google_speech/train/four",
        "/home/riccardopasini/Documents/Aclai/Datasets/google_speech/train/go",
        "/home/riccardopasini/Documents/Aclai/Datasets/google_speech/train/happy",
        "/home/riccardopasini/Documents/Aclai/Datasets/google_speech/train/house",
        "/home/riccardopasini/Documents/Aclai/Datasets/google_speech/train/left",
        "/home/riccardopasini/Documents/Aclai/Datasets/google_speech/train/marvin",
        "/home/riccardopasini/Documents/Aclai/Datasets/google_speech/train/nine",
        "/home/riccardopasini/Documents/Aclai/Datasets/google_speech/train/no",
        "/home/riccardopasini/Documents/Aclai/Datasets/google_speech/train/off",
        "/home/riccardopasini/Documents/Aclai/Datasets/google_speech/train/on",
        "/home/riccardopasini/Documents/Aclai/Datasets/google_speech/train/one",
        "/home/riccardopasini/Documents/Aclai/Datasets/google_speech/train/right",
        "/home/riccardopasini/Documents/Aclai/Datasets/google_speech/train/seven",
        "/home/riccardopasini/Documents/Aclai/Datasets/google_speech/train/sheila",
        "/home/riccardopasini/Documents/Aclai/Datasets/google_speech/train/six",
        "/home/riccardopasini/Documents/Aclai/Datasets/google_speech/train/stop",
        "/home/riccardopasini/Documents/Aclai/Datasets/google_speech/train/three",
        "/home/riccardopasini/Documents/Aclai/Datasets/google_speech/train/tree",
        "/home/riccardopasini/Documents/Aclai/Datasets/google_speech/train/two",
        "/home/riccardopasini/Documents/Aclai/Datasets/google_speech/train/up",
        "/home/riccardopasini/Documents/Aclai/Datasets/google_speech/train/wow",
        "/home/riccardopasini/Documents/Aclai/Datasets/google_speech/train/yes",
        "/home/riccardopasini/Documents/Aclai/Datasets/google_speech/train/zero",
    ]

    @info "Starting building dataset..."

    #--------------------------------------------------------------------------------------#
    #                                 dataset parameters                                   #
    #--------------------------------------------------------------------------------------#
    ds_type = :speaker_recognition
    sr = 8000
    afe_cols = 13
    stft_length = 256
    mel_bands = 26

    jld2_file = "/home/riccardopasini/results/speech/speaker_recognition/ds_spk_rec_train.jld2"

    #--------------------------------------------------------------------------------------#
    #                                 starting extraction                                  #
    #--------------------------------------------------------------------------------------#
    # initialize dataframe
    X = DataFrame()
    for i = 1:afe_cols
        colname = "a$i"
        X[!, colname] = Vector{Float64}[]
    end
    X[!, "id"] = String[] # salvo l'id del malcapitato

    for i in source_path
        println("working on dataset:")
        println(i)
        audio_features_collect(
            X=X,
            source_path=i,
            afe_cols=afe_cols,
            sr=sr,
            stft_length=stft_length,
            mel_bands=mel_bands
        )
    end

    @info "Save jld2 files..."

    df = X[:, 1:end-1]
    y = X[:, end]
    dataframe_validated = (df, y)
    jldsave(jld2_file, true; dataframe_validated)
    println("Dataset: ", jld2_file, " stored.")
end

#--------------------------------------------------------------------------------------#
#                                 csv means and vars                                   #
#--------------------------------------------------------------------------------------#
function csv_mean_var(jld2_file, afe_cols)
    csv_file = "/home/riccardopasini/results/speech/speaker_recognition/mean_variance.csv"

    labels_db = []
    for i in 1:4
        push!(labels_db, "m$i")
    end
    for i in 1:4
        push!(labels_db, "v$i")
    end

    d = jldopen(jld2_file)
    df, Y = d["dataframe_validated"]

    # reducing time series with mean
    # keeping only MFCC from coefficients 2 to 5 for debug
    Xm = mean.(select(df, Between(:a2, :a5)))

    Xm[!, :label] = Y

    @assert Xm isa DataFrame

    sub_df = DataFrames.groupby(Xm, :label)

    sub_Xm = []
    for i in sub_df
        tmp = select(i, Between(:a2, :a5))
        push!(sub_Xm, [mean.(eachcol(tmp))..., var.(eachcol(tmp))...])
    end

    open(csv_file, "w") do f
        CSV.write(f, [], writeheader=true, header=labels_db)
    end

    tmp = Array{Float64,2}(undef, size(sub_Xm, 1), 8)
    for i in 1:size(sub_Xm, 1)
        for j in 1:8
            tmp[i, j] = sub_Xm[i][j]
        end
    end

    CSV.write(
        csv_file,
        Tables.table(tmp),
        writeheader=false,
        append=true
    )
end

#--------------------------------------------------------------------------------------#
#                                dataset verification                                  #
#--------------------------------------------------------------------------------------#
function dataset_check()
    #--------------------------------------------------------------------------------------#
    #                                 dataset parameters                                   #
    #--------------------------------------------------------------------------------------#
    jld2_file = "/home/riccardopasini/results/speech/speaker_recognition/ds_spk_rec_train.jld2"

    #TODO
end
#--------------------------------------------------------------------------------------#
#                                        main                                          #
#--------------------------------------------------------------------------------------#
if build_dataset
    create_dataset()
end

jld2_file = "/home/riccardopasini/results/speech/speaker_recognition/ds_spk_rec_train.jld2"

if dataset_verification
    dataset_check() #TODO
    error("not yet implemented.")
end

if create_csv
    csv_mean_var(jld2_file, afe_cols)
end
#--------------------------------------------------------------------------------------#
#                                         setup                                        #
#--------------------------------------------------------------------------------------#
# choosen mfcc coefficients
cols = [:a2, :a3, :a4, :a5]
# tolerance for each mfcc coefficients
tolerance = [2.0, 1.2, 0.8, 0.4]

jld2_file = "/home/riccardopasini/results/speech/speaker_recognition/ds_spk_rec_train.jld2"

d = jldopen(jld2_file)
df, Y = d["dataframe_validated"]

@assert df isa DataFrame

# reducing time series with mean
# keeping only MFCC from coefficients 2 to 5 for debug
Xm = mean.(select(df, cols...))

# TODO normalize dataset

#--------------------------------------------------------------------------------------#
#                                  defining centroids                                  #
#--------------------------------------------------------------------------------------#
@info "Creating centroids..."

# static new clusters
new_found_cluster = DataFrame()
new_found_cluster[!, "id"] = Int64[]
for i = cols
    colname = "$i"
    new_found_cluster[!, colname] = Float64[]
end

# dataframe with all samples with id
id_Xm = DataFrame()
id_Xm[!, "id"] = Int64[]
for i = cols
    colname = "$i"
    id_Xm[!, colname] = Float64[]
end

counter_id = 1
for i in eachrow(Xm)

    # first element
    if isempty(new_found_cluster)
        push!(id_Xm, hcat(counter_id, i...))
        push!(new_found_cluster, hcat(counter_id, i...))
        counter_id += 1
    end

    tmp_clusters = [] # list of clusters matched

    for j in eachrow(new_found_cluster)
        pass = true # coefficients verification
        # check if every coefficient stays in tolerance
        for k in 1:length(cols)
            if pass && !(j[k+1] - tolerance[k] < i[k] < j[k+1] + tolerance[k])
                pass = false # at least one coefficient is out of tolerance
            end
        end

        if pass # the sample match with choosen cluster
            push!(tmp_clusters, j)
        end
    end

    # the sample doesn't match any existing cluster, create new one
    if isempty(tmp_clusters)
        push!(id_Xm, hcat(counter_id, i...))
        push!(new_found_cluster, hcat(counter_id, i...))
        global counter_id += 1

        # there's only one entries
    elseif size(tmp_clusters) == 1
        push!(id_Xm, hcat(tmp_clusters.id, i...))
        # j =  findall(x -> x == tmp_clusters.id, Xm.id)
        # new_mean = mean.(eachcol(j))

        # multiple choices, choose the closest
    else
        closest_c = Inf
        choosen_cluster = 0
        distance = 0
        for j in tmp_clusters
            for k in 1:length(cols)
                distance += (j[k+1] - i[k])^2
            end
            distance = sqrt(distance)
            if distance < closest_c
                closest_c = distance
                choosen_cluster = j.id
            end
        end

        # save
        if choosen_cluster != 0
            push!(id_Xm, hcat(choosen_cluster, i...))
        else
            error("Something went wrong with choosen_cluster.")
        end
    end
end

println(counter_id, " cluster defined.")

#--------------------------------------------------------------------------------------#
#                                 pseudo recursive part                                #
#--------------------------------------------------------------------------------------#
@info "Calculating new means..."

final_Xm = DataFrame()
n_iterate = 2
for iterate in 1:n_iterate
    centroid = DataFrames.groupby(id_Xm, :id)

    # cluster dataframe
    clusters = DataFrame()
    clusters[!, "id"] = Int64[]
    for i = cols
        colname = "$i"
        clusters[!, colname] = Float64[]
    end

    for i in centroid
        push!(clusters, mean.(eachcol(i)))
    end

    #--------------------------------------------------------------------------------------#
    @info "Creating centroids..."

    # dataframe with all samples with id
    iterate_Xm = DataFrame()
    iterate_Xm[!, "id"] = Int64[]
    for i = cols
        colname = "$i"
        iterate_Xm[!, colname] = Float64[]
    end

    iterate_id = findmax(clusters.id)[1] + 1
    for i in eachrow(Xm)

        tmp_clusters = [] # list of clusters matched

        for j in eachrow(clusters)
            pass = true # coefficients verification
            # check if every coefficient stays in tolerance
            for k in 1:length(cols)
                if pass && !(j[k+1] - tolerance[k] < i[k] < j[k+1] + tolerance[k])
                    pass = false # at least one coefficient is out of tolerance
                end
            end

            if pass # the sample match with choosen cluster
                push!(tmp_clusters, j)
            end
        end

        # the sample doesn't match any existing cluster
        if isempty(tmp_clusters)
            push!(iterate_Xm, hcat(iterate_id, i...))
            iterate_id += 1

            # there's only one entries
        elseif size(tmp_clusters) == 1
            push!(iterate_Xm, hcat(tmp_clusters.id, i...))

            # multiple choices, choose the closest
        else
            closest_c = Inf
            choosen_cluster = 0
            distance = 0
            for j in tmp_clusters
                for k in 1:length(cols)
                    distance += (j[k+1] - i[k])^2
                end
                distance = sqrt(distance)
                if distance < closest_c
                    closest_c = distance
                    choosen_cluster = j.id
                end
            end

            # save
            if choosen_cluster != 0
                push!(iterate_Xm, hcat(choosen_cluster, i...))
            else
                error("Something went wrong with choosen_cluster.")
            end
        end
    end

    println(iterate_id, " cluster defined.")
    println("Process ", iterate, " completed.")

    if iterate == n_iterate
        final_Xm = iterate_Xm
    end
end
#--------------------------------------------------------------------------------------#
#                               calculate centroids means                              #
#--------------------------------------------------------------------------------------#
@info "Calculating new means..."

centroid = DataFrames.groupby(final_Xm, :id)

# cluster dataframe
clusters = DataFrame()
clusters[!, "id"] = Int64[]
for i = cols
    colname = "$i"
    clusters[!, colname] = Float64[]
end

for i in centroid
    push!(clusters, mean.(eachcol(i)))
end









#--------------------------------------------------------------------------------------#
@info "Done."