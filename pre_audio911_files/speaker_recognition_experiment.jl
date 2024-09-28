using SoleAudio
# using DataFrames
# using Statistics
# using Clustering
# using SoleData
# using StatsBase
# using MLJ
# using Plots

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

#--------------------------------------------------------------------------------------#
#                              audio features extractione                              #
#--------------------------------------------------------------------------------------#
function audio_features_collect(;
    source_path::String,
    afe_cols::Int64,
    sr::Int64,
    stft_length::Int64,
    mel_bands::Int64
)
    # initialize dataframe
    X = DataFrame()
    for i = 1:afe_cols
        colname = "a$i"
        X[!, colname] = Vector{Float64}[]
    end
    X[!, "id"] = String[] # salvo l'id del malcapitato

    cd(source_path)
    for i in readdir()
        # load wav file
        x, sr = load_audio(i, sr=sr)

        # normalize
        x = x ./ maximum(abs.(x))

        # audio feature extraction
        afe = audio_features_extractor(
            x,
            dataset=:speaker_recognition,
            sr=sr,

            # fft
            stft_length=stft_length,
            window_type=[:hann, :periodic],
            # window_length=stft_length,
            # overlap_length=Int(round(stft_length * 0.500)),
            window_length=Int(round(0.03 * sr)),
            overlap_length=Int(round(0.02 * sr)),
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
            log_energy_pos=:replace,
            delta_window_length=9,
            delta_matrix=:standard,

            # spectral
            spectral_spectrum=:linear
        )
        push!(X, (vcat.(afe[j, :] for j in 1:afe_cols)..., split(i, "_")[1]))
    end

    return X
end

#--------------------------------------------------------------------------------------#
#                                        main                                          #
#--------------------------------------------------------------------------------------#

@info "Audio feature extraction started..."

for i in source_path
    print(i)
end

Xdf = audio_features_collect(
    source_path=source_path,
    afe_cols=afe_cols,
    sr=sr,
    stft_length=stft_length,
    mel_bands=mel_bands
)

@info "K-means clustering started..."

@assert Xdf isa DataFrame

X_means = mean.(Xdf[:, 1:end-1])
X_id = Xdf[:, end]

# kmeans, importance of initialization
# https://arxiv.org/pdf/1604.04893.pdf

R = kmeans(
    Matrix(X_means)',                    # in: data matrix (d x n) columns = obs
    k,                                  # in: number of centers
    init=:kmpp,                         # in: initialization algorithm
    maxiter=100,                        # in: maximum number of iterations
    tol=1.0e-6,                         # in: tolerance  of change at convergence
    display=:none,                      # in: level of display
    # distance=SqEuclidean(),           # in: function to calculate distance with
    # rng=Random.GLOBAL_RNG             # in: RNG object
)

@assert nclusters(R) == k                   # verify the number of clusters

a = assignments(R)                          # get the assignments of points to clusters
c = counts(R)                               # get the cluster sizes
M = R.centers                               # get the cluster centers

# scatter(
#     R.centers,
#     marker_z=R.centers,
#     color=:lightrainbow,
#     legend=false
# )

### speech detector ###
# load wav file
sr = 16000
x, sr = load_audio("/home/riccardopasini/Documents/Aclai/Datasets/SpcDS/SpcDS_gender_1000_60_100/WavFiles/common_voice_en_23616312.wav", sr=sr)
# normalize
x = x ./ maximum(abs.(x))
x = speech_detector(x, sr)

#--------------------------------------------------------------------------------------#
#                                  saving jld2 files                                   #
#--------------------------------------------------------------------------------------#


#--------------------------------------------------------------------------------------#

@info "Done."

# cross validation

# mfcc del tizio 1

# media delle medie è il centroide

# mlj package

# matrice di confusione

