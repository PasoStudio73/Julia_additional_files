## feature extraction launcher from arrow file
using Pkg
using Arrow
using CSV
using Catch22
using DataFrames
using JLD2
using DataStructures
using ConfigEnv

using PyCall
af = pyimport("audioflux")
librosa = pyimport("librosa")
plt = pyimport("matplotlib.pyplot")

dotenv()
LOCAL_PATH = ENV["DEST_PATH"]

arrow_file = "SpcDS_age2bins_605_30_95_nbs_matlab115_1024_256"
jld_file = "pub_age2bins_3s"

m_col = 115
label = "age"

df = DataFrame(Arrow.Table(string(LOCAL_PATH, "/", arrow_file, ".arrow")))
Xdf = select(df, Between(:a1, Symbol(String(:a), m_col)))
y = df[:, label]

d = jldopen(string(LOCAL_PATH, "/", jld_file, ".jld2"))
Cdf, Cy, attrs_n_meas = d["dataframe_validated"]

    @info "Part 1: arrow extraction..."
    ## import specs from arrow file_name
    """
    Format:
        SpcDS
        _"dataset type"
        _"samples * type"
        _"length in secs * 10"
        _"signal silence ratio"
        _"phone filter used"
        _"mfcc type"_"fft size"
        _"fft bands"
        .arrow
    """
    
    filename_split = split(arrow_file, "_")
    
    if (
        filename_split[2] == "gender"
    )
        label = "gender"
    elseif (
        filename_split[2] == "age2bins" ||
        filename_split[2] == "age4bins" ||
        filename_split[2] == "age8bins"
    )
        label = "age"
    elseif (
        filename_split[2] == "region2bins" ||
        filename_split[2] == "region4bins" ||
        filename_split[2] == "region8bins"
    )
        label = "region"
    else
        error("Unknown dataset type: ", filename_split[2])
    end
    
    samples = parse(Int64, filename_split[3])
    
    if (
        filename_split[2] == "gender" ||
        filename_split[2] == "region2bins"
    )
        total_samples = samples * 2
    elseif (
        filename_split[2] == "age2bins" ||
        filename_split[2] == "region4bins"
    )
        total_samples = samples * 4
    elseif (
        filename_split[2] == "age4bins" ||
        filename_split[2] == "region8bins"
    )
        total_samples = samples * 8
    elseif (
        filename_split[2] == "age8bins"
    )
        total_samples = samples * 16
    else
        error("Unknown dataset type: ", filename_split[2])
    end
    
    sample_length = parse(Float64, filename_split[4]) / 10
    
    if (
        filename_split[7] == "afmfcc"
    )
        m_col = 13
    elseif (
        filename_split[7] == "matlab115"
    )
        m_col = 115
    else
        error("Unknown application type: ", filename_split[6])
    end
    
    println("Arrow file spec:")
    println(
        "Label: ", label,
        ", type: ", filename_split[2],
        ", audio features: ", filename_split[7],
        ", number of features: ", m_col,
        ".")
    println(
        "Samples*type:", samples,
        ", total:", total_samples,
        ", of length:", sample_length,
        " seconds.")
    
    println("Balance check:")
    
    if (filename_split[2] == "gender")
        n_female = 0
        n_male = 0
        for i in eachindex(df[:, m_col+1])
            if (df[i, m_col+1] == "female")
                global n_female += 1
            else
                global n_male += 1
            end
        end
        println("female: ", n_female, ", male: ", n_male, ".")
    
    elseif (filename_split[2] == "age2bins")
        n_25 = 0
        n_70 = 0
        for i in eachindex(y)
            if (y[i] == "25")
                global n_25 += 1
            elseif (y[i] == "70")
                global n_70 += 1
            else
                error("found unknown label: ", y[i])
            end
        end
        println("25: ", n_25, ", 70: ", n_70, ".")
    
        # n_female_25 = 0
        # n_male_25 = 0
        # n_female_70 = 0
        # n_male_70 = 0
        # for i in eachindex(df[:, m_col+2])
        #     if (df[i, m_col+1] == "female" && df[i, m_col+2] == "25")
        #         n_female_25 += 1
        #     elseif (df[i, m_col+1] == "male" && df[i, m_col+2] == "25")
        #         n_male_25 += 1
        #     elseif (df[i, m_col+1] == "female" && df[i, m_col+2] == "70")
        #         n_female_70 += 1
        #     elseif (df[i, m_col+1] == "male" && df[i, m_col+2] == "70")
        #         n_male_70 += 1
        #     end
        # end
        # println(
        #     "female young: ", n_female_25,
        #     ", female old: ", n_female_70,
        #     ", male young: ", n_male_25,
        #     ", male old: ", n_male_70, ".")
    
        # elseif (filename_split[2] == "age4bins")
        #     n_female = 0
        #     n_male = 0
        #     for i in eachindex(df[:, m_col+1])
        #         if (df[i, m_col+1] == "female")
        #             n_female += 1
        #         else
        #             n_male += 1
        #         end
        #     end
        #     println("female: ", n_female, ", male: ", n_male, ".")
        #     println(f, "female: ", n_female, ", male: ", n_male, ".")
    
        #     n_female_20 = 0
        #     n_male_20 = 0
        #     n_female_40 = 0
        #     n_male_40 = 0
        #     n_female_60 = 0
        #     n_male_60 = 0
        #     n_female_80 = 0
        #     n_male_80 = 0
        #     for i in eachindex(df[:, m_col+2])
        #         if (df[i, m_col+1] == "female" && df[i, m_col+2] == "20")
        #             n_female_20 += 1
        #         elseif (df[i, m_col+1] == "male" && df[i, m_col+2] == "20")
        #             n_male_20 += 1
        #         elseif (df[i, m_col+1] == "female" && df[i, m_col+2] == "40")
        #             n_female_40 += 1
        #         elseif (df[i, m_col+1] == "male" && df[i, m_col+2] == "40")
        #             n_male_40 += 1
        #         elseif (df[i, m_col+1] == "female" && df[i, m_col+2] == "60")
        #             n_female_60 += 1
        #         elseif (df[i, m_col+1] == "male" && df[i, m_col+2] == "60")
        #             n_male_60 += 1
        #         elseif (df[i, m_col+1] == "female" && df[i, m_col+2] == "80")
        #             n_female_80 += 1
        #         elseif (df[i, m_col+1] == "male" && df[i, m_col+2] == "80")
        #             n_male_80 += 1
        #         end
        #     end
        #     println(
        #         "female 20: ", n_female_20,
        #         ", female 40: ", n_female_40,
        #         ", female 60: ", n_female_60,
        #         ", female 80: ", n_female_80, ".")
        #     println(
        #         "male 20: ", n_male_20,
        #         ", male 40: ", n_male_40,
        #         ", male 60: ", n_male_60,
        #         ", male 80: ", n_male_80, ".")
    
        # elseif (filename_split[2] == "age8bins")
        #     n_female = 0
        #     n_male = 0
        #     for i in eachindex(df[:, m_col+1])
        #         if (df[i, m_col+1] == "female")
        #             n_female += 1
        #         else
        #             n_male += 1
        #         end
        #     end
        #     println("female: ", n_female, ", male: ", n_male, ".")
        #     println(f, "female: ", n_female, ", male: ", n_male, ".")
    
        #     n_female_15 = 0
        #     n_male_15 = 0
        #     n_female_25 = 0
        #     n_male_25 = 0
        #     n_female_35 = 0
        #     n_male_35 = 0
        #     n_female_45 = 0
        #     n_male_45 = 0
        #     n_female_55 = 0
        #     n_male_55 = 0
        #     n_female_65 = 0
        #     n_male_65 = 0
        #     n_female_75 = 0
        #     n_male_75 = 0
        #     n_female_85 = 0
        #     n_male_85 = 0
        #     for i in eachindex(df[:, m_col+2])
        #         if (df[i, m_col+1] == "female" && df[i, m_col+2] == "15")
        #             n_female_15 += 1
        #         elseif (df[i, m_col+1] == "male" && df[i, m_col+2] == "15")
        #             n_male_15 += 1
        #         elseif (df[i, m_col+1] == "female" && df[i, m_col+2] == "25")
        #             n_female_25 += 1
        #         elseif (df[i, m_col+1] == "male" && df[i, m_col+2] == "25")
        #             n_male_25 += 1
        #         elseif (df[i, m_col+1] == "female" && df[i, m_col+2] == "35")
        #             n_female_35 += 1
        #         elseif (df[i, m_col+1] == "male" && df[i, m_col+2] == "35")
        #             n_male_35 += 1
        #         elseif (df[i, m_col+1] == "female" && df[i, m_col+2] == "45")
        #             n_female_45 += 1
        #         elseif (df[i, m_col+1] == "male" && df[i, m_col+2] == "45")
        #             n_male_45 += 1
        #         elseif (df[i, m_col+1] == "female" && df[i, m_col+2] == "55")
        #             n_female_55 += 1
        #         elseif (df[i, m_col+1] == "male" && df[i, m_col+2] == "55")
        #             n_male_55 += 1
        #         elseif (df[i, m_col+1] == "female" && df[i, m_col+2] == "65")
        #             n_female_65 += 1
        #         elseif (df[i, m_col+1] == "male" && df[i, m_col+2] == "65")
        #             n_male_65 += 1
        #         elseif (df[i, m_col+1] == "female" && df[i, m_col+2] == "75")
        #             n_female_75 += 1
        #         elseif (df[i, m_col+1] == "male" && df[i, m_col+2] == "75")
        #             n_male_75 += 1
        #         elseif (df[i, m_col+1] == "female" && df[i, m_col+2] == "85")
        #             n_female_85 += 1
        #         elseif (df[i, m_col+1] == "male" && df[i, m_col+2] == "85")
        #             n_male_85 += 1
        #         end
        #     end
        #     println(
        #         "female 15: ", n_female_15,
        #         ", female 25: ", n_female_25,
        #         ", female 35: ", n_female_35,
        #         ", female 45: ", n_female_45,
        #         ", female 55: ", n_female_55,
        #         ", female 65: ", n_female_65,
        #         ", female 75: ", n_female_75,
        #         ", female 85: ", n_female_85, ".")
        #     println(
        #         "male 15: ", n_male_15,
        #         ", male 25: ", n_male_25,
        #         ", male 35: ", n_male_35,
        #         ", male 45: ", n_male_45,
        #         ", male 55: ", n_male_55,
        #         ", male 65: ", n_male_65,
        #         ", male 75: ", n_male_75,
        #         ", male 85: ", n_male_85, ".")
    
        # elseif (filename_split[2] == "region2bins")
        #     region1 = 0
        #     region2 = 0
        #     for i in eachindex(df[:, m_col+1])
        #         if (df[i, m_col+1] == "1.0")
        #             region1 += 1
        #         else
        #             region2 += 1
        #         end
        #     end
        #     println("Asia: ", region1, ", NorthAmerica: ", region2, ".")
    
    else
        error("Unknown application type: ", label)
    end
    
    # println("NaN check:")
    
    # n_NaN = 0
    # for i in 1:nrow(Xdf)
    #     for j in 1:ncol(Xdf)
    #         for k in 1:length(Xdf[i, j])
    #             if (Xdf[i, j][k] == NaN)
    #                 n_NaN += 1
    #                 println("Found NaN in: ", i, " ", j)
    #             end
    #         end
    #     end
    # end
    
    # if (n_NaN == 0)
    #     println("No NaN values founded.")
    # end

    println()
    println("Xdf structure:")
    println("1-13: mfcc, 14-26: mfcc delta, 27-39: mfcc deltadelta, 40-52: gtcc, 53-65: gtcc delta, 66-78: gtcc deltadelta.")
    println("79:centroid, 80:crest, 81:decrease, 82:entropy, 83:flatness, 84:flux, 85:kurtosis, 86:rolloff, 87:skewness, 88:slope, 89:spread.")
    println("90-115: mel spectrogram, 26 points.")
    println()
    @info "Save arrow plot one sample."

    mfcc = Xdf[1, Between(:a1, :a13)]
    mfcc_delta = Xdf[1, Between(:a14, :a26)]
    mfcc_deltadelta = Xdf[1, Between(:a27, :a39)]
    gtcc = Xdf[1, Between(:a40, :a52)]
    gtcc_delta = Xdf[1, Between(:a53, :a65)]
    gtcc_deltadelta = Xdf[1, Between(:a66, :a78)]

    centroid = Xdf[1, :a79]
    crest = Xdf[1, :a80]
    decrease = Xdf[1, :a81]
    entropy = Xdf[1, :a82]
    flatness = Xdf[1, :a83]
    flux = Xdf[1, :a84]
    kurtosis = Xdf[1, :a85]
    rolloff = Xdf[1, :a86]
    skewness = Xdf[1, :a87]
    slope = Xdf[1, :a88]
    spread = Xdf[1, :a89]

    mel_spectrogram = Xdf[1, Between(:a90, :a115)]

    dim = length(mfcc[1])
    plt.figure().clear()

    display = Matrix{Float64}(undef, 0, dim)
    for i in 1:13
        global display = vcat(display, reshape(convert(Vector, mfcc[i]), 1, dim))
    end
    librosa.display.specshow(display)
    plt.savefig(string(LOCAL_PATH, "/", "arrow_mfcc.jpg"))
    plt.figure().clear()

    display = Matrix{Float64}(undef, 0, dim)
    for i in 1:13
        global display = vcat(display, reshape(convert(Vector, mfcc_delta[i]), 1, dim))
    end
    librosa.display.specshow(display)
    plt.savefig(string(LOCAL_PATH, "/", "arrow_mfcc_delta.jpg"))
    plt.figure().clear()

    display = Matrix{Float64}(undef, 0, dim)
    for i in 1:13
        global display = vcat(display, reshape(convert(Vector, mfcc_deltadelta[i]), 1, dim))
    end
    librosa.display.specshow(display)
    plt.savefig(string(LOCAL_PATH, "/", "arrow_mfcc_deltadelta.jpg"))
    plt.figure().clear()

    display = Matrix{Float64}(undef, 0, dim)
    for i in 1:13
        global display = vcat(display, reshape(convert(Vector, gtcc[i]), 1, dim))
    end
    librosa.display.specshow(display)
    plt.savefig(string(LOCAL_PATH, "/", "arrow_gtcc.jpg"))
    plt.figure().clear()

    display = Matrix{Float64}(undef, 0, dim)
    for i in 1:13
        global display = vcat(display, reshape(convert(Vector, gtcc_delta[i]), 1, dim))
    end
    librosa.display.specshow(display)
    plt.savefig(string(LOCAL_PATH, "/", "arrow_gtcc_delta.jpg"))
    plt.figure().clear()

    display = Matrix{Float64}(undef, 0, dim)
    for i in 1:13
        global display = vcat(display, reshape(convert(Vector, gtcc_deltadelta[i]), 1, dim))
    end
    librosa.display.specshow(display)
    plt.savefig(string(LOCAL_PATH, "/", "arrow_gtcc_deltadelta.jpg"))
    plt.figure().clear()

    librosa.display.waveshow(centroid')
    plt.savefig(string(LOCAL_PATH, "/", "arrow_centroid.jpg"))
    plt.figure().clear()
    librosa.display.waveshow(crest')
    plt.savefig(string(LOCAL_PATH, "/", "arrow_crest.jpg"))
    plt.figure().clear()
    librosa.display.waveshow(decrease')
    plt.savefig(string(LOCAL_PATH, "/", "arrow_decrease.jpg"))
    plt.figure().clear()
    librosa.display.waveshow(entropy')
    plt.savefig(string(LOCAL_PATH, "/", "arrow_entropy.jpg"))
    plt.figure().clear()
    librosa.display.waveshow(flatness')
    plt.savefig(string(LOCAL_PATH, "/", "arrow_flatness.jpg"))
    plt.figure().clear()
    librosa.display.waveshow(flux')
    plt.savefig(string(LOCAL_PATH, "/", "arrow_flux.jpg"))
    plt.figure().clear()
    librosa.display.waveshow(kurtosis')
    plt.savefig(string(LOCAL_PATH, "/", "arrow_kurtosis.jpg"))
    plt.figure().clear()
    librosa.display.waveshow(rolloff')
    plt.savefig(string(LOCAL_PATH, "/", "arrow_rolloff.jpg"))
    plt.figure().clear()
    librosa.display.waveshow(skewness')
    plt.savefig(string(LOCAL_PATH, "/", "arrow_skewness.jpg"))
    plt.figure().clear()
    librosa.display.waveshow(slope')
    plt.savefig(string(LOCAL_PATH, "/", "arrow_slope.jpg"))
    plt.figure().clear()
    librosa.display.waveshow(spread')
    plt.savefig(string(LOCAL_PATH, "/", "arrow_spread.jpg"))
    plt.figure().clear()

    display = Matrix{Float64}(undef, 0, dim)
    for i in 1:26
        global display = vcat(display, reshape(convert(Vector, mel_spectrogram[i]), 1, dim))
    end
    librosa.display.specshow(display)
    plt.savefig(string(LOCAL_PATH, "/", "arrow_mel_spectrogram.jpg"))
    plt.figure().clear()

    @info "Part 2: CSV extraction..."

    Cmfcc = Xdf[1, Between(:a1, :a13)]
    Cmfcc_delta = Xdf[1, Between(:a14, :a26)]
    Cmfcc_deltadelta = Xdf[1, Between(:a27, :a39)]
    Cgtcc = Xdf[1, Between(:a40, :a52)]
    Cgtcc_delta = Xdf[1, Between(:a53, :a65)]
    Cgtcc_deltadelta = Xdf[1, Between(:a66, :a78)]

    Ccentroid = Xdf[1, :a79]
    Ccrest = Xdf[1, :a80]
    Cdecrease = Xdf[1, :a81]
    Centropy = Xdf[1, :a82]
    Cflatness = Xdf[1, :a83]
    Cflux = Xdf[1, :a84]
    Ckurtosis = Xdf[1, :a85]
    Crolloff = Xdf[1, :a86]
    Cskewness = Xdf[1, :a87]
    Cslope = Xdf[1, :a88]
    Cspread = Xdf[1, :a89]

    Cmel_spectrogram = Xdf[1, Between(:a90, :a115)]

    dim = length(mfcc[1])
    plt.figure().clear()
    @info "Done..."