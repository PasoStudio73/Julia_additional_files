# ---------------------------------------------------------------------------- #
#                                mel filterbank                                #
# ---------------------------------------------------------------------------- #
struct MelFbSetup
    nbands::Int
    scale::Symbol # :mel_htk, :mel_slaney, :erb, :bark, :semitones, :tuned_semitones
    norm::Symbol # :bandwidth, :area, :none
    freq_range::Tuple{Int, Int}
    semitone_range::Tuple{Int, Int}
end

struct MelFbData
    fbank::AbstractArray{<:AbstractFloat}
	freq::AbstractVector{<:AbstractFloat}
end

struct MelFb
    sr::Int
    setup::MelFbSetup
    data::MelFbData
end

# ---------------------------------------------------------------------------- #
#                         scale convertions functions                          #
# ---------------------------------------------------------------------------- #
function hz2mel(hz::Tuple{Int64, Int64}, style::Symbol)
    style == :mel_htk && return @. 2595 * log10(1 + hz / 700)
    style == :mel_slaney && begin
        lin_step = 200 / 3
        return @. ifelse(hz < 1000, hz / lin_step,
            log(hz * 0.001) / (log(6.4) / 27) + (1000 / lin_step))
    end
    error("Unknown style ($style).")
end

function mel2hz(mel_range::Tuple{Float64, Float64}, n_bands::Int64, style::Symbol)
    mel = LinRange(mel_range[1], mel_range[2], n_bands + 2)
    style == :mel_htk && return @. 700 * (exp10(mel / 2595) - 1)
    style == :mel_slaney && begin
        lin_step = 200 / 3
        cp_mel = 1000 / lin_step
        return @. ifelse(
            mel < cp_mel, mel * lin_step, 1000 * exp(log(6.4) / 27 * (mel - cp_mel)))
    end
    error("Unknown style ($style).")
end

function hz2erb(hz::Tuple{Int64, Int64})
    @. log(10) * 1000 / (24.673 * 4.368) * log10(1 + 0.004368 * hz)
end

function erb2hz(erb_range::Tuple{Float64, Float64}, n_bands::Int64)
    erb = LinRange(erb_range[1], erb_range[2], n_bands)
    @. (10 ^ (erb / (log(10) * 1000 / (24.673 * 4.368))) - 1) / 0.004368
end

function hz2bark(hz::Tuple{Int64, Int64})
    bark = @. 26.81 * hz / (1960 + hz) - 0.53
    map(x -> x < 2 ? 0.85 * x + 0.3 : x > 20.1 ? 1.22 * x - 4.422 : x, bark)
end

function bark2hz(bark_range::Tuple{Float64, Float64}, n_bands::Int64)
    bark = LinRange(bark_range[1], bark_range[2], n_bands + 2)
    bark = map(x -> x < 2 ? (x - 0.3) / 0.85 : x > 20.1 ? (x + 0.22 * 20.1) / 1.22 : x, bark)
    @. 1960 * (bark + 0.53) / (26.28 - bark)
end

function hz2semitone(hz::Tuple{Int64, Int64})
    hz[1] == 0 ? hz = (20, hz[2]) : nothing
    @. 12 * log2(hz)
end

function semitone2hz(st_range::Tuple{Float64, Float64}, nbands::Int64)
    st_range_vec = st_range[1] .+ collect(0:(nbands+1)) / (nbands+1) * (st_range[2] - st_range[1])
    @. 2 ^ (st_range_vec / 12)
end

# ---------------------------------------------------------------------------- #
#                                 normalization                                #
# ---------------------------------------------------------------------------- #
function normalize!(filterbank::AbstractArray{Float64}, norm::Symbol, bw::AbstractVector{Float64})
    norm_funcs = Dict(
        :area => () -> sum(filterbank, dims = 2),
        :bandwidth => () -> bw / 2,
        :none => () -> 1
    )

    weight_per_band = get(norm_funcs, norm, norm_funcs[:none])()
    @. filterbank /= (weight_per_band + (weight_per_band == 0))
end

# ---------------------------------------------------------------------------- #
#                                   gammatone                                  #
# ---------------------------------------------------------------------------- #
function compute_gammatone_coeffs(sr::Int64, bands::AbstractVector{Float64})
    t = 1 / sr
    erb = @. bands / 9.26449 + 24.7
    filt = 1.019 * 2π * erb

    sqrt_plus = √(3 + 2^(3/2))
    sqrt_minus = √(3 - 2^(3/2))
    img = @. 2im * π * bands * t

    exp_f = @. exp(filt * t)
    exp_im = @. exp(2img)
    exp_it = @. -2 * exp_im * t
    exp_t = @. 2 * exp(-(filt * t) + img) * t

    cos_t = @. cos(2 * bands * π * t)
    sin_t = @. sin(2 * bands * π * t)
    cos_e = @. cos_t / exp_f
    sin_e = @. sin_t / exp_f

    a11 = @. -(t * cos_e + sqrt_plus * t * sin_e)
    a12 = @. -(t * cos_e - sqrt_plus * t * sin_e)
    a13 = @. -(t * cos_e + sqrt_minus * t * sin_e)
    a14 = @. -(t * cos_e - sqrt_minus * t * sin_e)

    gain = @. abs((
        (exp_it + exp_t * (cos_t - sqrt_minus * sin_t)) *
        (exp_it + exp_t * (cos_t + sqrt_minus * sin_t)) *
        (exp_it + exp_t * (cos_t - sqrt_plus * sin_t)) *
        (exp_it + exp_t * (cos_t + sqrt_plus * sin_t))) /
        (-2 / exp(2 * filt * t) - 2 * exp_im + 2 * (1 + exp_im) / exp_f)^4
    )

    n_bands = length(bands)
    b1 = @. -2 * cos_e
    b2 = @. exp(-2 * filt * t)

    cat([
            [
                t/gain[ind] a11[ind]/gain[ind] 0 1 b1[ind] b2[ind]
                t           a12[ind]           0 1 b1[ind] b2[ind]
                t           a13[ind]           0 1 b1[ind] b2[ind]
                t           a14[ind]           0 1 b1[ind] b2[ind]
            ] for ind in 1:n_bands
        ]..., dims=3)
end

function compute_old(sr::Int64, bands::AbstractVector{Float64})
    t = 1 / sr
    erb = bands ./ 9.26449 .+ 24.7
    filt = 1.019 * 2π * erb

    a0 = t
    a2 = 0
    b0 = 1

    b1 = -2 * cos.(2 * bands * π * t) ./ exp.(filt * t)
    b2 = exp.(-2 * filt * t)

    a11 = -(2 * t * cos.(2 * bands * π * t) ./ exp.(filt * t) +
            2 * sqrt(3 + 2^(3 / 2)) * t * sin.(2 * bands * π * t) ./ exp.(filt * t)) / 2
    a12 = -(2 * t * cos.(2 * bands * π * t) ./ exp.(filt * t) -
            2 * sqrt(3 + 2^(3 / 2)) * t * sin.(2 * bands * π * t) ./ exp.(filt * t)) / 2
    a13 = -(2 * t * cos.(2 * bands * π * t) ./ exp.(filt * t) +
            2 * sqrt(3 - 2^(3 / 2)) * t * sin.(2 * bands * π * t) ./ exp.(filt * t)) / 2
    a14 = -(2 * t * cos.(2 * bands * π * t) ./ exp.(filt * t) -
            2 * sqrt(3 - 2^(3 / 2)) * t * sin.(2 * bands * π * t) ./ exp.(filt * t)) / 2

    gain = abs.(
        (
        (-2 * exp.(4 * 1im * bands * π * t) .* t .+
         2 * exp.(-(filt .* t) .+ 2 * 1im * bands * π * t) .* t .*
         (cos.(2 * bands * π * t) .- sqrt(3 - 2^(3 / 2)) .*
                                     sin.(2 * bands * π * t))) .*
        (-2 * exp.(4 * 1im * bands * π * t) .* t .+
         2 * exp.(-(filt .* t) .+ 2 * 1im * bands * π * t) .* t .*
         (cos.(2 * bands * π * t) .+ sqrt(3 - 2^(3 / 2)) .*
                                     sin.(2 * bands * π * t))) .*
        (-2 * exp.(4 * 1im * bands * π * t) .* t .+
         2 * exp.(-(filt .* t) .+ 2 * 1im * bands * π * t) .* t .*
         (cos.(2 * bands * π * t) .-
          sqrt(3 + 2^(3 / 2)) .* sin.(2 * bands * π * t))) .*
        (-2 * exp.(4 * 1im * bands * π * t) .* t .+
         2 * exp.(-(filt .* t) .+ 2 * 1im * bands * π * t) .* t .*
         (cos.(2 * bands * π * t) .+ sqrt(3 + 2^(3 / 2)) .* sin.(2 * bands * π * t)))
    ) ./
        (-2 ./ exp.(2 * filt * t) .- 2 * exp.(4 * 1im * bands * π * t) .+
         2 * (1 .+ exp.(4 * 1im * bands * π * t)) ./ exp.(filt * t)) .^ 4
    )

    allfilts = ones(length(bands))
    fcoefs = [a0 * allfilts, a11, a12, a13, a14, a2 * allfilts, b0 * allfilts, b1, b2, gain]

    coeffs = zeros(4, 6, length(bands))

    a0 = fcoefs[1]
    a11 = fcoefs[2]
    a12 = fcoefs[3]
    a13 = fcoefs[4]
    a14 = fcoefs[5]
    a2 = fcoefs[6]
    b0 = fcoefs[7]
    b1 = fcoefs[8]
    b2 = fcoefs[9]
    gain = fcoefs[10]

    for ind in 1:length(bands)
        coeffs[:, :, ind] = [a0[ind]/gain[ind] a11[ind]/gain[ind] a2[ind]/gain[ind] b0[ind] b1[ind] b2[ind];
                             a0[ind] a12[ind] a2[ind] b0[ind] b1[ind] b2[ind];
                             a0[ind] a13[ind] a2[ind] b0[ind] b1[ind] b2[ind];
                             a0[ind] a14[ind] a2[ind] b0[ind] b1[ind] b2[ind]]
    end

    return coeffs
end

# -------------------------------------------------------------------------- #
#                                  f0 log mel                                #
# -------------------------------------------------------------------------- #
function calc_f0(stftvec::AbstractVector{<:AbstractFloat}, stftfreq::StepRangeLen{<:AbstractFloat}, semitone_range::Tuple{Int, Int},)
	x1 = findfirst((x)-> x >= semitone_range[1], stftfreq)
	x2 = findfirst((x)-> x >= semitone_range[2], stftfreq)

	peak_pos = argmax(stftvec[x1:x2-1])
	stftfreq[x1+peak_pos-1]
end

# ---------------------------------------------------------------------------- #
#                           design filterbank matrix                           #
# ---------------------------------------------------------------------------- #
function _get_melfb(
        stftvec::AbstractVector{<:AbstractFloat},
        stftfreq::StepRangeLen{<:AbstractFloat},
        stft_length::Int,
        sr::Int;
        nbands::Int = 26,
        scale::Symbol = :mel_htk, # :mel_htk, :mel_slaney, :erb, :bark, :semitones, :tuned_semitones
        norm::Symbol = :bandwidth,  # :bandwidth, :area, :none
        freq_range::Tuple{Int, Int} = (0, round(Int, sr / 2)),
        semitone_range::Tuple{Int, Int} = (200, 700),
)
    if scale == :erb
        erb_range = hz2erb(freq_range)
        filter_freq = erb2hz(erb_range, nbands)
        coeffs = compute_gammatone_coeffs(sr, filter_freq)
        c_test = compute_old(sr, filter_freq)

        iirfreqz = (b, a, n) -> fft([b; zeros(n - length(b))]) ./ fft([a; zeros(n - length(a))])
        sosfilt = (c, n) -> reduce((x, y) -> x .* y, map(row -> iirfreqz(row[1:3], row[4:6], n), eachrow(c)))
        apply_sosfilt = (i) -> abs.(sosfilt(coeffs[:, :, i], stft_length))

        filterbank = hcat(map(apply_sosfilt, 1:nbands)...)'
        # Derive Gammatone filter bandwidths as a function of center frequencies
        bw = 1.019 * 24.7 * (0.00437 * filter_freq .+ 1)

        # normalization
        (norm != :none) && normalize!(filterbank, norm, bw)

        rem(stft_length, 2) == 0 ? filterbank[:, 2:(stft_length ÷ 2)] .*= 2 : filterbank[:, 2:(stft_length ÷ 2 + 1)] .*= 2
        filterbank = filterbank[:, 1:(stft_length ÷ 2 + 1)]

    else
        if (scale == :mel_htk || scale == :mel_slaney)
            mel_range = hz2mel(freq_range, scale)
            band_edges = mel2hz(mel_range, nbands, scale)
        elseif  scale == :bark
            bark_range = hz2bark(freq_range)
            band_edges = bark2hz(bark_range, nbands)
        elseif scale == :semitones
            st_range = hz2semitone(freq_range)
            band_edges = semitone2hz(st_range, nbands)
        elseif scale == :tuned_semitones
            freq_range = (round(Int, calc_f0(stftvec, stftfreq, semitone_range)), freq_range[2])
            st_range = hz2semitone(freq_range)
            band_edges = semitone2hz(st_range, nbands)
        else
            error("Unknown filterbank frequency scale '($scale)', available scales are: :mel_htk, :mel_slaney, :erb, :bark, :semitones, , :semitones_tuned.")
        end

        filter_freq = band_edges[2:(end - 1)]
        nbands = length(filter_freq)

        p = [findfirst(stftfreq .> edge) for edge in band_edges]
        isnothing(p[end]) ? p[end] = length(stftfreq) : nothing

        # create triangular filters for each band
        bw = diff(band_edges)
        filterbank = zeros(nbands, length(stftfreq))

        for k in 1:nbands
            # rising side of triangle
            @. filterbank[k, p[k]:(p[k + 1] - 1)] = (stftfreq[p[k]:(p[k + 1] - 1)] - band_edges[k]) / bw[k]
            # falling side of triangle
            @. filterbank[k, p[k + 1]:(p[k + 2] - 1)] = (band_edges[k + 2] - stftfreq[p[k + 1]:(p[k + 2] - 1)]) / bw[k + 1]
        end

        bw = (band_edges[3:end] - band_edges[1:(end - 2)])

        # normalization
        (norm != :none) && normalize!(filterbank, norm, bw)
    end
    
    MelFb(sr, MelFbSetup(nbands, scale, norm, freq_range, semitone_range), MelFbData(filterbank, filter_freq))
end

# -------------------------------------------------------------------------- #
#                                   debug                                    #
# -------------------------------------------------------------------------- #
using Revise, BenchmarkTools, Audio911, DSP, Polynomials, FFTW, StatsBase, StaticArrays

TESTPATH = joinpath(dirname(pathof(Audio911)), "..", "test")
TESTFILE = "common_voice_en_23616312.wav"
# TESTFILE = "104_1b1_Al_sc_Litt3200_4.wav"
wavfile = joinpath(TESTPATH, TESTFILE)

sr = 16000
audio = load_audio(file=wavfile, sr=sr, norm=true)


nbands = 26
scale = :erb
freq_range = (0, round(Int, audio.sr / 2))
norm = :bandwidth
stft_length = 1024

stftspec = get_stft(audio=audio, stft_length=stft_length)

# mel filterbank module
melfb = get_melfb(
    stft=stftspec,
    nbands=nbands,
    scale=scale,
    norm=norm,
    freq_range=freq_range
);

stftvec = vec(sum(stftspec.data.spec, dims=2))
stftfreq = stftspec.data.freq
stft_length = stftspec.setup.stft_length
sr = stftspec.sr
nbands = 26
scale = :erb # :mel_htk, :mel_slaney, :erb, :bark, :semitones
norm = :bandwidth  # :bandwidth, :area, :none
freq_range = (0, round(Int, sr / 2))
f0_range = (200, 700)

mels = get_melspec(stft=stftspec, fbank=melfb, db_scale=true)



Nbands = 26
maxfreq = round(Int, sr / 2)
b_freq = 586
minfreq = 20
SEMITONE_MINFREQ = 20.0
function semitonebands(Nbands::Integer, maxfreq::Real, b_freq::Real, minfreq::Real = SEMITONE_MINFREQ)::Vector{Float64}
    hz2semtone(freq, base_freq) = 12 * log2(freq/base_freq)
    semtone2hz(z,    base_freq) = base_freq * (2 ^ (z / 12))
    minsemitone = hz2semtone(minfreq, base_freq)
    maxsemitone = hz2semtone(maxfreq, base_freq)
    semtone2hz.(minsemitone .+ collect(0:(nbands+1)) / (nbands+1) * (maxsemitone-minsemitone), base_freq);
end

a = semitonebands(Nbands, maxfreq, 100, minfreq)
b = semitonebands(Nbands, maxfreq, 200, minfreq)
c = semitonebands(Nbands, maxfreq, 400, minfreq)
d = semitonebands(Nbands, maxfreq, 1000, minfreq)