using Pkg
Pkg.activate("/home/paso/Documents/Aclai/audio-rules2024")
using Revise, Audio911, BenchmarkTools
using StaticArrays

using SpecialFunctions, Roots
using Statistics

TESTPATH = joinpath(dirname(pathof(Audio911)), "..", "test")
TESTFILE = "common_voice_en_23616312.wav"
# TESTFILE = "104_1b1_Al_sc_Litt3200_4.wav"
wavfile = joinpath(TESTPATH, TESTFILE)

# --- audio ------------------------------------------------------------------ #
sr = 16000
audio = load_audio(file=wavfile, sr=sr, norm=true);

# --- stft ------------------------------------------------------------------- #
stftspec = get_stft(audio=audio);

sr = stftspec.sr
# bands = collect(stftspec.data.freq);
bands = stftspec.data.freq;

# ---------------------------------------------------------------------------- #
function cgc1(sr::Int64, bands::AbstractVector{Float64})
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

# ---------------------------------------------------------------------------- #
function cgc2(sr::Int64, bands::AbstractVector{Float64})
    t = 1 / sr
    erb = @. bands / 9.26449 + 24.7
    filt = 1.019 * 2π * erb

    b1 = @. -2 * cos(2 * bands * π * t) / exp(filt * t)
    b2 = @. exp(-2 * filt * t)

    sqrt3plus2 = sqrt(3 + 2^(3/2))
    sqrt3minus2 = sqrt(3 - 2^(3/2))
    
    common_term = @. 2 * t / exp(filt * t)
    cos_term = @. common_term * cos(2 * bands * π * t)
    sin_term = @. common_term * sin(2 * bands * π * t)

    a11 = @. -(cos_term + sqrt3plus2 * sin_term) / 2
    a12 = @. -(cos_term - sqrt3plus2 * sin_term) / 2
    a13 = @. -(cos_term + sqrt3minus2 * sin_term) / 2
    a14 = @. -(cos_term - sqrt3minus2 * sin_term) / 2

    exp_term = @. exp(4im * bands * π * t)
    exp_filt_term = @. exp(-(filt * t) + 2im * bands * π * t)
    cos_sin_term = @. cos(2 * bands * π * t) + im * sin(2 * bands * π * t)

    numerator = @. (-2 * exp_term * t + 2 * exp_filt_term * t * (cos_sin_term - sqrt3minus2))
    numerator .*= @. (-2 * exp_term * t + 2 * exp_filt_term * t * (cos_sin_term + sqrt3minus2))
    numerator .*= @. (-2 * exp_term * t + 2 * exp_filt_term * t * (cos_sin_term - sqrt3plus2))
    numerator .*= @. (-2 * exp_term * t + 2 * exp_filt_term * t * (cos_sin_term + sqrt3plus2))

    denominator = @. (-2 / exp(2 * filt * t) - 2 * exp_term + 2 * (1 + exp_term) / exp(filt * t))^4

    gain = @. abs(numerator / denominator)

    n_bands = length(bands)
    coeffs = zeros(4, 6, n_bands)

    @inbounds for ind in 1:n_bands
        coeffs[:, :, ind] = [
            t/gain[ind]  a11[ind]/gain[ind]  0        1  b1[ind]  b2[ind]
            t            a12[ind]            0        1  b1[ind]  b2[ind]
            t            a13[ind]            0        1  b1[ind]  b2[ind]
            t            a14[ind]            0        1  b1[ind]  b2[ind]
        ]
    end

    return coeffs
end

# ---------------------------------------------------------------------------- #
@btime begin
    bands = collect(b)
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

    coeffs2 = zeros(4, 6, length(bands))

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
        coeffs2[:, :, ind] = [a0[ind]/gain[ind] a11[ind]/gain[ind] a2[ind]/gain[ind] b0[ind] b1[ind] b2[ind];
                             a0[ind] a12[ind] a2[ind] b0[ind] b1[ind] b2[ind];
                             a0[ind] a13[ind] a2[ind] b0[ind] b1[ind] b2[ind];
                             a0[ind] a14[ind] a2[ind] b0[ind] b1[ind] b2[ind]]
    end
end

# ---------------------------------------------------------------------------- #
@btime begin
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

    [SMatrix{4,6}(
        t/gain[ind], a11[ind]/gain[ind], 0, 1, b1[ind], b2[ind],
        t,           a12[ind],           0, 1, b1[ind], b2[ind],
        t,           a13[ind],           0, 1, b1[ind], b2[ind],
        t,           a14[ind],           0, 1, b1[ind], b2[ind]
    ) for ind in 1:n_bands]
end
;