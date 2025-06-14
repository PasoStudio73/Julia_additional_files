# ---------------------------------------------------------------------------- #
#                                    stft                                      #
# ---------------------------------------------------------------------------- #
function _get_stft(;
        x::AbstractVector{<:AbstractFloat},
        sr::Int,
        nfft::Union{Int, Time, Nothing} = nothing,
        winlength::Union{Int, Time, Nothing} = nothing,
        overlaplength::Union{Int, Time, Nothing} = nothing,
        wintype::Tuple{Symbol, Symbol} = (:hann, :periodic),
        norm::Symbol = :power, # :none, :power, :magnitude, :pow2mag
)
    # ms to sample conversion
    # typeof(nfft) <: Time && begin nfft = round(Int, ustrip(Int64, u"ms", nfft) * sr * 0.001) end
    # typeof(winlength) <: Time && begin winlength = round(Int, ustrip(Int64, u"ms", winlength) * sr * 0.001) end
    # typeof(overlaplength) <: Time && begin overlaplength = round(Int, ustrip(Int64, u"ms", overlaplength) * sr * 0.001) end

    # apply default parameters if not provided
    nfft = nfft !== nothing ? nfft : sr !== nothing ? (sr <= 8000 ? 256 : 512) : 512
    winlength = winlength !== nothing ? winlength : nfft
    overlaplength = overlaplength !== nothing ? overlaplength : round(Int, winlength / 2)

    @assert overlaplength<winlength "Overlap length must be smaller than winlength."

    x_length = size(x, 1)
    frames, win, wframes, _, _ = _get_frames(x, wintype, winlength, overlaplength)

    if winlength < nfft
        wframes = vcat(wframes, zeros(Float64, nfft - winlength, size(wframes, 2)))
    elseif winlength > nfft
        @error("FFT window size smaller than actual window size is highly discuraged.")
    end

    # take one side
    if mod(nfft, 2) == 0
        oneside = 1:Int(nfft / 2 + 1)   # even
    else
        oneside = 1:Int((nfft + 1) / 2)  # odd
    end

    # normalize
    norm_funcs = Dict(
        :power => x -> real.((x .* conj.(x))),
        :magnitude => x -> abs.(x),
        :pow2mag => x -> sqrt.(real.((x .* conj.(x))))
    )
    # check if spectrum_type is valid
    @assert haskey(norm_funcs, norm) "Unknown spectrum_type: $norm."

    spec = norm_funcs[norm](DSP.stft(x, winlength, overlaplength, nothing; onesided=true, nfft=nfft, fs=sr, window=win))
    freq = (sr / nfft) * (oneside .- 1)

    # Stft(sr, x_length, StftSetup(nfft, wintype, winlength, overlaplength, norm), StftData(spec, freq, win, frames))
end

### TEST DSP
@btime test = DSP.stft(x, 256, 128, nothing; onesided=true, nfft=256, fs=sr, window=win);

### TEST FFT
@btime begin
    if mod(nfft, 2) == 0
        oneside = 1:Int(nfft / 2 + 1)   # even
    else
        oneside = 1:Int((nfft + 1) / 2)  # odd
    end
    spec = fft(wframes, (1,))[oneside, :];
end

# -------------------------------------------------------------------------- #
#                                   debug                                    #
# -------------------------------------------------------------------------- #
using Revise, Audio911, BenchmarkTools
using FFTW, StaticArrays
# using Plots, Parameters, FFTW, DSP, StatsBase, NaNStatistics
# using Unitful, NamedArrays

TESTPATH = joinpath(dirname(pathof(Audio911)), "..", "test")
TESTFILE = "common_voice_en_23616312.wav"
# TESTFILE = "104_1b1_Al_sc_Litt3200_4.wav"
wavfile = joinpath(TESTPATH, TESTFILE)

sr = 16000
audio = load_audio(source=wavfile, sr=sr, norm=false);

x=audio.data
sr=16000
nfft=256
winlength=256
overlaplength=128
wintype=(:hann, :periodic)
norm=:power # :none, :power, :magnitude, :pow2mag

