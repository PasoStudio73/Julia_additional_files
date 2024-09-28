# -------------------------------------------------------------------------- #
#                                 debug F0                                   #
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

stftspec = get_stft(source=audio, nfft=512, norm=:magnitude);

x=stftspec.data.frames
sr=stftspec.sr
winlength=stftspec.setup.winlength

method = :nfc
freqrange = (50, 400)
mflength = 1

# ---------------------------------------------------------------------------- #
#                            fundamental frequency                             #
# ---------------------------------------------------------------------------- #
struct F0Setup
	method::Symbol
	freqrange::Tuple{Int64, Int64}
	mflength::Int64
end

struct F0Data
    f0::AbstractVector{<:AbstractFloat}
end

struct F0
    sr::Int64
    setup::F0Setup
    data::F0Data
end

# ---------------------------------------------------------------------------- #
#                                    utilities                                 #
# ---------------------------------------------------------------------------- #
function get_candidates(domain::AbstractArray{Float64}, edge::Tuple{Int64, Int64})
	peaks, locs = findmax(domain[collect(edge[2]:edge[1]), :], dims = 1)
	locs = edge[2] .+ map(i -> i[1], locs) .- 1

	return peaks, locs
end

function i_clip(x::AbstractVector{Float64}, range::Tuple{Int64, Int64})
	x[x.<range[1]] .= range[1]
	x[x.>range[2]] .= range[2]

	return x
end

# ---------------------------------------------------------------------------- #
#                             fundamental frequency                            #
# ---------------------------------------------------------------------------- #
function _get_f0(;
	x::AbstractArray{Float64},
	sr::Int64,
	winlength::Int64,
	method::Symbol = :nfc,
	freqrange::Tuple{Int64, Int64} = (50, 400),
	mflength::Int64 = 1,
)
	if method == :nfc
		edge = round.(Int, sr ./ freqrange)
		mxl = min(edge[1], winlength - 1)
		m2 = nextpow(2, 2 * winlength - 1)

		y_m2 = vcat(x, zeros(m2 - size(x, 1), size(x, 2)))
		c1 = real.(ifft(abs2.(fft(y_m2, 1)), 1)) ./ sqrt(m2)

		Rt = vcat(view(c1, m2-mxl+1:m2, :), view(c1, 1:mxl+1, :))
		lag = view(Rt, edge[1]+1+edge[2]:size(Rt,1), :) ./ sqrt.(view(Rt, edge[1] + 1, :))'

		domain = vcat(zeros(edge[2] - 1, size(lag, 2)), lag)

		_, locs = get_candidates(domain, edge)

		f0 = vec(sr ./ locs)

		## TODO
		# elseif f0.method == :srh
		# elseif f0.method == :pef
		# elseif f0.method == :cep
		# elseif f0.method == :lhs

		F0(sr, F0Setup(method, freqrange, mflength), F0Data(f0))
	end
end

function Base.show(io::IO, f0::F0)
    println(io, "F0 Estimation:")
    println(io, "  Sample Rate: $(f0.sr) Hz")
    println(io, "  Method: $(f0.setup.method)")
    println(io, "  Frequency Range: $(f0.setup.freqrange) Hz")
    println(io, "  F0: $(length(f0.data.f0)) points")
end

function Base.display(f0::F0)
    time = (0:length(f0.data.f0)-1)
    plot(time, f0.data.f0, 
         title="Fundamental Frequency Estimation",
         xlabel="Frame",
         ylabel="Frequency (Hz)",
         label="F0",
         linewidth=2,
         ylim=(0, maximum(f0.data.f0) * 1.1),
         legend=:none)
end

get_f0(; source::Stft, kwargs...) =_get_f0(; x=source.data.frames, sr=source.sr, winlength=source.setup.winlength, kwargs...)
