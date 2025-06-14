# -------------------------------------------------------------------------- #
#                                 debug F0                                   #
# -------------------------------------------------------------------------- #
using Revise, Audio911, BenchmarkTools
using FFTW, StaticArrays, SplitApplyCombine, DSP, Interpolations
# using Plots, Parameters, FFTW, DSP, StatsBase, NaNStatistics
# using Unitful, NamedArrays

TESTPATH = joinpath(dirname(pathof(Audio911)), "..", "test")
TESTFILE = "common_voice_en_23616312.wav"
# TESTFILE = "104_1b1_Al_sc_Litt3200_4.wav"
wavfile = joinpath(TESTPATH, TESTFILE)

sr = 16000
audio = load_audio(wavfile, sr; norm=false);
stftspec = get_stft(audio; nfft=512, norm=:magnitude);

x = combinedims(collect(stftspec.data.frames))
sr = stftspec.sr
win = stftspec.data.win
noverlap = stftspec.setup.noverlap

method = :srh
freqrange = (50, 400)
mflength = 1

f0 = get_f0(stftspec; method=method, freqrange=freqrange, mflength=mflength)

function parabolic_interpolation(domain, locs)
    offset = collect(0:size(domain, 2)-1) * size(domain, 1)
    locs_linear = locs' .+ offset

    a = domain[max.(locs_linear .- 1, offset .+ 1)]
    b = domain[locs_linear]
    c = domain[min.(locs_linear .+ 1, offset .+ size(domain, 1))]

    s = (c .- a) ./ (2 .* (2 .* b .- c .- a))

    domain_new = b .- 0.25 .* (a .- c) .* s
    clamp.(domain_new, 0, 1)
end

function _get_harmonic_ratio(
    x::AbstractArray{<:AbstractFloat},
	sr::Int;
	win::AbstractVector{<:AbstractFloat},
	noverlap::Int,
)
	nwin=512
	win = _get_window(:hamming, nwin, true)
	x_reframed = _get_wframes(x, win)
	x_pad = (vcat(col, zeros(eltype(col), nfft - nwin)) for col in x_reframed)

	highEdge = min(floor(Int, sr*0.04) , nwin-1)
	nfft = nextpow(2, 2 * nwin - 1)
	c1 = real.(ifft(abs.(fft(combinedims(collect(x_pad)), 1)) .^ 2, 1))
	r = c1[2:highEdge+1,:]

	totalPower = c1[1,:]
	partialPower = reverse(cumsum(combinedims(collect(x_reframed)).^2, dims=1), dims=1)
	partialPower = partialPower[2:highEdge+1, :]

	# Determine the lower edge of the range
	lowEdge = zeros(1, size(r, 2))

	for i in 1:size(r, 2)
		temp = findfirst(diff(sign.(r[:, i])) .!= 0)
		if isnothing(temp)
			lowEdge[i] = highEdge
		else
			lowEdge[i] = temp + 1
		end
	end

	domain = (r' ./ (sqrt.(totalPower .* partialPower') .+ sqrt(eps(eltype(x)))))'

	for i in 1:size(r, 2)
		domain[1:Int(max(lowEdge[i], 1)), i] .= 0
	end

	_, locs = findmax(domain, dims=1)
	locs = [idx[1] for idx in locs]
	parabolic_interpolation(domain, locs)
end

hr = _get_harmonic_ratio(x, sr; win=win, noverlap=noverlap)

f0 = get_f0(stftspec; method=method, freqrange=freqrange, mflength=mflength)
threshold = 0.85
mask = hr .< threshold
f0.data.f0[mask] .= NaN
