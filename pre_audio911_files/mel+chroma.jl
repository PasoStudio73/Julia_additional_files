function hz2mel(
	hz::Tuple{Int64, Int64},
	mel_style::Symbol = :htk, # :htk, :slaney
)
	if mel_style == :htk
		mel = 2595 * log10.(1 .+ reduce(vcat, getindex.(hz)) / 700)
	else # slaney
		hz = reduce(vcat, getindex.(hz))
		linStep = 200 / 3
		logStep = log(6.4) / 27
		changePoint = 1000
		changePoint_mel = changePoint / linStep
		isLinearRegion = hz .< changePoint
		mel = Float64.(hz)
		mel[isLinearRegion] .= hz[isLinearRegion] / linStep
		mel[.!isLinearRegion] .= changePoint_mel .+
								 log.(hz[.!isLinearRegion] / changePoint) / logStep
	end
	return mel
end # hz2mel

function mel2hz(
	mel::LinRange{Float64, Int64},
	mel_style::Symbol = :htk, # :htk, :slaney
)
	if mel_style == :htk
		hz = 700 * (exp10.(mel / 2595) .- 1)
	else
		linStep = 200 / 3
		logStep = log(6.4) / 27
		changePoint = 1000
		changePoint_mel = changePoint / linStep
		isLinearRegion = mel .< changePoint_mel
		hz = [mel;]
		hz[isLinearRegion] .= hz[isLinearRegion] * linStep
		hz[.!isLinearRegion] .= changePoint *
								exp.(logStep * (mel[.!isLinearRegion] .- changePoint_mel))
	end
	return hz
end # mel2hz

function hz2octs(freq, tuning = 440)
	# Convert a frequency in Hz into a real number counting  the octaves above A0. So hz2octs(440) = 4.0

	octs = log(freq ./ (tuning / 16)) ./ log(2)
end

function octs2hz(octs, tuning = 440)
	# Convert a real-number octave into a frequency in Hz.

	hz = (tuning / 16) .* (2 .^ octs)
end

function ifgram(X, N = 256, W = N, H = W / 2, SR = 1)
	s = length(X)
	# X = reshape(X, 1, :)

	win = 0.5 * (1 .- cos.(vcat([0:(W-1)]...) / W * 2 * pi))
	T = W / SR
	dwin = -pi / T * sin.(vcat([0:(W-1)]...) / W * 2 * pi)
	norm = 2 / sum(win)

	nhops = 1 + floor(Int, (s - W) / H)

	F = zeros(floor(Int, 1 + N / 2), nhops)
	D = zeros(floor(Int, 1 + N / 2), nhops)

	nmw1 = floor(Int, (N - W) / 2)
	nmw2 = N - W - nmw1

	ww = 2 * pi * [0:(N-1)] * SR / N

	for h ∈ 1:nhops
		u = X[(h-1)*H.+vcat([1:W]...)]
		wu = win .* u
		du = dwin .* u

		if N > W
			wu = vcat(zeros(1, nmw1), wu, zeros(1, nmw2))
			du = vcat(zeros(1, nmw1), du, zeros(1, nmw2))
		end
		if N < W
			wu = wu[-nmw1+[1:N]]
			du = du[-nmw1+[1:N]]
		end

		t1 = fft(fftshift(du))
		t2 = fft(fftshift(wu))
		D[:, h] = t2[1:(1+N/2)]' * norm

		t = t1 + im * (ww .* t2)
		a = real(t2)
		b = imag(t2)
		da = real(t)
		db = imag(t)
		instf = (1 / (2 * pi)) * (a .* db - b .* da) ./ ((a .* a + b .* b) + (abs.(t2) .== 0))

		F[:, h] = instf[1:(1+N/2)]'
	end

	return F, D
end

function ifptrack(d, w, sr, fminl = 150, fminu = 300, fmaxl = 2000, fmaxu = 4000)
	# Calculate the inst freq gram
	I, S = ifgram(d, w, w ÷ 2, w ÷ 4, sr)

	# Only look at bins up to 2 kHz
	maxbin = round(Int, fmaxu * (w / sr))
	minbin = round(Int, fminl * (w / sr))

	# Find plateaus in ifgram - stretches where delta IF is < thr
	ddif = vcat(I[2:maxbin, :], I[maxbin, :]) - vcat(I[1, :], I[1:(maxbin-1), :])

	# expected increment per bin = sr/w, threshold at 3/4 that
	dgood = abs.(ddif) .< 0.75 * sr / w

	# delete any single bins (both above and below are zero);
	dgood = dgood .* (vcat(dgood[2:maxbin, :], dgood[maxbin, :]) .> 0 .| vcat(dgood[1, :], dgood[1:(maxbin-1), :]) .> 0)

	p = zeros(size(dgood))
	m = zeros(size(dgood))

	# For each frame, extract all harmonic freqs & magnitudes
	for t ∈ 1:axes(I, 2)
		ds = dgood[:, t]'
		lds = length(ds)
		# find nonzero regions in this vector
		st = findall((ds .> 0) .& ([0; ds[1:(lds-1)]] .== 0))
		en = findall((ds .> 0) .& ([ds[2:lds]; 0] .== 0))
		npks = length(st)
		frqs = zeros(npks)
		mags = zeros(npks)
		for i ∈ 1:axes(st, 1)
			bump = abs.(S[st[i]:en[i], t])
			frqs[i] = (bump' * I[st[i]:en[i], t]) / (sum(bump) + (sum(bump) == 0))
			mags[i] = sum(bump)
			if frqs[i] > fmaxu
				mags[i] = 0
				frqs[i] = 0
			elseif frqs[i] > fmaxl
				mags[i] = mags[i] * max(0, (fmaxu - frqs[i]) / (fmaxu - fmaxl))
			end
			# downweight magnitudes below? 200 Hz
			if frqs[i] < fminl
				mags[i] = 0
				frqs[i] = 0
			elseif frqs[i] < fminu
				# 1 octave fade-out
				mags[i] = mags[i] * (frqs[i] - fminl) / (fminu - fminl)
			end
			if frqs[i] < 0
				mags[i] = 0
				frqs[i] = 0
			end
		end
		# then just keep the largest at each frame (for now)
		bin = round.(Int, (st .+ en) ./ 2)
		p[bin, t] = frqs
		m[bin, t] = mags
	end
	return p, m, S
end

function get_mel_norm_factor(spectrum_type::Symbol, fft_window::Vector{Float64})
	if spectrum_type == :power
		return 1 / (sum(fft_window)^2)
	elseif spectrum_type == :magnitude
		return 1 / sum(fft_window)
	else
		error("Unknown spectrum_type $spectrum_type.")
	end
end

### da generalizzare per 
### frequency_scale :mel, :bark, :erb
### filterbanl_design_domain :linear, :warped (da verificare se serve)
function design_filterbank(data::AudioData, setup::AudioSetup)
	# set the design domain ### da implementare in futuro
	setup.filterbank_design_domain == :linear ? design_domain = :linear : design_domain = setup.frequency_scale

	# compute band edges
	# TODO da inserire il caso :erb e :bark

	fb_range = hz2mel(setup.stft.freq_range, setup.mel_style)

	# mimic audioflux linear mel_style
	if setup.mel_style == :linear
		lin_fq = collect(0:(setup.stft.stft_length-1)) / setup.stft.stft_length * setup.sr
		setup.band_edges = lin_fq[1:(setup.mel_bands+2)]
	else
		setup.band_edges = mel2hz(
			LinRange(fb_range[1], fb_range[end], setup.mel_bands + 2), setup.mel_style)
	end

	### parte esclusiva per mel filterbank si passa a file designmelfilterbank.m
	# determine the number of bands
	num_edges = length(setup.band_edges)

	# determine the number of valid bands
	valid_num_edges = sum((setup.band_edges .- (setup.sr / 2)) .< sqrt(eps(Float64)))
	valid_num_bands = valid_num_edges - 2

	# preallocate the filter bank
	data.mel_filterbank = zeros(Float64, setup.stft.stft_length, setup.mel_bands)
	data.mel_frequencies = setup.band_edges[2:(end-1)]

	# Set this flag to true if the number of FFT length is insufficient to
	# compute the specified number of mel bands
	FFTLengthTooSmall = false

	# if :hz 
	linFq = collect(0:(setup.stft.stft_length-1)) / setup.stft.stft_length * setup.sr

	# Determine inflection points
	@assert(valid_num_edges <= num_edges)
	p = zeros(Float64, valid_num_edges, 1)

	for edge_n in 1:valid_num_edges
		for index in eachindex(linFq)
			if linFq[index] > setup.band_edges[edge_n]
				p[edge_n] = index
				break
			end
		end
	end

	FqMod = linFq

	# Create triangular filters for each band
	bw = diff(setup.band_edges)

	for k in 1:Int(valid_num_bands)
		# Rising side of triangle
		for j in Int(p[k]):(Int(p[k+1])-1)
			data.mel_filterbank[j, k] = (FqMod[j] - setup.band_edges[k]) / bw[k]
		end
		# Falling side of triangle
		for j in Int(p[k+1]):(Int(p[k+2])-1)
			data.mel_filterbank[j, k] = (setup.band_edges[k+2] - FqMod[j]) / bw[k+1]
		end
		emptyRange1 = p[k] .> p[k+1] - 1
		emptyRange2 = p[k+1] .> p[k+2] - 1
		if (!FFTLengthTooSmall && (emptyRange1 || emptyRange2))
			FFTLengthTooSmall = true
		end
	end

	# mirror two sided
	range = get_onesided_fft_range(setup.stft.stft_length)
	range = range[2:end]
	data.mel_filterbank[end:-1:(end-length(range)+1), :] = data.mel_filterbank[range, :]

	data.mel_filterbank = data.mel_filterbank'

	# normalizzazione    
	BW = setup.band_edges[3:end] - setup.band_edges[1:(end-2)]

	if (setup.filterbank_normalization == :area)
		weight_per_band = sum(data.mel_filterbank, dims = 2)
		if setup.frequency_scale != :erb
			weight_per_band = weight_per_band / 2
		end
	elseif (setup.filterbank_normalization == :bandwidth)
		weight_per_band = BW / 2
	else
		weight_per_band = ones(1, setup.mel_bands)
	end

	for i in 1:(setup.mel_bands)
		if (weight_per_band[i] != 0)
			data.mel_filterbank[i, :] = data.mel_filterbank[i, :] ./ weight_per_band[i]
		end
	end

	# get one side
	range = get_onesided_fft_range(setup.stft.stft_length)
	data.mel_filterbank = data.mel_filterbank[:, range]
	# manca la parte relativa a :erb e :bark

	# setta fattore di normalizzazione
	if setup.stft.window_norm
		win_norm_factor = get_mel_norm_factor(setup.spectrum_type, data.stft.stft_window)
		data.mel_filterbank = data.mel_filterbank * win_norm_factor
	end
end # function designMelFilterBank

function create_DCT_matrix(
	mel_coeffs::Int64,
)
	# create DCT matrix
	matrix = zeros(Float64, mel_coeffs, mel_coeffs)
	s0 = sqrt(1 / mel_coeffs)
	s1 = sqrt(2 / mel_coeffs)
	piCCast = 2 * pi / (2 * mel_coeffs)

	matrix[1, :] .= s0
	for k in 1:mel_coeffs, n in 2:mel_coeffs
		matrix[n, k] = s1 * cos(piCCast * (n - 1) * (k - 0.5))
	end

	matrix
end

function audioDelta(
	x::AbstractMatrix{T},
	window_length::Int64,
	source::Symbol = :standard,
) where {T <: AbstractFloat}

	# define window shape
	m = Int(floor(window_length / 2))
	b = collect(m:-1:(-m)) ./ sum((1:m) .^ 2)

	if source == :transposed
		filt(b, 1.0, x')'   #:audioflux setting
	else
		filt(b, 1.0, x)     #:matlab setting
	end
end

################################################################################
#                                    main                                      #
################################################################################
function get_mel_spec!(
	setup::AudioSetup,
	data::AudioData,
)
	if setup.frequency_scale == :mel
		design_filterbank(data, setup)

		hop_length = setup.stft.window_length - setup.stft.overlap_length
		num_hops = Int(floor((size(data.x, 1) - setup.stft.window_length) / hop_length) + 1)

		# apply filterbank
		# if (setup.spectrum_type == :power)
		data.mel_spectrogram = reshape(
			data.mel_filterbank * data.stft.stft, setup.mel_bands, num_hops)
		# else
		#     #TODO
		#     error("magnitude not yet implemented.")
		# end

		data.mel_spectrogram = transpose(data.mel_spectrogram)

	elseif setup.frequency_scale == :chroma
		# reference: https://www.ee.columbia.edu/~dpwe/resources/matlab/chroma-ansyn/#1

		A0 = 27.5 # Hz
		tuning = 440 # Hz
		f_ctr_log = log(setup.center_freq / A0) / log(2)

		fminl = octs2hz(hz2octs(setup.center_freq) - 2 * setup.gaussian_sd)
		fminu = octs2hz(hz2octs(setup.center_freq) - setup.gaussian_sd)
		fmaxl = octs2hz(hz2octs(setup.center_freq) + setup.gaussian_sd)
		fmaxu = octs2hz(hz2octs(setup.center_freq) + 2 * setup.gaussian_sd)

		ffthop = setup.stft.stft_length ÷ 4
		nchr = 12

		p, m, _ = ifptrack(data.x, setup.stft.stft_length, setup.sr, fminl, fminu, fmaxl, fmaxu)

		nbins, ncols = size(p)

		# chroma-quantized IF sinusoids
		Pocts = hz2octs(p .+ (p .== 0))
		Pocts[p.==0] .= 0
		# Figure best tuning alignment
		nzp = findall(p .> 0)
		hn, hx = hist(nchr .* Pocts[nzp] .- round.(nchr .* Pocts[nzp]), 100)
		centsoff = hx[findall(hn .== maximum(hn))]
		# Adjust tunings to align better with chroma
		Pocts[nzp] .-= centsoff[1] / nchr

		# Quantize to chroma bins
		PoctsQ = Pocts
		PoctsQ[nzp] = round.(nchr .* Pocts[nzp]) ./ nchr

		# map IF pitches to chroma bins
		Pmapc = round.(nchr .* (PoctsQ .- floor.(PoctsQ)))
		Pmapc[p.==0] .= -1
		Pmapc[Pmapc.==nchr] .= 0

		Y = zeros(nchr, ncols)
		for t ∈ 1:ncols
			Y[:, t] = (repeat([0:(nchr-1)]; outer = [1, size(Pmapc, 1)]) .== repeat(Pmapc[:, t]'; outer = [nchr, 1])) * m[:, t]
		end
		data.mel_spectrogram = Y
	end
end # melSpectrogram

# TODO prova a fare le delta del log mel
function get_log_mel!(
	setup::AudioSetup,
	data::AudioData,
)
	# Reference:
	# https://dsp.stackexchange.com/questions/85501/log-of-filterbank-energies

	# Rectify
	mel_spec = deepcopy(data.mel_spectrogram')

	if setup.normalization_type == :standard
		mel_spec[mel_spec.==0] .= floatmin(Float64)
	elseif setup.normalization_type == :dithered
		mel_spec[mel_spec.<1e-8] .= 1e-8
	else
		@warn("Unknown $setup.normalization_type normalization type, defaulting to standard.")
		mel_spec[mel_spec.==0] .= floatmin(Float64)
	end

	data.log_mel = log10.(mel_spec)'
end

function get_mfcc!(
	setup::AudioSetup,
	data::AudioData,
)
	# Rectify
	mel_spec = deepcopy(data.mel_spectrogram')

	if setup.normalization_type == :standard
		mel_spec[mel_spec.==0] .= floatmin(Float64)
	elseif setup.normalization_type == :dithered
		mel_spec[mel_spec.<1e-8] .= 1e-8
	else
		@warn("Unknown $setup.normalization_type normalization type, defaulting to standard.")
		mel_spec[mel_spec.==0] .= floatmin(Float64)
	end

	# Design DCT matrix
	DCTmatrix = create_DCT_matrix(setup.mel_bands)

	# apply DCT matrix
	if (setup.rectification == :log)
		coeffs = DCTmatrix * log10.(mel_spec)
	elseif (setup.rectification == :cubic_root)
		# apply DCT matrix
		coeffs = DCTmatrix * mel_spec .^ (1 / 3)
	else
		@warn("Unknown $rectification DCT matrix rectification, defaulting to log.")
		coeffs = DCTmatrix * log10.(mel_spec)
	end

	# reduce to mfcc coefficients
	data.mfcc_coeffs = coeffs[1:(setup.mfcc_coeffs), :]'

	# log energy calc
	if setup.log_energy_source == :mfcc
		log_energy = sum(eachrow(mel_spec .^ 2)) / setup.mel_bands

		if setup.normalization_type == :standard
			log_energy[log_energy.==0] .= floatmin(Float64)
		elseif setup.normalization_type == :dithered
			log_energy[log_energy.<1e-8] .= 1e-8
		end

		data.log_energy = log.(log_energy)
	end

	if (setup.log_energy_pos == :append)
		data.mfcc_coeffs = hcat(data.mfcc_coeffs, data.log_energy)
	elseif (setup.log_energy_pos == :replace)
		data.mfcc_coeffs = hcat(data.log_energy, data.mfcc_coeffs[:, 2:end])
	end
end

function get_mfcc_deltas!(
	setup::AudioSetup,
	data::AudioData,
)
	data.mfcc_delta = audioDelta(
		data.mfcc_coeffs, setup.delta_window_length, setup.delta_matrix)
	data.mfcc_deltadelta = audioDelta(
		data.mfcc_delta, setup.delta_window_length, setup.delta_matrix)
end
