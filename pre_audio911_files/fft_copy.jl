function get_onesided_fft_range(stft_length::Int64)
	if mod(stft_length, 2) == 0
		return collect(1:Int(stft_length / 2 + 1))   # EVEN
	else
		return collect(1:Int((stft_length + 1) / 2))  # ODD
	end
end # get_onesided_fft_range

#------------------------------------------------------------------------------#
#              fft version 1 as used in audio features extraction              #
#------------------------------------------------------------------------------#
function _get_stft(x::AbstractArray{Float64}, setup::AudioSetup)
	hop_length = setup.stft.window_length - setup.stft.overlap_length
	if isempty(setup.stft.window)
		setup.stft.window, _ = gencoswin(setup.stft.window_type[1], setup.stft.window_length, setup.stft.window_type[2])
	end

	# split in windows
	y = buffer(x, setup.stft.window_length, setup.stft.window_length - setup.stft.overlap_length)

	# apply window and take fft
	Z = fft(y .* setup.stft.window, (1,))

	# take one side
	logical_ossb = falses(setup.stft.stft_length)
	logical_ossb[get_onesided_fft_range(setup.stft.stft_length)] .= true
	Z = Z[logical_ossb, :]

	# log energy
	# reference: ETSI ES 201 108 V1.1.2 (2000-04)
	# https://www.3gpp.org/ftp/tsg_sa/TSG_SA/TSGS_13/Docs/PDF/SP-010566.pdf
	if setup.log_energy_pos != :none && setup.log_energy_source == :standard
		log_energy = sum(eachrow(y .^ 2))

		if setup.normalization_type == :standard
			log_energy[log_energy.==0] .= floatmin(Float64)
		elseif setup.normalization_type == :dithered
			log_energy[log_energy.<1e-8] .= 1e-8
		end
		log_energy = log.(log_energy)
	else
		log_energy = Vector{Float64}()
	end

	if setup.spectrum_type == :power
		real(Z .* conj(Z)), log_energy
	elseif setup.spectrum_type == :magnitude
		abs.(Z), log_energy
	else
		error("Unknown spectrum type: $(setup.spectrum_type)")
	end
end

get_stft!(setup::AudioSetup, data::AudioData) = data.stft.stft, data.log_energy = _get_stft(data.x, setup)

#------------------------------------------------------------------------------#
#                                   windowing                                  #
#------------------------------------------------------------------------------#
function _get_buffered_vector(
	x::AbstractVector{Float64};
	win_type::Tuple{Symbol, Symbol},
	win_length::Int64,
	overlap_length::Int64,
)
	frames = buffer(x, win_length, win_length - overlap_length)
	window, _ = gencoswin(win_type[1], win_length, win_type[2])

	return frames .* window, window
end

function _get_buffered_vector(x::AbstractVector{Float64}, s::AudioSetup)
	_get_buffered_vector(x, win_type = s.win_type, win_length = s.window_length, overlap_length = s.overlap_length)
end

#------------------------------------------------------------------------------#
#                                     stft                                     #
#------------------------------------------------------------------------------#
function _get_stft(
	x::AbstractArray{Float64},
	sr::Int64;
	stft_length::Int64,
	window::AbstractVector{Float64},
	freq_range::Tuple{Int64, Int64},
	spectrum_type::Symbol,
)
	@assert stft_length >= size(x, 1) "stft_length must be > window length. Got stft_length = $stft_length, window length = $(size(x,1))."

	# ensure x is of length stft_length
	# if the FFT window is larger than the window, the audio data will be zero-padded to match the size of the FFT window.
	# this zero-padding in the time domain results in an interpolation in the frequency domain, 
	# which can provide a more detailed view of the spectral content of the signal.
	x = size(x, 1) < stft_length ? vcat(x, zeros(eltype(x), stft_length - size(x, 1), size(x, 2))) : x[1:stft_length, :]

	# get fft
	Y = fft(x, (1,))

	# post process
	# trim to desired range
	bin_low = ceil(Int, freq_range[1] * stft_length / sr + 1)
	bin_high = floor(Int, freq_range[2] * stft_length / sr + 1)
	bins = collect(bin_low:bin_high)
	y = Y[bins, :]

	# convert to half-sided power or magnitude spectrum
	spectrum_funcs = Dict(
		:power => x -> (x .* conj.(x)) / (0.5 * sum(window^2)),
		:magnitude => x -> abs.(x) / (0.5 * sum(window)),
	)
	# check if spectrum_type is valid
	@assert haskey(spectrum_funcs, spectrum_type) "Unknown spectrum_type: $spectrum_type."

	y = spectrum_funcs[spectrum_type](y)

	# trim borders
	# halve the first bin if it's the lowest bin
	bin_low == 1 && (y[1, :] *= 0.5)
	# halve the last bin if it's the Nyquist bin and FFT length is even
	bin_high == fld(stft_length, 2) + 1 && iseven(stft_length) && (y[end, :] *= 0.5)

	# create frequency vector
	stft_freq = (sr / stft_length) * (bins .- 1)
	# shift final bin if fftLength is odd and the final range is full to fs/2.
	if stft_length % 2 != 0 && bin_high == floor(fftLength / 2 + 1)
		stft_freq[end] = sr * (stft_length - 1) / (2 * stft_length)
	end
	
	return y, stft_freq
end

_get_stft(x::AbstractArray{Float64}, s::AudioSetup) = _get_stft(
	x, 
	s.sr, 
	stft_length = s.stft_length, 
	window = s.window, 
	freq_range = s.freq_range, 
	spectrum_type = s.spectrum_type
	)

get_stft(x::AbstractArray{<:AbstractArray}, sr::Int64; kwargs...) = _get_stft(eltype(x) == Float64 ? x : Float64.(x), sr; kwargs...)

get_stft!(a::AudioObj) = a.data.stft, a.setup.stft_freq = _get_stft(a.data.x, a.setup)