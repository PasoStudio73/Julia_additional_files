

# setup.filterbank_design_domain == :linear ? design_domain = :linear : design_domain = setup.frequency_scale

# weights = zeros(Float64, setup.num_bands, Int(1 + floor(setup.stft.stft_length / 2)))

# fft_frequencies = rfftfreq(setup.stft.stft_length, setup.sr)
# melRange = hz2mel(setup.stft.freq_range, setup.mel_style)
# setup.band_edges = mel2hz(LinRange(melRange[1], melRange[end], setup.num_bands + 2), setup.mel_style)

# fdiff = diff(setup.band_edges)
# ramps = setup.band_edges .- fft_frequencies'

# for i in 1:setup.num_bands
#     lower = -ramps[i] / fdiff[i]
#     upper = ramps[i + 2] / fdiff[i + 1]

#     # .. then intersect them with each other and zero
#     weights[i] = max(0, min(lower, upper))
# end

# for i in range(n_mels):
    # # lower and upper slopes for all bins
    # lower = -ramps[i] / fdiff[i]
    # upper = ramps[i + 2] / fdiff[i + 1]

    # # .. then intersect them with each other and zero
    # weights[i] = np.maximum(0, np.minimum(lower, upper))


# compute band edges
# TODO da inserire il caso :erb e :bark
melRange = hz2mel(setup.stft.freq_range, setup.mel_style)
setup.band_edges = mel2hz(LinRange(melRange[1], melRange[end], setup.num_bands + 2), setup.mel_style)

### parte esclusiva per mel filterbank si passa a file designmelfilterbank.m
# determine the number of bands
num_edges = length(setup.band_edges)

# determine the number of valid bands
valid_num_edges = sum((setup.band_edges .- (setup.sr / 2)) .< sqrt(eps(Float64)))
valid_num_bands = valid_num_edges - 2

# preallocate the filter bank
data.mel_filterbank = zeros(Float64, setup.stft.stft_length, setup.num_bands)
data.mel_frequencies = setup.band_edges[2:end-1]

# Set this flag to true if the number of FFT length is insufficient to
# compute the specified number of mel bands
FFTLengthTooSmall = false

# if :hz 
linFq = collect(0:setup.stft.stft_length-1) / setup.stft.stft_length * setup.sr

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
    for j in Int(p[k]):Int(p[k+1])-1
        data.mel_filterbank[j, k] = (FqMod[j] - setup.band_edges[k]) / bw[k]
    end
    # Falling side of triangle
    for j = Int(p[k+1]):Int(p[k+2])-1
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
data.mel_filterbank[end:-1:end-length(range)+1, :] = data.mel_filterbank[range, :]

# data.mel_filterbank = data.mel_filterbank'