using Audio911
using FFTW

TESTPATH = joinpath(dirname(pathof(Audio911)), "..", "test")
TESTFILE = "common_voice_en_23616312.wav"

sr_src = 16000
x, sr = load_audio(joinpath(TESTPATH, TESTFILE), sr = sr_src)

dct_type = 2
dim, n = first_non_singleton_dimension(Float64.(x))

function first_non_singleton_dimension(x::AbstractArray{Float64})
	sz = size(x)
	dim = findfirst(x -> x != 1, sz)
	if isnothing(dim)
		dim = 1
		n = 1
	else
		transform_length = sz[dim]
	end
	return dim, transform_length
end

function create_DCT_matrix(
	x_size::Int64,
)
	# create DCT matrix
	matrix = zeros(Float64, x_size, x_size)
	s0 = sqrt(1 / x_size)
	s1 = sqrt(2 / x_size)
	piCCast = 2 * pi / (2 * x_size)

	matrix[1, :] .= s0
	for k in 1:x_size, n in 2:x_size
		matrix[n, k] = s1 * cos(piCCast * (n - 1) * (k - 0.5))
	end

	matrix
end

function get_dct(
	x::AbstractArray{Float64};
	dct_type::Int64 = 2,
	dim::Int64 = 0,
	n::Int64 = 0,
)
	if dim == 0 && n == 0
		dim, n = first_non_singleton_dimension(x)
	elseif n == 0
		n = size(x, dim)
	elseif dim == 0
		dim, _ = first_non_singleton_dimension(x)
	end

	scale = sqrt.([1 / (2 * (n - 1)), 1 / (2 * n), 1 / (2 * n), 1 / (2 * n)])
	dcscale = sqrt.([2, 1 / 2, 2, 1])

    DCT_matrix = create_DCT_matrix(n)

    ### DCT type 1
    if dct_type == 1
        x = x .* scale(type)
        idc = dimselect(dim, size(x))
        x[1 .+ idc] = x[1 .+ idc] * dcscale(type)
        if size(x, dim) >= n
            x[end .- idc] = x[end .- idc] * dcscale(type)
        end
    ### DCT type 3
    elseif dct_type == 3
        x = x .* scale(type)
        idc = 1 .+ dimselect(dim, size(x))
        x[idc] = x[idc] * dcscale(type)
    end


end

function get_dct(x::AbstractArray{T}; kwargs...) where {T <: AbstractFloat}
	get_dct(Float64.(x); kwargs...)
end
