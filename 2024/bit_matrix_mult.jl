# ----------------------------------------------------------- #
#              Bit Matrix Multiplication in Julia             #
# ----------------------------------------------------------- #
using BenchmarkTools
using LinearAlgebra
using LoopVectorization
# using Random

s = 100

a_bool = rand(Bool, s, s)
b_bool = rand(Bool, s, s)

@btime c_bool = a_bool * b_bool
# 143.741 μs (3 allocations: 78.21 KiB)

a_bm = BitMatrix(rand(Bool, s, s))
b_bm = BitMatrix(rand(Bool, s, s))

@btime c_bm = a_bm * b_bm
# 573.465 μs (3 allocations: 78.21 KiB)

a_int8 = Int8.(a_bool)
b_int8 = Int8.(b_bool)

a_int8 = Int8.(rand(Bool, s, s))
b_int8 = Int8.(rand(Bool, s, s))
@btime c_int8 = a_int8 * b_int8
# 78.045 μs (3 allocations: 9.90 KiB)

a_f32 = Float32.(rand(Bool, s, s))
b_f32 = Float32.(rand(Bool, s, s))
@btime c_f32 = a_f32 * b_f32
# 25.077 μs (3 allocations: 39.15 KiB)

@btime begin
    a_bool = rand(Bool, s, s)
    b_bool = rand(Bool, s, s)
    c_bool = a_bool * b_bool
end
# 159.436 μs (9 allocations: 98.01 KiB)

@btime begin
    a_int8 = Int8.(rand(Bool, s, s))
    b_int8 = Int8.(rand(Bool, s, s))
    c_int8 = a_int8 * b_int8
end
# 111.020 μs (15 allocations: 49.49 KiB)

@btime begin
    a_f32 = Float32.(rand(Bool, s, s))
    b_f32 = Float32.(rand(Bool, s, s))
    c_f32 = a_f32 * b_f32
end
# 51.349 μs (15 allocations: 137.24 KiB)

@btime begin
    c_turbo = zeros(Int, s, s)
    @turbo for j in 1:s, k in 1:s, i in 1:s
        c_turbo[i, j] += a_bool[i, k] * b_bool[k, j]
    end
end
# 348.068 μs (38 allocations: 79.46 KiB)

@btime begin
    c_turbo = zeros(Float32, s, s)
    @turbo for j in 1:s, k in 1:s, i in 1:s
        c_turbo[i, j] += a_f32[i, k] * b_f32[k, j]
    end
end
# 40.840 μs (37 allocations: 40.38 KiB)

@btime begin 
    c_workspace = similar(a_f32)
    mul!(c_workspace, a_f32, b_f32)
end
# 23.810 μs (3 allocations: 39.15 KiB)

@btime begin 
    c_blas_workspace = similar(a_f32)
    BLAS.gemm!('N', 'N', 1.0f0, a_f32, b_f32, 0.0f0, c_blas_workspace)
end
# 24.818 μs (3 allocations: 39.15 KiB)
