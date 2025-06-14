module MyPatch

import SplitApplyCombine
using SplitApplyCombine: _inneraxes, _combine_tuples, slice_inds
import SplitApplyCombine: combinedims, _combinedims

@inline function combinedims(a::Base.Generator)
    N = length(SplitApplyCombine.axes(a))
    M = length(SplitApplyCombine._inneraxes(a))
    _combinedims(a, Val(ntuple(i -> N + i, M)))
end

function SplitApplyCombine._combinedims(a::Base.Generator, od::Val{outer_dims}) where {outer_dims}
    firstinner = first(iterate(a))
    return SplitApplyCombine._combinedims(a, od, firstinner)
end

function SplitApplyCombine._combinedims(a::Base.Generator, od::Val{outer_dims}, firstinner::AbstractVector) where {outer_dims}
    outeraxes = axes(a)
    inneraxes = SplitApplyCombine._inneraxes(a)
    ndims_total = length(outeraxes) + length(inneraxes)
    newaxes = SplitApplyCombine._combine_tuples(ndims_total, outer_dims, outeraxes, inneraxes)

    T = eltype(firstinner)
    out = Array{T}(undef, length.(newaxes)...)
    for (j,v) in zip(CartesianIndices(outeraxes),a)
        I = SplitApplyCombine.slice_inds(j, Val(outer_dims), Val(ndims_total))
        view(out, I...) .= v
    end
    return out
end

# function SplitApplyCombine._combinedims(a::Base.Generator, od::Val{outer_dims}, firstinner::UnitRange) where {outer_dims}
#     T = eltype(firstinner)
#     out = Array{T}(undef, length(firstinner), length(a))
#     for (i,v) in enumerate(a)
#         out[:,i] .= v
#     end
#     return out
# end

end

# using SplitApplyCombine
# using BenchmarkTools
# @btime SplitApplyCombine.combinedims((collect(x:x+1000) for x in 1:100000))
# @btime SplitApplyCombine.combinedims(collect((collect(x:x+1000) for x in 1:100000)))


# @code_warntype SplitApplyCombine.combinedims((collect(x:x+1000) for x in 1:100000))
# @code_native SplitApplyCombine.combinedims((collect(x:x+1000) for x in 1:100000))