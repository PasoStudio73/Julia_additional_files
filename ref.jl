using Test
using SoleXplorer
using MLJ
using DataFrames, Random
const SX = SoleXplorer

Xc, yc = @load_iris
Xc = DataFrame(Xc)

modelc = symbolic_analysis(Xc, yc)

root = modelc.sole[1].root

info = (featurenames=[],supporting_predictions=[],supporting_labels=[])
info_ref = Ref(info)

struct P
    s::SX.Branch
    i::Base.RefValue{<:NamedTuple}

    P(s) = new(s, info_ref)  # Use the shared reference
end

a = P(root)
@btime P(root)
# 917.897 ns (2 allocations: 80 bytes)

info_ref[] = modelc.sole[1].info
# Now a.i[] will have the updated info

mutable struct S
    p::P
    info::NamedTuple

    function S(p)
        info = (supporting_predictions=[],supporting_labels=[],featurenames=[],classlabels=[])
        info_ref = Ref(info)
        new(p, info_ref)
    end
end

