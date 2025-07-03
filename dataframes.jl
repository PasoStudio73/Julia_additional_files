using Test
using MLJ, SoleXplorer
using DataFrames, Random
using SoleData
const SX = SoleXplorer

Xc, yc = @load_iris
Xc = DataFrame(Xc)

# Xr, yr = @load_boston
# Xr = DataFrame(Xr)

Tree = @load DecisionTreeClassifier pkg=DecisionTree
tree = Tree()

@show typeof(tree)
@show typeof(Xc)
@show typeof(yc)

e1f = evaluate(
    tree, Xc, yc;
    resampling=CV(shuffle=false),
    measures=[log_loss, accuracy],
    per_observation=false,
    verbosity=0
)

@btime begin
    Tree = @load DecisionTreeClassifier pkg=DecisionTree
    tree = Tree()
    e1f = evaluate(
        tree, Xc, yc;
        resampling=CV(shuffle=true),
        measures=[log_loss, accuracy],
        per_observation=false,
        verbosity=0
    )
end
# 1.734 ms (10593 allocations: 649.28 KiB)

@btime begin
    model, _, _ = symbolic_analysis(
        Xc, yc;
        model=(;type=:decisiontree),
        resample = (type=CV, params=(;shuffle=true)),
        measures=(log_loss, accuracy)
    )
end

# Xts, yts = SoleData.load_arff_dataset("NATOPS")

_, ds = prepare_dataset(
    Xc, yc;
    model=(;type=:decisiontree),
    resample = (type=Holdout, params=(;shuffle=true)),
    preprocess=(;train_ratio=0.7, rng=Xoshiro(1)),
)



using StaticArrays

@btime begin
    a = nrow(Xc)
    b = ncol(Xc)
    c = Array(Xc)
    D = SizedArray{Tuple{a,b}}(c)
end
# 4.008 μs (16 allocations: 5.32 KiB)

@btime begin
    D = SizedArray{Tuple{nrow(Xc),ncol(Xc)}}(Array(Xc))
end
# 4.000 μs (16 allocations: 5.32 KiB)

@btime begin
    D = Array(Xc)
end
# 1.724 μs (5 allocations: 4.85 KiB)

# check_row_consistency
@btime begin
    for row in eachrow(Xc)
        # skip cols with only scalar values
        any(el -> el isa AbstractArray, row) || continue
        
        # find first array element to use as reference
        ref_idx = findfirst(el -> el isa AbstractArray, row)
        ref_idx === nothing && continue
        
        ref_size = size(row[ref_idx])
        
        # check if any array element has different size (short-circuit)
        if any(row) do el
                el isa AbstractArray && size(el) != ref_size
            end
            return false
        end
    end
    return true
end
# 141.105 μs (2401 allocations: 65.64 KiB)

D = SizedArray{Tuple{nrow(Xc),ncol(Xc)}}(Array(Xc))
@btime begin
    for row in eachrow(D)
        # skip cols with only scalar values
        any(el -> el isa AbstractArray, row) || continue
        
        # find first array element to use as reference
        ref_idx = findfirst(el -> el isa AbstractArray, row)
        ref_idx === nothing && continue
        
        ref_size = size(row[ref_idx])
        
        # check if any array element has different size (short-circuit)
        if any(row) do el
                el isa AbstractArray && size(el) != ref_size
            end
            return false
        end
    end
    return true
end
# 51.758 μs (601 allocations: 25.81 KiB)

M = Array(Xc)
@btime begin
    for row in eachrow(M)
        # skip cols with only scalar values
        any(el -> el isa AbstractArray, row) || continue
        
        # find first array element to use as reference
        ref_idx = findfirst(el -> el isa AbstractArray, row)
        ref_idx === nothing && continue
        
        ref_size = size(row[ref_idx])
        
        # check if any array element has different size (short-circuit)
        if any(row) do el
                el isa AbstractArray && size(el) != ref_size
            end
            return false
        end
    end
    return true
end
# 36.963 μs (601 allocations: 28.16 KiB)

