using DataFrames, CSV, JLD2, Catch22
using StatsBase
using SoleBase: nat_sort
using Random

d = jldopen("/home/paso/Documents/Aclai/ItaData2024/jld2_files/respiratory/respiratory_Pneumonia_semitones_14.jld2")
X, Y = d["dataframe_validated"]
split_threshold = 0.8
dataseed = 1

Xsort = deepcopy(X)
Xsort[!, :label] = Y[:, 1]
Xsort[!, :id] = Y[:, 2]
Xsort[!, :index] = 1:nrow(X)

trainset = Vector{Vector{Int64}}()
testset = Vector{Vector{Int64}}()

Random.seed!(dataseed)
for i in groupby(Xsort, :label)
    nsamples = size(i, 1)
    ntrain = round(Int, nsamples * split_threshold)
    ntest = nsamples - ntrain
    train_indices = Int64[]
    test_indices = Int64[]

    unique_ids = unique(i.id)
    shuffled_ids = shuffle(unique_ids)

    n = 0
    k = 1
    while n < ntrain
        sel = i[i.id .== shuffled_ids[k], :index]
        train_indices = vcat(train_indices, sel)
        k += 1
        n += length(sel)
    end

    for j in k:length(shuffled_ids)
        sel = i[i.id .== shuffled_ids[j], :index]
        test_indices = vcat(test_indices, sel)
    end
    trainset = push!(trainset, train_indices)
    testset = push!(testset, test_indices)
end

min_length = minimum(length.(trainset))
shortened_trainset = [v[1:min_length] for v in trainset]
merged_trainset = vcat(shortened_trainset...)

min_length = minimum(length.(testset))
shortened_testset = [v[1:min_length] for v in testset]
merged_testset = vcat(shortened_testset...)

#### TEST ####
bronchie = jldopen("/home/paso/Documents/Aclai/ItaData2024/jld2_files/respiratory/respiratory_Bronchiectasis_semitones_14.jld2")
x, y = bronchie["dataframe_validated"]

train = [30, 40, 57, 10, 31, 52, 61, 58, 59, 75, 3, 16, 45, 54, 86, 9, 29, 7, 14, 17, 33, 36, 37, 53, 78, 82, 5, 42, 47, 64, 74, 12, 21, 34, 38, 70, 76, 22, 26, 32, 48, 51, 56, 83, 90, 4, 25, 28, 73, 81, 79, 6, 23, 87, 27, 71, 72, 13, 19, 68, 2, 50, 55, 43, 67, 89, 15, 65, 66, 8, 63, 85, 119, 122, 137, 144, 146, 147, 155, 156, 165, 171, 175, 178, 96, 132, 135, 157, 163, 103, 104, 107, 113, 134, 138, 159, 160, 161, 166, 170, 180, 93, 94, 95, 98, 99, 100, 101, 102, 106, 108, 109, 111, 114, 116, 117, 118, 120, 121, 123, 124, 125, 127, 128, 130, 131, 133, 140, 141, 142, 143, 145, 149, 153, 158, 162, 169, 173, 176, 97, 105, 110, 115, 139]

y_train = y[train, :]

all_indices = Set(1:size(y,1))
test = collect(setdiff(all_indices, Set(train)))
y_test = y[test, :]

y_check = y[train, 2]
y_checktest = y[test, 2]

overlap = intersect(Set(y_check), Set(y_checktest))