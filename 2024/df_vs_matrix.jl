using Test
using SoleXplorer
using DataFrames
using StatsBase: sample
using BenchmarkTools

X, y = load_arff_dataset("NATOPS")
num_cols_to_sample, num_rows_to_sample, rng = 10, 50, Xoshiro(11)
chosen_cols = sample(rng, 1:size(X, 2), num_cols_to_sample; replace=false)
chosen_rows = sample(rng, 1:size(X, 1), num_rows_to_sample; replace=false)
X = X[chosen_rows, chosen_cols]
y = y[chosen_rows]

@show X[1, :]
@show y[1:10]


### Matrix Version ###
rng = Xoshiro(11)
d = train_test(X, y; model=(type=:decisiontree,), preprocess=(;rng))
d.model

rng = Xoshiro(11)
m = train_test(X, y; model=(type=:modaldecisiontree,), preprocess=(;rng))
m.model

@btime begin
    rng = Xoshiro(11)
    d = train_test(X, y; model=(type=:decisiontree,), preprocess=(;rng))
end

@btime begin
    rng = Xoshiro(11)
    m = train_test(X, y; model=(type=:modaldecisiontree,), preprocess=(;rng))
end