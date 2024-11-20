using Sole, MultiData, MLJ
using SoleXplorer
using StatsBase, Random
using DataFrames
using BenchmarkTools
using Random

# X, y = SoleData.load_arff_dataset("NATOPS");
# @test SoleData.islogiseed(X) == true
# ninst, nvars, vnames = SoleData.ninstances(X), SoleData.nvariables(X), SoleData.varnames(X)

# può avere senso pensa di avere un dataframe dove ogni colonna (feature) potrebbe avere lunghezza, tipo o dimensione differente?
# tipo colonna 1, vettore di misurazioni temporali, colonna 2 vettore di misurazioni spaziali, colonna 3 float con una sola misurazione?

# dataframe di 1 colonna, con vettori di lunghezza variabile
# ipotizzo che i mondi vadano settati feature per feature, quindi colonna per colonna
# e mi metto nella situazione d'avere misurazioni variabili == differente lunghezza
Random.seed!(123)
random_vectors = [rand(Float64, rand(5:10)) for _ in 1:10]
df = DataFrame(feature = random_vectors)

# trova tutti i mondi possibili, basandoti sulla lunghezza massima
vnames = [:feature]
maxsize = argmax(length.(df[!, feature]))
possible_worlds = SoleData.allworlds(df, maxsize)

# filtra i mondi
# f1(x) = length(x) ≥ 3
f1(x) = length(x) == 4
wf = SoleLogics.FunctionalWorldFilter(f1)
filtered_worlds = collect(SoleLogics.filterworlds(wf, possible_worlds))

wf_lf = SoleLogics.IntervalLengthFilter(≥, 3)
filtered_worlds = collect(SoleLogics.filterworlds(wf_lf, possible_worlds))

# genero un dataset con i le features applicate ai mondi filtrati
features = [maximum, mean]
nwindows = length(filtered_worlds)

X = DataFrame([v => Float64[] for v in [string(j, "(", i, ")w", k) for j in features for i in vnames for k in 1:nwindows]])
for row in eachrow(df)
    push!(X, vec(Float64[world.y ≤ length(row[1]) ? f(row[1][world.x:world.y]) : NaN for f in features, world in filtered_worlds]))
end

@show(X)


