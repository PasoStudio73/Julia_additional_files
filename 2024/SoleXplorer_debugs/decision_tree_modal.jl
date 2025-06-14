using DataFrames, SoleData, SoleBase

df, y = SoleData.load_arff_dataset("NATOPS");
vnames = names(df)
features = [maximum, minimum]

# prop_sole_dt = SoleXplorer.get_solemodel(X, y, :win_decision_tree, features; 
#     nwindows=nwindows, 
#     relative_overlap=relative_overlap, 
#     train_ratio=train_ratio, 
#     rng=rng)

nwindows = 4
relative_overlap = 0.0



X = DataFrame([v => Float64[] for v in [string(j, "(", i, ")w", k) for j in features for i in vnames for k in 1:nwindows]])
win_df = [vcat(SoleBase.movingwindow.(Array(row); nwindows = nwindows, relative_overlap = relative_overlap)...) for row in eachrow(df)]
push!(X, [vcat([map(f, i) for f in features for i in row]...) for row in eachrow(win_df)]...)

for row in eachrow(win_df)
    # for i in row
        a= [vcat([map(f, i) for f in features for i in row]...) for row in eachrow(win_df)]
    # end
end

X = DataFrame([v => Float64[] for v in [string(j, "(", i, ")") for j in features for i in vnames]])
push!(X, [vcat([map(f, Array(row)) for f in features]...) for row in eachrow(df)]...)

for row in eachrow(df)

end