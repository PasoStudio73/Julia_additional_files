# using Arrow
using DataFrames
# using Catch22
using CSV
using JLD2

##
# arrow_path = "/home/riccardopasini/Documents/Aclai/Datasets/SpcDS/"
# # jld2_path = "/home/riccardopasini/results/"

# ds_name = "SpcDS_a20vs70_800_40_100_nbs_opt_matlab_male"

# m_col = 115
# LABEL = "age"
# df = DataFrame(Arrow.Table(string(arrow_path, ds_name, ".arrow")))
# Xdf = select(df, Between(:a1, Symbol(String(:a), m_col)))
# y = df[:, LABEL]

# println("Xdf, numbers of columns: ", size(Xdf, 2), ", rows: ", size(Xdf, 1))

# ## check NaN
# for i in 1:size(Xdf, 1)
#     for j in 1:size(Xdf, 2)
#         for k in 1:length(Xdf[i,j])
#             if (Xdf[i,j][k] == NaN)
#                 error("Trovato NaN in: ", i, " ", j)
#             end
#         end
#     end
# end

d = jldopen("/home/riccardopasini/Documents/Aclai/Datasets/Common_voice_ds/6/spcds_age2split_audio_features_full_female.jld2")
Xdf, Y = d["dataframe_validated"]

# n_NaN = 0
# for i in 1:nrow(Xdf)
#     for j in 1:ncol(Xdf)
#         for k in 1:length(Xdf[i, j])
#             if (Xdf[i, j][k] == NaN)
#                 # n_NaN += 1
#                 println("Found NaN in: ", i, " ", j)
#             end
#         end
#     end
# end

a = 0
b = 0
for i in Y
    if i == "25"
        a += 1
    elseif i == "70"
        b +=1
    end
end