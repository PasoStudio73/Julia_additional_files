# using Arrow
using DataFrames
using CSV
using JLD2

# af = jldopen("/home/riccardopasini/Documents/Aclai/Datasets/SpcDS/SpcDS_gender_1000_60_100_nbs_mfccExtended.jld2")
# ad, ay = af["dataframe_validated"]
# sa = jldopen("/home/riccardopasini/Documents/Aclai/Datasets/SpcDS/SpcDS_gender_1000_60_100_nbs_mfccAclai.jld2")
# sd, sy = sa["dataframe_validated"]

sa1 = jldopen("/home/riccardopasini/Documents/Aclai/Datasets/LibriSpeech/speaker_recognition/jld2/srTest_1_4.jld2")
s1, s1y = sa1["dataframe_validated"]
# sa2 = jldopen("/home/riccardopasini/Documents/Aclai/Datasets/SpcDS/SpcDS_gender_1000_60_100_nbs_soleaudio.jld2")
# s2, s2y = sa2["dataframe_validated"]

