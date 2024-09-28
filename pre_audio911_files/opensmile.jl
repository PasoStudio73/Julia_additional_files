using PyCall

# af = pyimport("audioflux")
librosa = pyimport("librosa")
opensmile = pyimport("opensmile")

sr_src = 8000
audio, sr = librosa.load("/home/riccardopasini/Documents/Aclai/Julia_additional_files/test.wav", sr=sr_src, mono=true)

smile = opensmile.Smile(
    feature_set=opensmile.FeatureSet.eGeMAPSv02,
    feature_level=opensmile.FeatureLevel.Functionals,
)
eGeMAPSv02_Functionals = smile.feature_names

smile = opensmile.Smile(
    feature_set=opensmile.FeatureSet.eGeMAPSv02,
    feature_level=opensmile.FeatureLevel.LowLevelDescriptors,
)
eGeMAPSv02_LowLevelDescriptors = smile.feature_names

lld = smile.process_signal(
    audio,
    sr
)

smile = opensmile.Smile(
    feature_set=opensmile.FeatureSet.ComParE_2016,
    feature_level=opensmile.FeatureLevel.Functionals
)
ComParE_2016_Functionals = smile.feature_names

smile = opensmile.Smile(
    feature_set=opensmile.FeatureSet.ComParE_2016,
    feature_level=opensmile.FeatureLevel.LowLevelDescriptors
)
ComParE_2016_LowLevelDescriptors = smile.feature_names
lld = smile.process_signal(
    audio,
    sr
)

smile = opensmile.Smile(
    feature_set=opensmile.FeatureSet.ComParE_2016,
    feature_level=opensmile.FeatureLevel.LowLevelDescriptors_Deltas
)
ComParE_2016_LowLevelDescriptors_Deltas = smile.feature_names

# smile.process_signal(
#     audio,
#     sr
# )

# open("openSmile_features_sets.txt", "w") do file
#     println(file, "OpenSMILE features sets")
#     println(file, "")
#     println(file, "eGeMAPSv02_Functionals")
#     println(file, eGeMAPSv02_Functionals)

#     println(file, "")
#     println(file, "eGeMAPSv02_LowLevelDescriptors")
#     println(file, eGeMAPSv02_LowLevelDescriptors)

#     println(file, "")
#     println(file, "ComParE_2016_Functionals")
#     println(file, ComParE_2016_Functionals)

#     println(file, "")
#     println(file, "ComParE_2016_LowLevelDescriptors")
#     println(file, ComParE_2016_LowLevelDescriptors)

#     println(file, "")
#     println(file, "ComParE_2016_LowLevelDescriptors_Deltas")
#     println(file, ComParE_2016_LowLevelDescriptors_Deltas)
# end