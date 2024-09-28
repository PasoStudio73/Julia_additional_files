using PyCall

librosa = pyimport("librosa")
audio, sr = librosa.load("/home/riccardopasini/Documents/Aclai/Julia_additional_files/test.wav", sr=8000, mono=true)

include("/home/riccardopasini/results/wave-utils/wav-process.jl")

mel_num=26
coeff_num=13

a, b = mfcc_extended(audio, sr, mel_num=mel_num, coeff_num=coeff_num)

vcat.(
    (string("spec", "$i") for i in 1:mel_num)..., 
    (string("mfcc", "$i") for i in 1:coeff_num)...,
    (string("delta", "$i") for i in 1:coeff_num)...,
    (string("ddelta", "$i") for i in 1:coeff_num)...,
    "centroid", "crest", "decrease", "flatness", "flux", "kurtosis", "rolloff", "skewness", "slope", "spread"
    )


label=[]
for i in 1:mel_num
    push!(label, (string("spec", "$i")))
end
for i in 1:coeff_num
    push!(label, (string("mfcc", "$i")))
end
for i in 1:coeff_num
    push!(label, (string("delta", "$i")))
end
for i in 1:coeff_num
    push!(label, (string("ddelta", "$i")))
end
push!(label, "centroid")
push!(label, "crest")
push!(label, "decrease")
push!(label, "flatness")
push!(label, "flux")
push!(label, "kurtosis")
push!(label, "rolloff")
push!(label, "skewness")
push!(label, "slope")
push!(label, "spread")

push!(a, label)
