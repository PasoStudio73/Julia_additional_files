using Audio911

# function load_audio(
#     filename::AbstractString,
#     sr::Int64=16000
# )
#     Audio(py"load_audio"(filename, sr)...)
# end

# function save_audio(
#     filename::AbstractString,
#     x::AbstractVector{<:AbstractFloat},
#     sr::Int64=16000
# )
#     py"save_audio"(filename, x, sr)
# end

TESTPATH = joinpath(dirname(pathof(Audio911)), "..", "test")
TESTFILE = "common_voice_en_23616312.wav"
# TESTFILE = "104_1b1_Al_sc_Litt3200_4.wav"
wavfile = joinpath(TESTPATH, TESTFILE)

sr_src = 16000
x = load_audio(wavfile, sr_src)