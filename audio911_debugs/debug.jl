# -------------------------------------------------------------------------- #
#                                   debug                                    #
# -------------------------------------------------------------------------- #
using Revise, Audio911, BenchmarkTools
# using Plots, Parameters, FFTW, DSP, StatsBase, NaNStatistics
# using Unitful, NamedArrays

TESTPATH = joinpath(dirname(pathof(Audio911)), "..", "test")
TESTFILE = "common_voice_en_23616312.wav"
# TESTFILE = "104_1b1_Al_sc_Litt3200_4.wav"
wavfile = joinpath(TESTPATH, TESTFILE)

sr = 16000
audio = load_audio(source=wavfile, sr=sr, norm=false);


