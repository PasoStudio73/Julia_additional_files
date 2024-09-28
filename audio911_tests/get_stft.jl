# -------------------------------------------------------------------------- #
#                           test audio911 get_stft                           #
# -------------------------------------------------------------------------- #
using Revise, Audio911, BenchmarkTools
# using Plots, Parameters, FFTW, StatsBase, NaNStatistics
# using Unitful, NamedArrays

TESTPATH = joinpath(dirname(pathof(Audio911)), "..", "test")
TESTFILE = "common_voice_en_23616312.wav"
# TESTFILE = "104_1b1_Al_sc_Litt3200_4.wav"
wavfile = joinpath(TESTPATH, TESTFILE)

sr = 16000
audio = load_audio(source=wavfile, sr=sr, norm=false);

# benchmark
@btime get_stft(audio)

# matlab afe power spectrogram
s = get_stft(audio, halve=false); s.data.spec
# odd stft
s = get_stft(audio, nfft= 251, halve=false); s.data.spec
# matlab afe magnitude spectrogram
s = get_stft(audio, norm=:magnitude, halve=false); s.data.spec
# nfft > nwin
s = get_stft(audio, nfft=800, nwin=400, halve=false); s.data.spec

