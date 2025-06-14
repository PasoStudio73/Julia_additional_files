using Revise, Audio911, BenchmarkTools
using DSP
using MFCC
# using Plots, Parameters, FFTW, DSP, StatsBase, NaNStatistics
# using Unitful, NamedArrays

TESTPATH = joinpath(dirname(pathof(Audio911)), "..", "test")
TESTFILE = "common_voice_en_23616312.wav"
# TESTFILE = "104_1b1_Al_sc_Litt3200_4.wav"
wavfile = joinpath(TESTPATH, TESTFILE)

audio = load_audio(source=wavfile, sr=16000)

@btime Audio911.mfcc(audio, sr=16000);
@btime MFCC.mfcc(audio.data, 16000);

@btime s1 = get_stft(source=audio, nfft=512, wintype=(:hann, :periodic));
@btime s2 = DSP.stft(audio.data, 512, 256, nothing; onesided=true, nfft=512, fs=16000, window=hanning);

