# -------------------------------------------------------------------------- #
#                                   debug                                    #
# -------------------------------------------------------------------------- #
using Revise, Audio911, BenchmarkTools
# using Plots, Parameters, FFTW, DSP, StatsBase, NaNStatistics
# using Unitful, NamedArrays

TESTPATH = joinpath(dirname(pathof(Audio911)), "..", "test")
TESTFILE = "common_voice_en_23616312.wav"; sr = 16000
# TESTFILE = "104_1b1_Al_sc_Litt3200_4.wav"; sr = 16000
# TESTFILE = "03-01-02-01-02-01-05.wav"; sr = 48000
wavfile = joinpath(TESTPATH, TESTFILE)

audio = load_audio(wavfile, sr, norm=true);
save_audio(audio, "/home/paso/Documents/Aclai/Julia_additional_files/audio911_tests/speech.wav")

# matlab standard
a, outidx = speech_detector(audio)
save_audio(a, "/home/paso/Documents/Aclai/Julia_additional_files/audio911_tests/speech_detector.wav")

# various args
a, outidx = speech_detector(audio, nfft=4096, nwin=4096)
save_audio(a, "/home/paso/Documents/Aclai/Julia_additional_files/audio911_tests/speech_detector.wav")

a, outidx = speech_detector(audio, merge_distance=10000)
save_audio(a, "/home/paso/Documents/Aclai/Julia_additional_files/audio911_tests/speech_detector.wav")

a, outidx = speech_detector(audio, thresholds=(0,0))
save_audio(a, "/home/paso/Documents/Aclai/Julia_additional_files/audio911_tests/speech_detector.wav")

a, outidx = speech_detector(audio, thresholds=(0,0), spread_threshold=0.02)
save_audio(a, "/home/paso/Documents/Aclai/Julia_additional_files/audio911_tests/speech_detector.wav")