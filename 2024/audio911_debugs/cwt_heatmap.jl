using Revise, Audio911, Plots

TESTPATH = joinpath(dirname(pathof(Audio911)), "..", "test")
TESTFILE = "common_voice_en_23616312.wav"
# TESTFILE = "104_1b1_Al_sc_Litt3200_4.wav"
wavfile = joinpath(TESTPATH, TESTFILE)

sr_src = 16000
X = load_audio(wavfile, sr_src)

wavelet = :bump # :morse, :morlet, :bump
morse_params = (3,60)
vpo = 10

freq_range = (100,8000)

get_stft!(X, norm=:none)
get_lin_spec!(X, X.stft; norm=:none, db_scale=false, freq_range=freq_range)
heatmap(X.lin_spec.spec)

get_cwt_spec!(X, X.audio; wavelet, morse_params, vpo, freq_range, norm=:magnitude, db_scale=true)

get_mel_fb!(X, X.stft, freq_range=freq_range)
get_cwt_fb!(X, X.audio, freq_range=freq_range)
get_cwt_spec!(X, X.mel_fb; norm=:magnitude, db_scale=true)
get_cwt_spec!(X, X.cwt_fb; norm=:magnitude, db_scale=true)

# heatmap(X.lin_spec.spec)
heatmap(X.cwt_spec.spec)

