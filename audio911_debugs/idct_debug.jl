
# -------------------------------------------------------------------------- #
#                                   debug                                    #
# -------------------------------------------------------------------------- #
using Revise, Plots
using Audio911, StaticArrays

# ---------------------------------------------------------------------------- #
#                       audio911 audio features extractor                      #
# ---------------------------------------------------------------------------- #
function afe(x::String, audioparams::NamedTuple)
    # -------------------------------- parameters -------------------------------- #
    # audio module
    sr = audioparams.sr
    norm = false
    speech_detection = false
    # stft module
    nfft = audioparams.nfft
    win_type = (:hann, :periodic)
    win_length = audioparams.nfft
    overlap_length = round(Int, audioparams.nfft / 2)
    stft_norm = :power                      # :power, :magnitude, :pow2mag
    # mel filterbank module
    nbands = audioparams.nbands
    scale = audioparams.scale               # :mel_htk, :mel_slaney, :erb, :bark, :semitones, :tuned_semitones
    melfb_norm = :bandwidth                 # :bandwidth, :area, :none
    freq_range = audioparams.freq_range
    # mel spectrogram module
    db_scale = audioparams.db_scale
    ncoeffs = audioparams.ncoeffs
    rectification = :log                    # :log, :cubic_root
    dither = true

    # --------------------------------- functions -------------------------------- #
    # audio module
    audio = load_audio(
        file=x,
        sr=sr,
        norm=norm,
    );

    stftspec = get_stft(
        audio=audio,
        nfft=nfft,
        win_type=win_type,
        win_length=win_length,
        overlap_length=overlap_length,
        norm=stft_norm
    );

    # mel filterbank module
    melfb = get_melfb(
        stft=stftspec,
        nbands=nbands,
        scale=scale,
        norm=melfb_norm,
        freq_range=freq_range
    );

    # mel spectrogram module
    melspec =  get_melspec(
        stft=stftspec,
        fbank=melfb,
        db_scale=db_scale
    );

    get_mfcc(
        source=melspec,
        ncoeffs=ncoeffs,
        rectification=rectification,
        dither=dither,
    );
end

TESTPATH = joinpath(dirname(pathof(Audio911)), "..", "test")
TESTFILE = "common_voice_en_23616312.wav"
# TESTFILE = "104_1b1_Al_sc_Litt3200_4.wav"
wavfile = joinpath(TESTPATH, TESTFILE)

# extra parameters:
sr = 16000
audioparams = (
    sr=sr,
    nfft = 512,
    win_type = (:hann, :periodic),
    win_length = 512,
    overlap_length = 256,
    nbands = 32 ,
    scale = :mel_htk,
    freq_range = (0, round(Int, sr/2)),
    db_scale = false,
    ncoeffs = 13,
)

mfcc = afe(wavfile, audioparams)

function dimselect(idim::Int64, dim::Tuple{Int64, Int64})
    nel = prod(dim)
    dcterm = prod(dim[1:min(idim-1, 2)])
    
    if idim <= 2
        nskip = dcterm * dim[idim]
    else
        nskip = dcterm
    end
    
    vec([i + j for i in 0:dcterm-1, j in 0:nskip:4095])
end

function idct(; mfcc::Mfcc, type::Int64 = 2)
    coeffs = mfcc.mfcc
    dim = size(coeffs)

    sq2, sq2i, sqn = √2, 1 / √2, √n
    scale = @SVector[sq2 * √(n-1), sq2 * sqn, sq2 * sqn, sq2 * sqn]
    dcscale = @SVector [sq2i, sq2, sq2i, 1.0]

    if type == 1
        error("TODO")
    elseif type == 2
        coeffs .*= scale[type]
        idc = 1 + dimselect()
        x(idc) = x(idc) * dcscale(type)
    elseif type == 3
        error("TODO")
    elseif type == 4
        error("TODO")
    else
        error("type must be 1, 2, 3, or 4.")
    end
end