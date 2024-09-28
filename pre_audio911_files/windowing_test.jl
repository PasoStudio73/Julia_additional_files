include("/home/riccardopasini/.julia/dev/SoleAudio.jl/src/windowing/windowing.jl")
include("/home/riccardopasini/.julia/dev/SoleAudio.jl/src/windowing/windows.jl")

window_type = :hann_v2
window_length = 256

w, a = gencoswin(window_type, window_length, :symmetric)

