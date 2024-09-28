using PyCall
using Conda

Conda.add_channel("conda-forge")
# Conda.add_channel("tanky25")
# Conda.add_channel("pytorch")

Conda.rm("audioflux")
Conda.add("librosa")
Conda.add("pywavelets")
Conda.add("scikit-learn")
# Conda.add("pytorch")
# Conda.add("torchaudio")

## destroy conda envoirment
Conda.pip_interop(true)
Conda.pip("install", "tsfel")
# Conda.pip("install", "opensmile")
# Conda.pip("install", "libf0")
# Conda.pip("install", "spafe")

Conda.update()

af = pyimport("audioflux")
librosa = pyimport("librosa")
# wt = pyimport("pywt")
# os = pyimport("opensmile")
# libf0 = pyimport("libf0")
# torch = pyimport("torch")
# torchaudio = pyimport("torchaudio.transforms")
# plt = pyimport("matplotlib.pyplot")

sr_src = 8000
audio, sr = librosa.load("/home/riccardopasini/Documents/Aclai/Julia_additional_files/test.wav", sr=sr_src, mono=true)
