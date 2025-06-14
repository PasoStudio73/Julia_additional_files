using Sole
using SoleXplorer
using Random, StatsBase, JLD2, DataFrames

filename = "/home/paso/Documents/Aclai/Julia_additional_files/SoleXplorer_debugs/respiratory_Pneumonia.jld2"
df = jldopen(filename)
X, y = df["X"], df["y"]
rng = Random.Xoshiro(1)
train_seed = 11;

####################################################################################################à


using Interpolations

function resample_vector(v::AbstractVector, target_length::Int)
    original_indices = collect(1:length(v))
    target_indices = range(1, length(v), length=target_length)
    itp = interpolate((original_indices,), v, Gridded(Linear()))
    resampled_v = itp(target_indices)
    return resampled_v
end

# Assuming X is your DataFrame
using DataFrames

# Example vector
v = X[1, 2]

# Resample to a target length of 500
target_length = 50
v_resampled = resample_vector(v, target_length)

function calculate_window_params(vec_length::Int, num_windows::Int)
    # Ensure at least windows of size 1
    window_size = max(floor(Int, vec_length / num_windows), 1)
    # Overlap the windows evenly
    step_size = max(floor(Int, (vec_length - window_size) / (num_windows - 1)), 1)
    return window_size, step_size
end

function adaptive_moving_windows(v::AbstractVector, num_windows::Int)
    window_size, step_size = calculate_window_params(length(v), num_windows)
    windows = [v[i:i+window_size-1] for i in 1:step_size:length(v)-window_size+1]
    return windows[1:num_windows]  # Ensure exactly num_windows are returned
end

# Example usage
num_windows = 10
v = X[1, 2]
windows = adaptive_moving_windows(v, num_windows)

function divide_into_segments(v::AbstractVector, num_segments::Int)
    indices = round.(Int, range(1, length(v)+1, length=num_segments+1))
    segments = [v[indices[i]:indices[i+1]-1] for i in 1:num_segments]
    return segments
end

# Example usage
num_segments = 10
v = X[1, 2]
segments = divide_into_segments(v, num_segments)

function pad_or_truncate_vector(v::AbstractVector, target_length::Int)
    if length(v) > target_length
        return v[1:target_length]
    else
        padding = fill(NaN, target_length - length(v))
        return vcat(v, padding)
    end
end

# Example usage
target_length = 500
v = X[1, 2]
v_standardized = pad_or_truncate_vector(v, target_length)

####################################################################################à

function get_moving_windows_with_overlap(v::AbstractVector, nwindows::Int, overlap_percentage::Float64)
    L = length(v)
    overlap = overlap_percentage / 100  # Convert percentage to fraction
    
    # Validate inputs
    if overlap < 0.0 || overlap >= 1.0
        error("Overlap percentage must be between 0% (inclusive) and less than 100%.")
    end
    
    if nwindows < 1
        error("Number of windows (nwindows) must be at least 1.")
    end
    
    # Calculate denominator to compute w
    denominator = (nwindows - 1) * (1 - overlap) + 1
    
    # Avoid division by zero
    if denominator <= 0
        error("Invalid combination of nwindows and overlap leading to non-positive denominator.")
    end
    
    # Calculate maximum window size w
    w = floor(Int, L / denominator)
    w = max(w, 1)  # Ensure window size is at least 1
    
    # Calculate step size s
    s_float = w * (1 - overlap)
    s = max(round(Int, s_float), 1)  # Round to nearest integer, at least 1
    
    # Recalculate total length covered
    total_length = (nwindows - 1) * s + w
    
    # Adjust w and s if necessary
    while total_length > L && w > 1
        w -= 1
        s_float = w * (1 - overlap)
        s = max(round(Int, s_float), 1)
        total_length = (nwindows - 1) * s + w
    end
    
    # Final check
    if total_length > L
        error("Cannot fit the specified number of windows within the vector length with the given overlap.")
    end
    
    # Generate window start indices
    start_indices = [1 + (i - 1) * s for i in 1:nwindows]
    
    # Extract windows
    windows = [v[start:start + w - 1] for start in start_indices]
    
    return windows
end

# Example vector
v = collect(1:100)  # Vector of length 100

# Parameters
nwindows = 5
overlap_percentage = 50.0  # 50% overlap

# Get moving windows
windows = get_moving_windows_with_overlap(v, nwindows, overlap_percentage)

# Display windows
for (i, window) in enumerate(windows)
    println("Window $i (Size $(length(window))): ", window)
end

using DataFrames

function apply_moving_windows_to_dataframe(X::DataFrame, nwindows::Int, overlap_percentage::Float64)
    windows_list = Array{Any, 2}(undef, size(X, 1), size(X, 2))
    for row in 1:size(X, 1)
        for col in 1:size(X, 2)
            v = X[row, col]
            windows = get_moving_windows_with_overlap(v, nwindows, overlap_percentage)
            windows_list[row, col] = windows
        end
    end
    return windows_list
end

# Parameters
nwindows = 5
overlap_percentage = 50.0

# Apply to DataFrame
windows_array = apply_moving_windows_to_dataframe(X, nwindows, overlap_percentage)

# Accessing windows for a specific vector
sample_windows = windows_array[1, 2]  # Windows for vector at X[1, 2]

#####################################################################################à


function calculate_window_params(vec_length::Int, nwindows::Int)
    window_size = max(floor(Int, vec_length / nwindows), 1)
    step_size = max(floor(Int, (vec_length - window_size) / (nwindows - 1)), 1)
    return window_size, step_size
end

function adaptive_moving_windows(intervals::Vector{SoleLogics.Interval{Int64}}; nwindows::Int, overlap::AbstractFloat=0.1, kwargs...)
    (nwindows ≥ 1) || throw(ArgumentError("Number of windows must be at least 1."))
    (0.0 ≤ overlap ≤ 1.0) || throw(ArgumentError("Overlap ratio must be between 0.0 and 1.0"))
    
    total_length = last(intervals).y - first(intervals).x
    total_length ≥ nwindows || throw(ArgumentError("Number of windows ($nwindows) is greater than total length ($total_length)"))
    
    winsize = round(Int, total_length / ((nwindows - 1) * (1 - overlap) + 1))
    _overlap = round(Int, winsize * overlap)

    step = winsize - _overlap
    starts = collect(1:step:((intervals[end].y - winsize)))



    worlds = absolute_movingwindow(intervals; winsize=winsize, overlap=_overlap)

    if length(worlds) < nwindows
        worlds = absolute_movingwindow(intervals; winsize=winsize, overlap=_overlap+1)
    elseif length(worlds) > nwindows && _overlap > 0
        worlds = absolute_movingwindow(intervals; winsize=winsize, overlap=_overlap-1)
    elseif length(worlds) > nwindows
        worlds = absolute_movingwindow(intervals; winsize=winsize+1, overlap=_overlap)
    end

    _total_length = winsize * (1 + (nwindows - 1) * (1 - overlap))

    # Adjust w and s if necessary
    while _total_length > total_length && winsize > 1
        winsize -= 1
        _overlap = round(Int, winsize * overlap)
        _total_length = winsize * (1 + (nwindows - 1) * (1 - overlap))
    end

    _total_length ≤ total_length || throw(ArgumentError("Cannot fit the specified number of windows within the vector length with the given overlap."))
    
    # Final check
    if total_length > L
        error("Cannot fit the specified number of windows within the vector length with the given overlap.")
    end
    
    # Generate window start indices
    start_indices = [1 + (i - 1) * s for i in 1:nwindows]
    
    # Extract windows
    windows = [v[start:start + w - 1] for start in start_indices]
    
    return windows
end
