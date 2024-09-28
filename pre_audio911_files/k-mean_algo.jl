# to create dataframes and load
using DataFrames
using JLD2
using SoleData
# to use the above function kmeans()
using Clustering
# to visualise our clusters formed
using Plots
# RDatasets to load the already made datasets
using RDatasets

# Functions involved:

# -dataset(): We can access Fisher’s iris data set using this function and passing the arguments as “datasets” and “iris”.
# -Matrix(): Constructs an uninitialized Matrix{T} of size m×n. 
# -collect(): It returns an array of all items in the specified collection or iterator.
# -scatter(): It is used to return scatter plot of dataset.
# -rand(): Pick a random element or array of random elements from the set of values specified and dimensions 
#  and no of numbers is passed as an argument.
# -nclusters(R): This function is used to match the number of clusters formed to the number of clusters we’ve passed 
#  in kmeans() function here R represents the results of returned clusters from kmeans().

# loading the data and storing it in iris dataframe
iris = dataset("datasets", "iris")

# Fetching the each value of data
# using collect() function and
# storing in features
features = collect(Matrix(iris[:, 1:4])')

# running  K-means for the 4 clusters
result = kmeans(features, 4)

# plot with the point color mapped
# to the assigned cluster index
scatter(
    iris.PetalLength,
    iris.PetalWidth,
    marker_z=result.assignments,
    color=:blue,
    legend=false
)
