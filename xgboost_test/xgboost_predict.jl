using MLJ
using DataFrames
using MLJXGBoostInterface
using SoleModels
import XGBoost as XGB
using CategoricalArrays
using Random

function predict_xgboost_bag(trees, X; n_classes=0, objective="binary:logistic")
    n_samples = size(X, 1)
    ntree_limit = length(trees)
    n_classes == 0 && throw(ArgumentError("n_classes must be specified for multi-class predictions"))
    
    # Initialize predictions
    if startswith(objective, "multi:softprob") || startswith(objective, "multi:softmax")
        # For multi-class probabilities, we need a matrix
        raw_preds = zeros(Float64, n_samples, n_classes)
    else
        # For binary and regression, a vector is sufficient
        raw_preds = zeros(Float64, n_samples)
    end
    
    # Iterate through trees and accumulate predictions
    for i in 1:ntree_limit
        tree = trees[i]
        tree_preds = predict_tree(tree, X)
        @show i, tree_preds
        if startswith(objective, "multi:softprob") || startswith(objective, "multi:softmax")
            # For multi-class softprob, each tree outputs predictions for a specific class
            class_idx = (i - 1) % n_classes + 1
            raw_preds[:, class_idx] .+= tree_preds
        else
            # For binary or regression, simply add the predictions
            raw_preds .+= tree_preds
        end
        @show raw_preds
    end
    # Apply appropriate transformation based on objective
    if objective == "binary:logistic"
        # Apply sigmoid transformation
        return 1.0 ./ (1.0 .+ exp.(-raw_preds))
    elseif objective == "multi:softprob"
        # Apply softmax transformation
        exp_preds = exp.(raw_preds)
        # exp_preds = raw_preds
        row_sums = sum(exp_preds, dims=2)
        return exp_preds ./ row_sums
    elseif objective == "multi:softmax"
        # Return class with highest score
        if n_classes > 1
            _, indices = findmax(raw_preds, dims=2)
            return [idx[2] for idx in indices]
        else
            return raw_preds .> 0
        end
    elseif objective == "count:poisson"
        # Apply exponential transformation for Poisson
        return exp.(raw_preds)
    else
        # For regression or other objectives, return raw predictions
        return raw_preds
    end
end

function predict_tree(tree, X)
    n_samples = size(X, 1)
    predictions = zeros(Float64, n_samples)
    
    for i in 1:n_samples
        predictions[i] = traverse_tree(tree, X[i, :])
    end
    return predictions
end

function traverse_tree(tree, x)
    # Start at root node
    node = tree  # Adjust based on your tree structure
    
    # Traverse until reaching a leaf
    while !isempty(node.children)
        # Get the split feature and value
        feature_idx = node.split
        split_value = node.split_condition
        
        # Decide which child to go to
        if x[feature_idx] < split_value
            node = node.children[1]
        else
            node = node.children[2]
        end
    end
    @show node.leaf
    # Return the leaf value
    return node.leaf
    # return exp(node.leaf)
end

X, y = @load_iris
X = DataFrame(X)

seed, num_round, eta = 3, 1, 0.1
rng = Xoshiro(seed)
train, test = partition(eachindex(y), train_ratio; shuffle=true, rng)
X_train, y_train = X[train, :], y[train]
X_test, y_test = X[test, :], y[test]
XGTrees = MLJ.@load XGBoostClassifier pkg=XGBoost
model = XGTrees(; num_round, eta, objective="multi:softprob")
mach = machine(model, X_train, y_train)
fit!(mach)
trees = XGB.trees(mach.fitresult[1])
solem = solemodel(trees, Matrix(X_train), y_train; classlabels, featurenames)
preds = apply(solem, DataFrame(X_test[28,:]))
predsl = CategoricalArrays.levelcode.(categorical(preds)) .- 1

# # For binary classification
# binary_probs = predict_xgboost_bag(mtrs, X_test, objective="binary:logistic")
# binary_preds = binary_probs .> 0.5  # Convert to binary predictions

# # For multi-class classification
rename!(X_test, [:f0, :f1, :f2, :f3])
class_probs = predict_xgboost_bag(trees, DataFrame(X_test[28,:]); n_classes=3, objective="multi:softprob")
class_preds = [argmax(probs) for probs in eachrow(class_probs)] .-1

bst = XGB.xgboost((X_train, yl_train); num_round, eta, num_class=3, objective="multi:softprob")
xtrs = XGB.trees(bst)
yyy = XGB.predict(bst, DataFrame(X_test[28,:]))

class_preds == yyy

# # For regression
# reg_preds = predict_xgboost_bag(mtrs, X_test, objective="reg:squarederror")

# num_round = 20
# eta = 0.3
# yl_train = CategoricalArrays.levelcode.(categorical(y_train)) .- 1
# bst = XGB.xgboost((X_train, yl_train); num_round, eta, num_class=3, objective="multi:softmax")
# ŷ = XGB.predict(bst, X_test)

class_probs = predict_xgboost_bag(trees, DataFrame(X_test[28,:]); n_classes=3, objective="multi:softprob")

# 1×4 DataFrame
#  Row │ sepal_length  sepal_width  petal_length  petal_width 
#      │ Float64       Float64      Float64       Float64     
# ─────┼──────────────────────────────────────────────────────
#    1 │          6.9          3.1           4.9          1.5

# setosa     = -0.072997041   exp 0.9296035807187184
# virginica  =  0.141176477   exp 1.1516278658834618
# versicolor =  0.0239999983  exp 1.024290316149328  

# setosa     = -0.072997041
# virginica  =  0.141176477       
# versicolor =  0.0239999983 