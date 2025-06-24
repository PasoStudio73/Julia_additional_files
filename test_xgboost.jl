"""
Simple Pure Julia XGBoost Regression Test
"""

using Random, Statistics

# Tree node structure
mutable struct TreeNode
    feature_idx::Union{Int, Nothing}
    threshold::Union{Float64, Nothing}
    left::Union{TreeNode, Nothing}
    right::Union{TreeNode, Nothing}
    value::Union{Float64, Nothing}  # For leaf nodes
    is_leaf::Bool
    
    TreeNode() = new(nothing, nothing, nothing, nothing, nothing, false)
end

# XGBoost Regressor structure
mutable struct XGBoostRegressor
    n_estimators::Int
    max_depth::Int
    learning_rate::Float64
    reg_lambda::Float64  # L2 regularization
    min_child_weight::Float64
    base_score::Float64
    trees::Vector{TreeNode}
    
    function XGBoostRegressor(; n_estimators=10, max_depth=3, learning_rate=0.3,
                             reg_lambda=1.0, min_child_weight=1.0)
        new(n_estimators, max_depth, learning_rate, reg_lambda, 
            min_child_weight, 0.0, TreeNode[])
    end
end

"""
Calculate gradient and hessian for squared error loss
"""
function calculate_gradients_hessians(y_true::Vector{Float64}, y_pred::Vector{Float64})
    gradients = y_pred .- y_true  # ŷ - y
    hessians = ones(length(y_true))  # Always 1 for squared error
    return gradients, hessians
end

"""
Calculate the optimal leaf weight for XGBoost
"""
function calculate_leaf_weight(gradients::Vector{Float64}, hessians::Vector{Float64}, reg_lambda::Float64)
    G = sum(gradients)
    H = sum(hessians)
    return -G / (H + reg_lambda)
end

"""
Calculate gain for a potential split
"""
function calculate_gain(grad_left::Vector{Float64}, hess_left::Vector{Float64},
                       grad_right::Vector{Float64}, hess_right::Vector{Float64},
                       grad_all::Vector{Float64}, hess_all::Vector{Float64}, reg_lambda::Float64)
    
    G_L = sum(grad_left)
    H_L = sum(hess_left)
    G_R = sum(grad_right)
    H_R = sum(hess_right)
    G = sum(grad_all)
    H = sum(hess_all)
    
    gain = 0.5 * ((G_L^2 / (H_L + reg_lambda)) + (G_R^2 / (H_R + reg_lambda)) - (G^2 / (H + reg_lambda)))
    return gain
end

"""
Find the best split for a node
"""
function find_best_split(X::Matrix{Float64}, gradients::Vector{Float64}, hessians::Vector{Float64}, 
                        indices::Vector{Int}, reg_lambda::Float64, min_child_weight::Float64)
    
    best_gain = -Inf
    best_feature = nothing
    best_threshold = nothing
    best_left_indices = Int[]
    best_right_indices = Int[]
    
    n_features = size(X, 2)
    
    # Try each feature
    for feature_idx in 1:n_features
        # Get unique values for this feature
        feature_values = unique(X[indices, feature_idx])
        
        if length(feature_values) < 2
            continue
        end
        
        # Try each possible threshold
        for i in 1:(length(feature_values)-1)
            threshold = (feature_values[i] + feature_values[i+1]) / 2
            
            # Split indices based on threshold
            left_indices = Int[]
            right_indices = Int[]
            
            for idx in indices
                if X[idx, feature_idx] < threshold
                    push!(left_indices, idx)
                else
                    push!(right_indices, idx)
                end
            end
            
            # Check minimum child weight constraint
            if length(left_indices) < min_child_weight || length(right_indices) < min_child_weight
                continue
            end
            
            # Calculate gain
            grad_left = gradients[left_indices]
            hess_left = hessians[left_indices]
            grad_right = gradients[right_indices]
            hess_right = hessians[right_indices]
            grad_all = gradients[indices]
            hess_all = hessians[indices]
            
            gain = calculate_gain(grad_left, hess_left, grad_right, hess_right, 
                                grad_all, hess_all, reg_lambda)
            
            if gain > best_gain
                best_gain = gain
                best_feature = feature_idx
                best_threshold = threshold
                best_left_indices = copy(left_indices)
                best_right_indices = copy(right_indices)
            end
        end
    end
    
    return best_gain, best_feature, best_threshold, best_left_indices, best_right_indices
end

"""
Build a decision tree for XGBoost
"""
function build_tree(X::Matrix{Float64}, gradients::Vector{Float64}, hessians::Vector{Float64}, 
                   indices::Vector{Int}, max_depth::Int, current_depth::Int,
                   reg_lambda::Float64, min_child_weight::Float64)
    
    node = TreeNode()
    
    # Check stopping criteria
    if current_depth >= max_depth || length(indices) < 2 * min_child_weight
        # Create leaf node
        node.is_leaf = true
        node.value = calculate_leaf_weight(gradients[indices], hessians[indices], reg_lambda)
        return node
    end
    
    # Find best split
    gain, feature_idx, threshold, left_indices, right_indices = 
        find_best_split(X, gradients, hessians, indices, reg_lambda, min_child_weight)
    
    # If no good split found, create leaf
    if gain <= 0 || isnothing(feature_idx)
        node.is_leaf = true
        node.value = calculate_leaf_weight(gradients[indices], hessians[indices], reg_lambda)
        return node
    end
    
    # Create internal node
    node.feature_idx = feature_idx
    node.threshold = threshold
    node.left = build_tree(X, gradients, hessians, left_indices, max_depth, 
                          current_depth + 1, reg_lambda, min_child_weight)
    node.right = build_tree(X, gradients, hessians, right_indices, max_depth, 
                           current_depth + 1, reg_lambda, min_child_weight)
    
    return node
end

"""
Predict using a single tree
"""
function predict_tree(tree::TreeNode, x::Vector{Float64})
    if tree.is_leaf
        return tree.value
    end
    
    if x[tree.feature_idx] < tree.threshold
        return predict_tree(tree.left, x)
    else
        return predict_tree(tree.right, x)
    end
end

"""
Fit the XGBoost model
"""
function fit!(model::XGBoostRegressor, X::Matrix{Float64}, y::Vector{Float64})
    n_samples = length(y)
    
    # Initialize base score as mean of target values
    model.base_score = mean(y)
    
    # Initialize predictions with base score
    y_pred = fill(model.base_score, n_samples)
    
    # Clear any existing trees
    empty!(model.trees)
    
    # Build trees iteratively
    for iter in 1:model.n_estimators
        # Calculate gradients and hessians
        gradients, hessians = calculate_gradients_hessians(y, y_pred)
        
        # Build tree to fit negative gradients
        tree = build_tree(X, gradients, hessians, collect(1:n_samples), 
                         model.max_depth, 0, model.reg_lambda, model.min_child_weight)
        
        push!(model.trees, tree)
        
        # Update predictions
        for i in 1:n_samples
            tree_pred = predict_tree(tree, X[i, :])
            y_pred[i] += model.learning_rate * tree_pred
        end
        
        # Calculate loss for monitoring
        loss = mean((y .- y_pred).^2) / 2
        println("Round $iter, Loss: $(Base.round(loss, digits=6))")
    end
end

"""
Make predictions
"""
function predict_xgb(model::XGBoostRegressor, X::Matrix{Float64})
    n_samples = size(X, 1)
    predictions = fill(model.base_score, n_samples)
    
    # Add contribution from each tree
    for tree in model.trees
        for i in 1:n_samples
            tree_pred = predict_tree(tree, X[i, :])
            predictions[i] += model.learning_rate * tree_pred
        end
    end
    
    return predictions
end

"""
Calculate mean squared error
"""
function mse(y_true::Vector{Float64}, y_pred::Vector{Float64})
    return mean((y_true .- y_pred).^2)
end

"""
Calculate R²
"""
function r2_score(y_true::Vector{Float64}, y_pred::Vector{Float64})
    ss_res = sum((y_true .- y_pred).^2)
    ss_tot = sum((y_true .- mean(y_true)).^2)
    return 1 - (ss_res / ss_tot)
end

# Test the implementation
function test_xgboost()
    println("Testing Pure Julia XGBoost Implementation")
    println("=" ^ 50)
    
    # Create synthetic dataset
    Random.seed!(42)
    n_samples = 1000
    n_features = 5
    
    # Generate synthetic regression data
    X = randn(n_samples, n_features)
    # True relationship: y = 2*x1 + 3*x2 - x3 + 0.5*x4 + x5 + noise
    y = 2*X[:, 1] + 3*X[:, 2] - X[:, 3] + 0.5*X[:, 4] + X[:, 5] + 0.1*randn(n_samples)
    
    # Split data
    n = length(y)
    train_size = Int(0.8 * n)
    indices = randperm(n)
    train_idx = indices[1:train_size]
    test_idx = indices[train_size+1:end]
    
    X_train = X[train_idx, :]
    y_train = y[train_idx]
    X_test = X[test_idx, :]
    y_test = y[test_idx]
    
    println("Dataset: Synthetic Regression Data")
    println("Training samples: $(length(y_train))")
    println("Test samples: $(length(y_test))")
    println("Features: $(size(X_train, 2))")
    println()
    
    # Train XGBoost model
    println("Training XGBoost...")
    model = XGBoostRegressor(
        n_estimators=50,
        max_depth=6,
        learning_rate=0.1,
        reg_lambda=1.0,
        min_child_weight=1.0
    )
    
    @time fit!(model, X_train, y_train)
    
    # Make predictions
    y_pred_train = predict_xgb(model, X_train)
    y_pred_test = predict_xgb(model, X_test)
    
    # Calculate metrics
    train_mse = mse(y_train, y_pred_train)
    test_mse = mse(y_test, y_pred_test)
    train_r2 = r2_score(y_train, y_pred_train)
    test_r2 = r2_score(y_test, y_pred_test)
    
    println("\nResults:")
    println("Training MSE: $(Base.round(train_mse, digits=4))")
    println("Test MSE: $(Base.round(test_mse, digits=4))")
    println("Training R²: $(Base.round(train_r2, digits=4))")
    println("Test R²: $(Base.round(test_r2, digits=4))")
    
    # Test with simple data to verify correctness
    println("\n" * "="^50)
    println("Testing with simple linear data")
    
    # Simple test case: y = 2*x + 1
    X_simple = reshape(collect(1.0:10.0), 10, 1)
    y_simple = 2.0 * X_simple[:, 1] .+ 1.0
    
    model_simple = XGBoostRegressor(n_estimators=20, max_depth=3, learning_rate=0.1)
    fit!(model_simple, X_simple, y_simple)
    
    y_pred_simple = predict_xgb(model_simple, X_simple)
    simple_mse = mse(y_simple, y_pred_simple)
    
    println("Simple linear test MSE: $(Base.round(simple_mse, digits=6))")
    println("Expected vs Predicted (first 5):")
    for i in 1:5
        println("  $(y_simple[i]) vs $(Base.round(y_pred_simple[i], digits=3))")
    end
end

# Run the test
test_xgboost()
