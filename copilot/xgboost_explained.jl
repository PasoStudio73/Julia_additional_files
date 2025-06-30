"""
XGBoost Squared Error Regression - Pure Julia Implementation

This file explains and implements XGBoost for regression with squared error loss.

EXPLANATION:
===========

XGBoost (eXtreme Gradient Boosting) is a gradient boosting framework that builds
an ensemble of decision trees sequentially. Here's how it works for regression:

1. LOSS FUNCTION:
   For regression, XGBoost uses squared error loss:
   L(y, ŷ) = ½(y - ŷ)²

2. GRADIENT AND HESSIAN:
   - Gradient (1st derivative): g = ∂L/∂ŷ = ŷ - y
   - Hessian (2nd derivative): h = ∂²L/∂ŷ² = 1

3. ALGORITHM:
   a) Initialize predictions with base score (typically mean of y)
   b) For each boosting round:
      - Calculate gradients and hessians
      - Build a tree to minimize the regularized objective
      - Update predictions: ŷ_new = ŷ_old + η * tree_prediction
   c) Final prediction is sum of all tree predictions

4. TREE BUILDING:
   - Find splits that maximize gain: Gain = ½[(G_L²/(H_L+λ)) + (G_R²/(H_R+λ)) - (G²/(H+λ))]
   - Leaf weights: w = -G/(H+λ) where G=sum(gradients), H=sum(hessians), λ=regularization

5. KEY DIFFERENCES FROM STANDARD DECISION TREES:
   - Uses gradients/hessians instead of raw target values
   - Regularization in leaf weight calculation
   - Gain calculation considers both gradient and hessian information

IMPLEMENTATION:
==============
"""

using Random, Statistics

# Simple tree node structure
mutable struct SimpleTreeNode
    feature_idx::Union{Int, Nothing}      # Which feature to split on
    threshold::Union{Float64, Nothing}    # Split threshold
    left::Union{SimpleTreeNode, Nothing}  # Left child
    right::Union{SimpleTreeNode, Nothing} # Right child
    value::Union{Float64, Nothing}        # Leaf value (prediction)
    is_leaf::Bool                         # Is this a leaf node?
    
    SimpleTreeNode() = new(nothing, nothing, nothing, nothing, nothing, false)
end

# XGBoost regressor
struct SimpleXGBoost
    n_estimators::Int
    max_depth::Int
    learning_rate::Float64
    reg_lambda::Float64
    trees::Vector{SimpleTreeNode}
    base_score::Float64
    
    function SimpleXGBoost(n_estimators=10, max_depth=3, learning_rate=0.3, reg_lambda=1.0)
        new(n_estimators, max_depth, learning_rate, reg_lambda, SimpleTreeNode[], 0.0)
    end
end

"""
Calculate gradients and hessians for squared error loss
"""
function calc_grad_hess(y_true, y_pred)
    gradients = y_pred - y_true  # ∂L/∂ŷ = ŷ - y
    hessians = ones(length(y_true))  # ∂²L/∂ŷ² = 1 (constant for squared error)
    return gradients, hessians
end

"""
Calculate optimal leaf weight: w = -G/(H + λ)
"""
function calc_leaf_weight(gradients, hessians, reg_lambda)
    G = sum(gradients)
    H = sum(hessians)
    return -G / (H + reg_lambda)
end

"""
Calculate gain for a split
"""
function calc_gain(grad_left, hess_left, grad_right, hess_right, grad_parent, hess_parent, reg_lambda)
    G_L, H_L = sum(grad_left), sum(hess_left)
    G_R, H_R = sum(grad_right), sum(hess_right)
    G_P, H_P = sum(grad_parent), sum(hess_parent)
    
    # Gain formula from XGBoost paper
    gain = 0.5 * ((G_L^2)/(H_L + reg_lambda) + (G_R^2)/(H_R + reg_lambda) - (G_P^2)/(H_P + reg_lambda))
    return gain
end

"""
Find best feature and threshold for splitting
"""
function find_best_split(X, gradients, hessians, reg_lambda, min_samples_leaf=1)
    best_gain = -Inf
    best_feature = nothing
    best_threshold = nothing
    
    n_samples, n_features = size(X)
    
    for feature_idx in 1:n_features
        # Get unique values and sort them
        values = sort(unique(X[:, feature_idx]))
        
        if length(values) < 2
            continue  # Can't split on this feature
        end
        
        # Try splits between consecutive unique values
        for i in 1:(length(values)-1)
            threshold = (values[i] + values[i+1]) / 2
            
            # Split data
            left_mask = X[:, feature_idx] .< threshold
            right_mask = .!left_mask
            
            # Check minimum samples constraint
            if sum(left_mask) < min_samples_leaf || sum(right_mask) < min_samples_leaf
                continue
            end
            
            # Calculate gain
            grad_left = gradients[left_mask]
            hess_left = hessians[left_mask]
            grad_right = gradients[right_mask]
            hess_right = hessians[right_mask]
            
            gain = calc_gain(grad_left, hess_left, grad_right, hess_right, 
                           gradients, hessians, reg_lambda)
            
            if gain > best_gain
                best_gain = gain
                best_feature = feature_idx
                best_threshold = threshold
            end
        end
    end
    
    return best_gain, best_feature, best_threshold
end

"""
Build a single tree recursively
"""
function build_tree(X, gradients, hessians, max_depth, current_depth, reg_lambda)
    node = SimpleTreeNode()
    
    # Stopping criteria
    if current_depth >= max_depth || length(gradients) < 2
        # Create leaf
        node.is_leaf = true
        node.value = calc_leaf_weight(gradients, hessians, reg_lambda)
        return node
    end
    
    # Find best split
    gain, feature_idx, threshold = find_best_split(X, gradients, hessians, reg_lambda)
    
    # If no good split found, create leaf
    if gain <= 0 || isnothing(feature_idx)
        node.is_leaf = true
        node.value = calc_leaf_weight(gradients, hessians, reg_lambda)
        return node
    end
    
    # Create internal node
    node.feature_idx = feature_idx
    node.threshold = threshold
    
    # Split data
    left_mask = X[:, feature_idx] .< threshold
    right_mask = .!left_mask
    
    # Recursively build children
    node.left = build_tree(X[left_mask, :], gradients[left_mask], hessians[left_mask], 
                          max_depth, current_depth + 1, reg_lambda)
    node.right = build_tree(X[right_mask, :], gradients[right_mask], hessians[right_mask], 
                           max_depth, current_depth + 1, reg_lambda)
    
    return node
end

"""
Predict with a single tree
"""
function predict_tree(node, x)
    if node.is_leaf
        return node.value
    end
    
    if x[node.feature_idx] < node.threshold
        return predict_tree(node.left, x)
    else
        return predict_tree(node.right, x)
    end
end

"""
Train XGBoost model
"""
function train_xgboost(X, y, n_estimators=10, max_depth=3, learning_rate=0.3, reg_lambda=1.0)
    # Initialize
    base_score = mean(y)
    y_pred = fill(base_score, length(y))
    trees = SimpleTreeNode[]
    
    println("Training XGBoost with $(n_estimators) trees...")
    println("Base score (mean of y): $(round(base_score, digits=4))")
    
    # Build trees sequentially
    for round in 1:n_estimators
        # Calculate gradients and hessians
        gradients, hessians = calc_grad_hess(y, y_pred)
        
        # Build tree
        tree = build_tree(X, gradients, hessians, max_depth, 0, reg_lambda)
        push!(trees, tree)
        
        # Update predictions
        for i in 1:size(X, 1)
            tree_pred = predict_tree(tree, X[i, :])
            y_pred[i] += learning_rate * tree_pred
        end
        
        # Calculate and print loss
        loss = mean((y - y_pred).^2) / 2
        println("Round $round: Loss = $(round(loss, digits=6))")
    end
    
    return trees, base_score
end

"""
Make predictions with trained model
"""
function predict_xgboost(trees, base_score, X, learning_rate=0.3)
    n_samples = size(X, 1)
    predictions = fill(base_score, n_samples)
    
    # Add contribution from each tree
    for tree in trees
        for i in 1:n_samples
            tree_pred = predict_tree(tree, X[i, :])
            predictions[i] += learning_rate * tree_pred
        end
    end
    
    return predictions
end

"""
Calculate metrics
"""
function calculate_metrics(y_true, y_pred)
    mse = mean((y_true - y_pred).^2)
    rmse = sqrt(mse)
    mae = mean(abs.(y_true - y_pred))
    
    # R² score
    ss_res = sum((y_true - y_pred).^2)
    ss_tot = sum((y_true .- mean(y_true)).^2)
    r2 = 1 - (ss_res / ss_tot)
    
    return mse, rmse, mae, r2
end

"""
DEMONSTRATION
"""
function demo_xgboost()
    println("="^60)
    println("XGBoost Squared Error Regression - Pure Julia Demo")
    println("="^60)
    
    # Create synthetic data
    Random.seed!(42)
    n_samples = 200
    n_features = 3
    
    # Generate features
    X = randn(n_samples, n_features)
    
    # True relationship: y = 3*x1 - 2*x2 + x3 + noise
    y = 3*X[:, 1] - 2*X[:, 2] + X[:, 3] + 0.3*randn(n_samples)
    
    println("Dataset created:")
    println("- Samples: $n_samples")
    println("- Features: $n_features")
    println("- True relationship: y = 3*x1 - 2*x2 + x3 + noise")
    println()
    
    # Split data
    train_size = Int(0.8 * n_samples)
    indices = randperm(n_samples)
    train_idx = indices[1:train_size]
    test_idx = indices[train_size+1:end]
    
    X_train, y_train = X[train_idx, :], y[train_idx]
    X_test, y_test = X[test_idx, :], y[test_idx]
    
    println("Data split:")
    println("- Training: $(length(y_train)) samples")
    println("- Testing: $(length(y_test)) samples")
    println()
    
    # Train model
    trees, base_score = train_xgboost(X_train, y_train, 
                                     n_estimators=50, 
                                     max_depth=4, 
                                     learning_rate=0.1, 
                                     reg_lambda=1.0)
    
    println("\\nModel trained with $(length(trees)) trees")
    println()
    
    # Make predictions
    y_pred_train = predict_xgboost(trees, base_score, X_train, 0.1)
    y_pred_test = predict_xgboost(trees, base_score, X_test, 0.1)
    
    # Calculate metrics
    train_mse, train_rmse, train_mae, train_r2 = calculate_metrics(y_train, y_pred_train)
    test_mse, test_rmse, test_mae, test_r2 = calculate_metrics(y_test, y_pred_test)
    
    println("Results:")
    println("--------")
    println("Training Metrics:")
    println("  MSE:  $(round(train_mse, digits=4))")
    println("  RMSE: $(round(train_rmse, digits=4))")
    println("  MAE:  $(round(train_mae, digits=4))")
    println("  R²:   $(round(train_r2, digits=4))")
    println()
    println("Test Metrics:")
    println("  MSE:  $(round(test_mse, digits=4))")
    println("  RMSE: $(round(test_rmse, digits=4))")
    println("  MAE:  $(round(test_mae, digits=4))")
    println("  R²:   $(round(test_r2, digits=4))")
    println()
    
    # Show some predictions vs actual
    println("Sample Predictions vs Actual (first 10 test samples):")
    println("Predicted  |  Actual   |  Error")
    println("-" * 35)
    for i in 1:min(10, length(y_test))
        pred = y_pred_test[i]
        actual = y_test[i]
        error = abs(pred - actual)
        println("$(lpad(round(pred, digits=3), 8))  |  $(lpad(round(actual, digits=3), 8))  |  $(round(error, digits=3))")
    end
    
    println("\\n" * "="^60)
    println("KEY CONCEPTS DEMONSTRATED:")
    println("="^60)
    println("1. Gradient Boosting: Each tree fits the residuals (gradients) of previous predictions")
    println("2. Squared Error Loss: Simple MSE with gradients = ŷ - y, hessians = 1")
    println("3. Regularization: λ parameter prevents overfitting in leaf weight calculation")
    println("4. Learning Rate: Controls step size (0.1 used here for stability)")
    println("5. Tree Depth: Limits complexity of individual trees (depth=4 here)")
    println("6. Sequential Learning: Each tree improves upon the ensemble's mistakes")
end

# Run the demonstration
demo_xgboost()
