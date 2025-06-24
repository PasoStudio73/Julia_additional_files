println("XGBoost Squared Error Regression Explanation")
println("=" ^ 50)

println("""
CONCEPT OVERVIEW:
================

XGBoost (eXtreme Gradient Boosting) for regression with squared error works as follows:

1. LOSS FUNCTION:
   L(y, ŷ) = ½(y - ŷ)²
   
   This is the standard mean squared error loss function.

2. GRADIENTS AND HESSIANS:
   For gradient boosting, we need the first and second derivatives:
   
   Gradient:  g = ∂L/∂ŷ = ŷ - y  (residual)
   Hessian:   h = ∂²L/∂ŷ² = 1    (constant for squared error)

3. ALGORITHM STEPS:
   a) Initialize predictions with base score (typically mean of y)
   b) For each boosting round:
      - Calculate gradients g_i = ŷ_i - y_i for all samples
      - Calculate hessians h_i = 1 for all samples  
      - Build a regression tree to predict -g_i (negative gradients)
      - Find optimal leaf weights using: w = -G/(H + λ)
        where G = sum of gradients in leaf, H = sum of hessians, λ = regularization
      - Update predictions: ŷ_new = ŷ_old + η × tree_prediction
   c) Final prediction is the sum of all tree contributions

4. TREE BUILDING:
   - Split criterion maximizes gain:
     Gain = ½[(G_L²/(H_L+λ)) + (G_R²/(H_R+λ)) - (G²/(H+λ))]
   - Each leaf gets weight: w = -G/(H+λ)

5. KEY PARAMETERS:
   - n_estimators: Number of trees (more trees = more complex model)
   - max_depth: Maximum depth of each tree (controls overfitting)
   - learning_rate (eta): Step size (smaller = more conservative learning)
   - reg_lambda: L2 regularization (larger = more regularization)

EXAMPLE WORKFLOW:
================

1. Start with initial prediction = mean(y)
2. Calculate residuals (gradients) = current_predictions - y
3. Build tree to predict these residuals
4. Add tree's predictions (scaled by learning_rate) to ensemble
5. Repeat until convergence or max iterations

The magic is that each new tree focuses on the mistakes of the current ensemble,
gradually improving the overall prediction quality.
""")

println("This implementation demonstrates the core XGBoost algorithm in pure Julia.")
println("Each component (gradient calculation, tree building, prediction) is")
println("implemented from scratch to show exactly how the algorithm works.")
