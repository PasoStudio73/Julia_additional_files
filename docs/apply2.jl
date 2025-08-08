"""
apply.jl - Unified Model Conversion Interface

This module provides `apply(ds, X, y)` methods that:
1. Extract fitted models from MLJ machine wrappers
2. Convert them to SoleModels symbolic representations
3. Create logical datasets and apply symbolic models
4. Return symbolic models ready for rule extraction and analysis

Supported ML packages:
- DecisionTree.jl (trees, random forests, AdaBoost)
- ModalDecisionTrees.jl (modal trees, forests, boosting)
- XGBoost.jl (classifiers and regressors)

All methods handle both regular and hyperparameter-tuned models.
"""

# ...existing code...

"""
    get_base_score(m::MLJ.Machine) -> Union{Number,Nothing}

Extract base_score from XGBoost models, handling both regular and tuned variants.
Returns `nothing` if the model doesn't have a base_score property.
"""
get_base_score(m::MLJ.Machine)

"""
    get_encoding(classes_seen) -> Dict

Create mapping from internal class indices to MLJ class labels.
Used for preserving class information in classification models.
"""
get_encoding(classes_seen)

"""
    get_classlabels(encoding) -> Vector{String}

Extract ordered class labels from encoding dictionary.
Ensures consistent class ordering across model conversions.
"""
get_classlabels(encoding)


"""
    apply(ds::DecisionTreeApply, X, y)

Convert single decision trees (classifier/regressor) to symbolic models.
Extracts feature names and tree structure from fitted MLJ machine.
"""

"""
    apply(ds::TunedDecisionTreeApply, X, y) 

Handle hyperparameter-tuned decision trees using best model results.
"""

"""
    apply(ds::PropositionalDataSet{RandomForestClassifier}, X, y)

Convert random forest classifier to symbolic ensemble model.
Key feature: Preserves class label ordering via `classlabels` parameter.
"""

"""
    apply(ds::PropositionalDataSet{AdaBoostStumpClassifier}, X, y)

Convert AdaBoost classifier to weighted symbolic ensemble.
Extracts stump weights and maintains class label consistency.
"""

"""
    apply(ds::ModalDecisionTreeApply, X, y)

Handle modal decision trees using the model's built-in `sprinkle` utility.
The sprinkle function handles symbolic model creation internally.
"""

"""
    apply(ds::PropositionalDataSet{XGBoostClassifier}, X, y)

Convert XGBoost classifier with proper class label handling.
- Extracts trees from booster
- Creates class encoding dictionary
- Converts features to Float32 for compatibility
"""

"""
    apply(ds::PropositionalDataSet{XGBoostRegressor}, X, y)

Convert XGBoost regressor with base_score propagation.
Handles base_score logic to maintain prediction consistency.
"""