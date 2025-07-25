### SoleXplorer
"""
   SoleXplorer

[`SoleXplorer`](https://github.com/aclai-lab/SoleXplorer.jl) is a comprehensive toolbox for 
symbolic machine learning and explainable AI in Julia. It integrates symbolic learning 
algorithms, rule extraction methods, and time series feature extraction capabilities with 
MLJ's machine learning ecosystem.

SoleXplorer brings together functionality from the Sole ecosystem components, providing 
a unified interface for symbolic learning, modal logic, and explainable AI workflows.

# Core Features

- **Symbolic Learning**: Decision trees, ensembles, and rule-based models through the Sole ecosystem
- **Rule Extraction**: Post-hoc explainability via various rule extraction algorithms  
- **Time Series Analysis**: Feature extraction using Catch22 and custom featuresets
- **MLJ Integration**: Full compatibility with MLJ's machine learning interface
- **Modal Logic**: Support for modal decision trees and temporal reasoning

# Components

- **SoleBase.jl**: Core types and windowing functions for symbolic learning, including 
  `Label` types (`CLabel`, `RLabel`, `XGLabel`) and windowing strategies (`movingwindow`, 
  `wholewindow`, `splitwindow`, `adaptivewindow`)

- **SoleData.jl**: Data structures and utilities for symbolic datasets, including 
  `scalarlogiset` and ARFF dataset loading capabilities

- **SoleModels.jl**: Symbolic model implementations including `DecisionTree`, 
  `DecisionEnsemble`, `DecisionSet`, and rule extraction via `RuleExtractor`

- **SolePostHoc.jl**: Post-hoc explainability methods, featuring `InTreesRuleExtractor` 
  and other rule extraction algorithms for model interpretation

- **MLJ Ecosystem**: Full integration with MLJ for model evaluation, tuning, and 
  performance assessment including classification measures (`accuracy`, `confusion_matrix`, 
  `kappa`, `log_loss`) and regression measures (`rms`, `l1`, `l2`, `mae`, `mav`)

- **Time Series Features**: Comprehensive feature extraction via Catch22 library with 
  predefined feature sets (`base_set`, `catch9`, `catch22_set`, `complete_set`)

- **External Models**: Integration with popular ML libraries including DecisionTree.jl, 
  XGBoost.jl, and modal decision tree implementations

# Typical Workflow

```julia
using SoleXplorer

# Load and prepare data
dataset = load_arff_dataset("path/to/data.arff")
X, y = setup_dataset(dataset)

# Apply windowing for time series
windowed_data = MovingWindow(window_size=10)(X)

# Train a modal decision tree
model = ModalDecisionTree()
mach = machine(model, windowed_data, y)
fit!(mach)

# Extract rules for explainability
extractor = InTreesRuleExtractor()
rules = extractrules(mach, extractor)

# Evaluate performance
cv_results = evaluate!(mach, resampling=CV(nfolds=5), measure=accuracy)
```

"""
module SoleXplorer
using  Reexport

# ---------------------------------------------------------------------------- #
#                                Sole Ecosystem                                #
# ---------------------------------------------------------------------------- #

# Core symbolic learning types and windowing functions
using  SoleBase: Label, CLabel, RLabel, XGLabel
using  SoleBase: movingwindow, wholewindow, splitwindow, adaptivewindow

# Data structures for symbolic datasets  
using  SoleData: scalarlogiset

# Symbolic models and rule extraction
using  SoleModels: Branch, ConstantModel
using  SoleModels: DecisionEnsemble, DecisionTree
using  SoleModels: AbstractModel, solemodel, weighted_aggregation, apply!
using  SoleModels: RuleExtractor, DecisionSet

# Post-hoc explainability methods
using  SolePostHoc
@reexport using SolePostHoc: InTreesRuleExtractor
# Additional rule extractors (commented for future use):
# @reexport using SolePostHoc: 
#     LumenRuleExtractor, BATreesRuleExtractor, REFNERuleExtractor, RULECOSIPLUSRuleExtractor     

# ---------------------------------------------------------------------------- #
#                                     MLJ                                      #
# ---------------------------------------------------------------------------- #

# Core MLJ functionality
using  MLJ
using  MLJ: MLJBase, MLJTuning

# Performance measures for classification
@reexport using MLJ: accuracy, confusion_matrix, kappa, log_loss

# Performance measures for regression  
@reexport using MLJ: rms, l1, l2, mae, mav

# ---------------------------------------------------------------------------- #
#                              External Packages                               #
# ---------------------------------------------------------------------------- #

# Data manipulation and utilities
using  DataFrames
using  Random

# Dataset loading capabilities
@reexport using SoleData: load_arff_dataset

# ---------------------------------------------------------------------------- #
#                                   Types                                      #
# ---------------------------------------------------------------------------- #

"""
    Optional{T}

Type alias for `Union{T, Nothing}`, representing optional values that may be missing.
"""
const Optional{T} = Union{T, Nothing}

# ---------------------------------------------------------------------------- #
#                            Time Series Features                              #
# ---------------------------------------------------------------------------- #

# Time series feature extraction via Catch22
using  Catch22

# Include custom feature sets and utilities
include("featureset.jl")

# Export individual Catch22 features
export mode_5, mode_10, embedding_dist, acf_timescale, acf_first_min, ami2, trev, outlier_timing_pos
export outlier_timing_neg, whiten_timescale, forecast_error, ami_timescale, high_fluctuation, stretch_decreasing
export stretch_high, entropy_pairs, rs_range, dfa, low_freq_power, centroid_freq, transition_variance, periodicity

# Export feature set collections
export base_set, catch9, catch22_set, complete_set

# ---------------------------------------------------------------------------- #
#                               Data Interfaces                                #
# ---------------------------------------------------------------------------- #

# MLJ resampling strategies
@reexport using MLJ: Holdout, CV, StratifiedCV, TimeSeriesCV

# Custom data partitioning
include("partition.jl")
export partition

# Data transformation and windowing
include("treatment.jl")
export WinFunction, MovingWindow, WholeWindow, SplitWindow, AdaptiveWindow

# Hyperparameter optimization strategies
@reexport using MLJ: Grid, RandomSearch, LatinHypercube
@reexport using MLJParticleSwarmOptimization: ParticleSwarm, AdaptiveParticleSwarm

# ---------------------------------------------------------------------------- #
#                                ML Models                                     #
# ---------------------------------------------------------------------------- #

# Standard decision tree implementations
using MLJDecisionTreeInterface
@reexport using MLJDecisionTreeInterface: 
    DecisionTreeClassifier, DecisionTreeRegressor,
    RandomForestClassifier, RandomForestRegressor,
    AdaBoostStumpClassifier

# Modal logic decision trees
using ModalDecisionTrees
@reexport using ModalDecisionTrees: 
    ModalDecisionTree, ModalRandomForest, ModalAdaBoost

# Gradient boosting models
using XGBoost, MLJXGBoostInterface
@reexport using MLJXGBoostInterface: 
    XGBoostClassifier, XGBoostRegressor

# ---------------------------------------------------------------------------- #
#                                Core Modules                                  #
# ---------------------------------------------------------------------------- #

# Dataset handling and encoding
include("dataset.jl")
export code_dataset, range
export setup_dataset

# Model application utilities
include("apply.jl")

# Training and testing workflows  
include("train_test.jl")
export train_test

# Rule extraction functionality
include("extractrules.jl")

# Symbolic analysis tools
include("symbolic_analysis.jl")
export symbolic_analysis

end