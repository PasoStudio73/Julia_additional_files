using Sole
using SoleXplorer
using Random, StatsBase, JLD2, DataFrames, CSV
using Downloads, CategoricalArrays
using MLJBase, MLJTuning
using ShapML

# the directory containing this file: (.../src/)
const MODULE_DIR = dirname(@__FILE__)

const COERCE_ADULT = (
    :age => Continuous,
    :workclass => Multiclass,
    :fnlwgt => Continuous,
    :education => Multiclass,
    :education_num => Continuous,
    :marital_status => Multiclass,
    :occupation => Multiclass,
    :relationship => Multiclass,
    :race => Multiclass,
    :sex => Multiclass,
    :capital_gain => Continuous,
    :capital_loss => Continuous,
    :hours_per_week => Continuous,
    :native_country => Multiclass,
    :income_per_year => Multiclass,
)

"""
Checks whether the dataset is already present in data directory. Downloads it if not present.
"""
function ensure_download(url::String, file::String)
    fpath = joinpath(MODULE_DIR, file)
    if !isfile(fpath)
        Downloads.download(url, fpath)
    end
end

"""
Macro to Load the [Adult dataset](https://archive.ics.uci.edu/ml/datasets/adult)
It has 14 features and 32561 rows. The protected attributes are race and sex.
This dataset is used to predict whether income exceeds 50K dollars per year.

Returns (X, y)
"""
macro load_adult()
    quote
        # url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
        # fname = "adult.data"
        # cols = ["age", "workclass", "fnlwgt", "education",
        #     "education_num", "marital_status", "occupation",
        #     "relationship", "race", "sex", "capital_gain",
        #     "capital_loss", "hours_per_week", "native_country",
        #     "income_per_year"
        # ]
        # ensure_download(url, fname)
        fpath = joinpath(MODULE_DIR, fname)
        data = DataFrame(CSV.File(fpath, header=cols, silencewarnings=true, delim=", "); copycols = false)
        # Warning is silenced to supress warnings for lesser number of columns

        data = dropmissing(data, names(data))
        data.income_per_year = map(data.income_per_year) do η
            η == "<=50K" ? 0 : 1
        end

        coerce!(data, COERCE_ADULT...)
        coerce!(data, :income_per_year => Multiclass)
        y, X = unpack(data, ==(:income_per_year), col -> true)
        (X, y)
    end
end

# Load train and test data
X, y = @load_adult
X, _ = code_dataframe = SoleXplorer.code_dataframe(X, y)

train_seed = 7
rng        = Random.Xoshiro(train_seed)
Random.seed!(train_seed)
ds = prepare_dataset(X, y, train_ratio=0.8, valid_ratio=0.8, algo=:classification, shuffle=true)
modelset = validate_modelset(models, typeof(y))
# Train the model
params = (
    num_round=5000,
    eta=0.01,
    objective="binary:logistic",
    subsample=0.5,
    base_score=mean(unwrap.(ds.ytrain)),
    eval_metric="logloss",
    early_stopping_rounds=20,
    watchlist=makewatchlist
)

result = traintest(X, y; 
    models=(
        type=:xgboost_classifier,
        algo=":classification",
        params=(
            num_round=5000,
            eta=0.01,
            objective="binary:logistic",
            subsample=0.5,
            base_score=mean(unwrap.(y)),
            eval_metric=["logloss"],
            early_stopping_rounds=20,
            watchlist=makewatchlist
        ),
    ),
    # with early stopping a validation set is required
    preprocess=(
        train_ratio=0.8,
        valid_ratio=0.8,
        shuffle=true
    )
)


