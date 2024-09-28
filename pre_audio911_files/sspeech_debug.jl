using Pkg
using CSV
using DataFrames
using JLD2
using DataStructures

include("scanner.jl")
include("datasets/dataset-analysis.jl")

train_seed = 1

file_debug = "/home/riccardopasini/Documents/sspeech_debug/debug.jld2"
label = "gender"

results_dir = "/home/riccardopasini/Documents/sspeech_debug/"

iteration_progress_json_file_path = results_dir * "/progress.json"
data_savedir = results_dir * "data_cache"
model_savedir = results_dir * "models_cache"

dry_run = false
skip_training = false
save_datasets = false
perform_consistency_check = false
iteration_blacklist = []
use_training_form = :supportedlogiset_with_memoization

tree_args = []

for loss_function in [nothing] # ModalDecisionTrees.variance
    for min_samples_leaf in [2, 4] # [1,2]
        for min_purity_increase in [0.01, 0.05, 0.1]
            for max_purity_at_leaf in [Inf, 0.5, 0.6]
                # for max_purity_at_leaf in [10]
                push!(tree_args,
                    (
                        loss_function=loss_function,
                        min_samples_leaf=min_samples_leaf,
                        min_purity_increase=min_purity_increase,
                        max_purity_at_leaf=max_purity_at_leaf,
                        perform_consistency_check=perform_consistency_check,
                    )
                )
            end
        end
    end
end

n_nsdt_folds = 4

nsdt_args = []

for loss_function in []
    for min_samples_leaf in [4, 16]
        for min_purity_increase in [0.1, 0.05, 0.02, 0.01] # , 0.0075, 0.005, 0.002]
            for max_purity_at_leaf in [Inf, 0.001] # , 0.01, 0.2, 0.4, 0.6]
                push!(nsdt_args,
                    (;
                        loss_function=loss_function,
                        min_samples_leaf=min_samples_leaf,
                        min_purity_increase=min_purity_increase,
                        max_purity_at_leaf=max_purity_at_leaf,
                        perform_consistency_check=perform_consistency_check
                    )
                )
            end
        end
    end
end

nsdt_training_args = []
for model_type in [:lstm] # [:lstm, :tran]
    for epochs in [100] #
        for batch_size in [32] # 64 # 16
            for hidden_size in [64] #
                for code_size in [1] #
                    for save in [false] #
                        for use_subseries in [false]
                            for learning_rate in [1e-5] # 5e-5 # 1e-4
                                for patience in [nothing] # , nothing]
                                    push!(nsdt_training_args,
                                        (;
                                            model_type=model_type,
                                            epochs=epochs,
                                            batch_size=batch_size,
                                            hidden_size=hidden_size,
                                            code_size=code_size,
                                            save=save,
                                            use_subseries=use_subseries,
                                            learning_rate=learning_rate,
                                            patience=patience
                                        )
                                    )
                                end
                            end
                        end
                    end
                end
            end
        end
    end
end

nsdt_finetuning_args = []
for epochs in [10]
    for batch_size in [4] # 2
        for save in [false]
            for use_subseries in [false]
                for learning_rate in [5e-6] # 5e-5
                    for patience in [5] # , nothing]
                        push!(nsdt_finetuning_args,
                            (;
                                epochs=epochs,
                                batch_size=batch_size,
                                save=save,
                                use_subseries=use_subseries,
                                learning_rate=learning_rate,
                                patience=patience
                            )
                        )
                    end
                end
            end
        end
    end
end

n_forest_runs = 3
optimize_forest_computation = true

forest_args = []

for ntrees in [100]
    for n_subfeatures in [half_f]
        for n_subrelations in [id_f]
            for partial_sampling in [0.7]
                push!(forest_args, (
                    n_subfeatures=n_subfeatures,
                    ntrees=ntrees,
                    partial_sampling=partial_sampling,
                    n_subrelations=n_subrelations,
                    loss_function=nothing, # ModalDecisionTrees.entropy
                    perform_consistency_check=perform_consistency_check,
                ))
            end
        end
    end
end

modal_args = (;
    initconditions=ModalDecisionTrees.start_without_world,
    allow_global_splits=false
)

data_modal_args = (;
relations = [globalrel, SoleLogics.IARelations...]
)

log_level = SoleBase.LogDetail

timing_mode = :time

round_dataset_to_datatype = false

traintest_threshold = 1.0
split_threshold = 0.8

exec_dataseed = [(1:5)...]

exec_dataset_name = [
    (file_debug, label)
]

exec_moving_average_args = [
    (
        nwindows=5,
        relative_overlap=0.2,
    ),
]

exec_mixed_conditions = [
    "canonical",
]

exec_ranges = (;
    dataset_name=exec_dataset_name,
    moving_average_args_params=exec_moving_average_args,
    mixed_conditions=exec_mixed_conditions
)

dataset_function = (dataset_name, moving_average_args_params) -> begin

    (dataset_name, y) = dataset_name

    @show dataset_name

    d = jldopen(file_debug)
    df, Y, vars_n_meas = d["dataframe_validated"]

    @assert df isa DataFrame

    X, variable_names = SoleData.dataframe2cube(df)

    X = moving_average_filter(X; moving_average_args_params...)

    Y = string.(Y)
    Y=Y[:,1]
    Y = Vector{String}(Y)

    dataset = (X, Y)

    dataset, variable_names
end

iteration_whitelist = []

models_to_study = Dict([])

models_to_study = Dict(JSON.json(k) => v for (k, v) in models_to_study)

mkpath(results_dir)

if "-f" in ARGS
    if isfile(iteration_progress_json_file_path)
        println("Backing up existing $(iteration_progress_json_file_path)...")
        backup_file_using_creation_date(iteration_progress_json_file_path)
    end
end

# Copy scan script into the results folder
if PROGRAM_FILE != ""
    backup_file_using_creation_date(PROGRAM_FILE; copy_or_move=:copy, out_path=results_dir)
end

exec_ranges_names, exec_ranges_iterators = collect(string.(keys(exec_ranges))), collect(values(exec_ranges))
history = load_or_create_history(
    iteration_progress_json_file_path, exec_ranges_names, exec_ranges_iterators
)

n_interations = 0
n_interations_done = 0

####################### IL CICLO FOR #######################################################
# for params_combination in IterTools.product(exec_ranges_iterators...)
params_combination = (("/home/riccardopasini/Documents/sspeech_debug/debug.jld2", "gender"), (nwindows=5, relative_overlap=0.2), "canonical")

params_namedtuple = (; zip(Symbol.(exec_ranges_names), params_combination)...)

global n_interations += 1

run_name = join([replace(string(values(value)), ", " => ",") for value in params_combination], ",")

global n_interations_done += 1

(
    dataset_name,
    moving_average_args,
    mixed_conditions,
) = params_combination

dataset_fun_sub_params = (
    dataset_name,
    moving_average_args,
)

cur_modal_args = deepcopy(modal_args)
cur_data_modal_args = deepcopy(data_modal_args)

# Load Dataset
dataset, variable_names = @cachefast "dataset" data_savedir dataset_fun_sub_params dataset_function

cur_data_modal_args = merge(cur_data_modal_args, (;
    mixed_conditions=begin
        if isnothing(mixed_conditions)
            vcat([
                begin
                    [
                        (≥, SoleModels.UnivariateFeature{Float64}(var, get_patched_feature(meas, :pos))),
                        (≤, SoleModels.UnivariateFeature{Float64}(var, get_patched_feature(meas, :neg)))
                    ]
                end for (var, meas) in vars_n_meas
            ]...)
        else
            mixed_conditions_dict[mixed_conditions]
        end
    end
))

##############################################################################

X, Y = dataset

dataset = ([X], Y)

_, Y = dataset
class_count = get_class_counts(Y, true)

todo_dataseeds = filter((dataseed) -> !iteration_in_history(history, (params_namedtuple, dataseed)), exec_dataseed)

dataset_slices = begin
    n_insts = length(Y)

    Vector{Tuple{Integer,Union{DatasetSlice,Tuple{DatasetSlice,DatasetSlice}}}}([(dataseed, begin
        if dataseed == 0
            balanced_dataset_slice(
                dataset,
                [dataseed];
                ninstances_per_class=floor(Int, minimum(class_count) * 1.0),
                also_return_discarted=false
            )[1]
        else
            balanced_dataset_slice(
                dataset,
                [dataseed];
                ninstances_per_class=floor(Int, minimum(class_count) * split_threshold),
                also_return_discarted=true
            )[1]
        end
    end) for dataseed in todo_dataseeds])
end

if dry_run == :dataset_only
    
end

##############################################################################

if dry_run == false
    exec_scan(
        params_namedtuple,
        dataset;
        ### Training params
        train_seed=train_seed,
        modal_args=cur_modal_args,
        tree_args=tree_args,
        forest_args=forest_args,
        n_forest_runs=n_forest_runs,
        optimize_forest_computation=optimize_forest_computation,
        nsdt_args=nsdt_args,
        n_nsdt_folds=n_nsdt_folds,
        ### Dataset params
        data_modal_args=cur_data_modal_args,
        dataset_slices=dataset_slices,
        round_dataset_to_datatype=round_dataset_to_datatype,
        use_training_form=use_training_form,
        ### Run params
        results_dir=results_dir,
        data_savedir=data_savedir,
        model_savedir=model_savedir,
        timing_mode=timing_mode,
        ### Misc
        save_datasets=save_datasets,
        skip_training=skip_training,
        callback=(dataseed) -> begin
            # Add this step to the "history" of already computed iteration
            push_iteration_to_history!(history, (params_namedtuple, dataseed))
            save_history(iteration_progress_json_file_path, history)
        end
    )
end
# end

println("Done!")
println("# Iterations $(n_interations_done)/$(n_interations)")
println("Complete..Exit.")

# Notify the Telegram Bot
@error "Done!"

close(logfile_io);

exit(0)
