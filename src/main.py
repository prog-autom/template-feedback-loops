import argparse
import json
import os
from tqdm import tqdm

from experiment import *
from results import MultipleResults

from joblib.parallel import Parallel, delayed

parser = None
folder = None
run_times = None
random_seed = None


def create_model(model_params):
    if 'ridge_model' in model_params:
        model_name = 'ridge_model'
        model = lambda : ridge_model(**model_params['ridge_model'])
        print(f"Using model: {model_name}")
        yield model, model_name


def create_dataset(dataset_params):
    if 'make_regression' in dataset_params:
        data_name = "make_regression"
        print(f"Using dataset: {data_name}")
        dataset = lambda : get_synthetic_dataset(**dataset_params['make_regression'])
        yield dataset, data_name


def single_model(dataset_params, model_params, params):
    print(f"Running single-model experiment")
    for dataset, data_name in create_dataset(dataset_params):
        X, y, _ = dataset()
        for model, model_name in create_model(model_params):
            for trial in tqdm(range(0, run_times)):
                save_to = f"{folder}/{data_name}/{trial}"
                os.makedirs(f"{save_to}", exist_ok=True)

                prepare_params = {k: params[k] for k in params.keys() & {'train_size'}}
                prepare_params.update({k: dataset_params[data_name][k]
                                       for k in dataset_params[data_name].keys() & {'use_log'}})
                single_model_experiment(X, y, model,
                                        model_name=f"{save_to}/{model_name}", **prepare_params)


def hidden_loop(dataset_params, model_params, params):
    print(f"Running hidden-loop experiment")
    for dataset, data_name in create_dataset(dataset_params):
        X, y, _ = dataset()
        for model, model_name in create_model(model_params):
            hle_results = MultipleResults(model_name, **HiddenLoopExperiment.default_state)

            def process_trial(trial):
                hle_local = MultipleResults(model_name, **HiddenLoopExperiment.default_state)
                hle = HiddenLoopExperiment(X, y, model, model_name)

                prepare_params = {k: params[k] for k in params.keys() & {'train_size', 'A'}}
                prepare_params.update({k: dataset_params[data_name][k]
                                       for k in dataset_params[data_name].keys() & {'use_log'}})
                hle.prepare_data(**prepare_params)

                loop_params = {k: params[k] for k in params.keys() & {'adherence', 'usage', 'step'}}
                hle.run_experiment(**loop_params)

                hle_local.add_state(trial=trial, **vars(hle))

                return hle_local

            results = Parallel(n_jobs=-1, verbose=10)(delayed(process_trial)(trial) for trial in range(0, run_times))
            for hle_result in results:
                hle_results.add_results(**hle_result.get_state)

            target_folder = f"{folder}/{data_name}"
            os.makedirs(target_folder, exist_ok=True)
            hle_results.plot_multiple_results(target_folder, **HiddenLoopExperiment.default_figures)
            hle_results.save_state(f"{target_folder}")


def init_random(random_seed):
    return init_random_state(random_seed)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("kind", type=str, help="Kind of experiment: single-model, hidden-loop")
    parser.add_argument("--params", type=str, help="A json string with experiment parameters")
    parser.add_argument("--model_params", type=str, help="A json string with model name and parameters")
    parser.add_argument("--folder", type=str, help="Save results to this folder", default="./results")
    parser.add_argument("--random_seed", type=int, help="Use the provided value to init the random state", default=42)
    parser.add_argument("--run_times", type=int, help="How many time to repeat the trial", default=1)
    parser.add_argument("--dataset", type=str, help="Name of the dataset ('boston') or json for make_regression", default='\"boston\"')
    args = parser.parse_args()
    model_str = args.model_params
    params_str = args.params
    dataset_str = args.dataset
    kind = args.kind

    folder = args.folder
    random_seed = args.random_seed
    run_times = args.run_times
    os.makedirs(folder, exist_ok=True)

    model_dict = json.loads(model_str)
    params_dict = json.loads(params_str)
    dataset_params = json.loads(dataset_str)

    init_random_state(random_seed)

    if kind == "single-model":
        single_model(dataset_params, model_dict, params_dict)
    elif kind == "hidden-loop":
        hidden_loop(dataset_params, model_dict, params_dict)
    else:
        parser.error("Unknown experiment kind: " + kind)