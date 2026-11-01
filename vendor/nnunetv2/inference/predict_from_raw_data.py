import inspect
import itertools
import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
from queue import Queue
from threading import Thread
from time import sleep
from typing import Tuple, Union, List, Optional
import numpy as np
import torch
from acvl_utils.cropping_and_padding.padding import pad_nd_image
from batchgenerators.dataloading.multi_threaded_augmenter import MultiThreadedAugmenter
from batchgenerators.utilities.file_and_folder_operations import load_json, join, isfile, maybe_mkdir_p, isdir, subdirs, \
    save_json
from torch import nn
from torch._dynamo import OptimizedModule
from torch.nn.parallel import DistributedDataParallel
from tqdm import tqdm
import nnunetv2
from nnunetv2.configuration import default_num_processes
from nnunetv2.inference.data_iterators import PreprocessAdapterFromNpy, preprocessing_iterator_fromfiles, \
    preprocessing_iterator_fromnpy
from nnunetv2.inference.export_prediction import export_prediction_from_logits, \
    convert_predicted_logits_to_segmentation_with_correct_shape
from nnunetv2.inference.sliding_window_prediction import compute_gaussian, \
    compute_steps_for_sliding_window
from nnunetv2.utilities.file_path_utilities import get_output_folder, check_workers_alive_and_busy
from nnunetv2.utilities.find_class_by_name import recursive_find_python_class
from nnunetv2.utilities.helpers import empty_cache, dummy_context
from nnunetv2.utilities.json_export import recursive_fix_for_json_export
from nnunetv2.utilities.label_handling.label_handling import determine_num_input_channels
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager, ConfigurationManager
from nnunetv2.utilities.utils import create_lists_from_splitted_dataset_folder
from nnunetv2.inference.experiment_registry import (
    ACTIVE_EXPERIMENTS,
    DEFAULT_RESULTS_ROOT as REGISTRY_RESULTS_ROOT,
    EXPERIMENT_SUITES,
    make_tablea_self_domain_experiment,
)
from nnunetv2.evaluation.confidence_error_profiling import (
    generate_confidence_profile_for_folder,
)
from nnunetv2.evaluation.evaluate_predictions import compute_metrics_on_folder2
import random
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)


DEFAULT_RESULTS_ROOT = REGISTRY_RESULTS_ROOT


def _safe_set_torch_interop_threads(num_threads: int) -> None:
    try:
        torch.set_num_interop_threads(num_threads)
    except RuntimeError as e:
        if "cannot set number of interop threads" not in str(e):
            raise
        print(f"[Inference] torch interop threads already initialized; keep existing setting ({e})")


def _configure_inference_device(device_name: str) -> torch.device:
    assert device_name in ['cpu', 'cuda', 'mps'], (
        f"-device must be either cpu, mps or cuda. Other devices are not tested/supported. Got: {device_name}."
    )
    if device_name == 'cpu':
        torch.set_num_threads(multiprocessing.cpu_count())
        return torch.device('cpu')
    if device_name == 'cuda':
        torch.set_num_threads(1)
        _safe_set_torch_interop_threads(1)
        return torch.device('cuda')
    return torch.device('mps')


def _resolve_registered_experiment_paths(
    experiment: dict,
    results_root: Optional[str] = None,
) -> dict:
    from nnunetv2.paths import nnUNet_raw

    results_root = DEFAULT_RESULTS_ROOT if results_root is None else results_root
    model_folder = get_output_folder(
        experiment["dataset_id"],
        experiment["trainer_name"],
        experiment.get("plans_name", "nnUNetPlans"),
        experiment.get("configuration", "2d"),
    )
    output_folder = join(
        results_root,
        experiment["result_name"],
        experiment["dataset_id"],
        experiment.get("split_name", "self_domain"),
        f'fold_{experiment.get("fold", 4)}',
    )
    checkpoint_name = experiment.get("checkpoint_name", "checkpoint_final.pth")
    return {
        "model_folder": model_folder,
        "input_folder": join(nnUNet_raw, experiment["dataset_name"], experiment.get("input_subdir", "imagesTs")),
        "gt_folder": join(nnUNet_raw, experiment["dataset_name"], "labelsTs"),
        "output_folder": output_folder,
        "dataset_json_file": join(output_folder, 'dataset.json'),
        "plans_file": join(output_folder, 'plans.json'),
        "checkpoint_name": checkpoint_name,
        "checkpoint_file": join(model_folder, f'fold_{experiment.get("fold", 4)}', checkpoint_name),
    }


def _maybe_write_registered_postprocess(
    experiment: dict,
    results_root: Optional[str] = None,
    num_processes: int = 1,
) -> bool:
    resolved = _resolve_registered_experiment_paths(experiment, results_root)
    output_folder = resolved["output_folder"]
    gt_folder = resolved["gt_folder"]
    dataset_json_file = resolved["dataset_json_file"]
    plans_file = resolved["plans_file"]

    if not isdir(output_folder) or not isdir(gt_folder):
        print(f'[Experiment] postprocess skipped: missing folder for {experiment["name"]}')
        return False
    if not isfile(dataset_json_file) or not isfile(plans_file):
        print(f'[Experiment] postprocess skipped: missing dataset/plans json for {experiment["name"]}')
        return False

    try:
        compute_metrics_on_folder2(
            gt_folder,
            output_folder,
            dataset_json_file,
            plans_file,
            output_file=join(output_folder, 'summary.json'),
            num_processes=num_processes,
            chill=True,
        )
        print(f'[Experiment] summary={join(output_folder, "summary.json")}')
        generate_confidence_profile_for_folder(
            pred_folder=output_folder,
            gt_folder=gt_folder,
            dataset_json_file=dataset_json_file,
            plans_file=plans_file,
            output_file=join(output_folder, 'confidence_profile.json'),
        )
        print(f'[Experiment] confidence_profile={join(output_folder, "confidence_profile.json")}')
    except Exception as e:
        print(f'[Experiment] postprocess skipped: {e}')
        return False
    return True


def _maybe_write_registered_confidence_profile(
    experiment: dict,
    model_folder: str,
    output_folder: str,
) -> None:
    del model_folder, output_folder
    _maybe_write_registered_postprocess(experiment=experiment)


def _is_tablea_model_folder(model_folder: str) -> bool:
    if os.path.basename(model_folder).startswith('nnUNetTrainer_tableA'):
        return True

    fold_folders = subdirs(model_folder, prefix='fold_', join=False) if isdir(model_folder) else []
    for fold_name in fold_folders:
        manifest_file = join(model_folder, fold_name, 'experiment_manifest.json')
        if not isfile(manifest_file):
            continue
        manifest = load_json(manifest_file)
        if manifest.get('network_class') == 'nnunetv2.nets.UNet.UNetTableA':
            return True
        if str(manifest.get('trainer_class', '')).startswith('nnUNetTrainer_tableA'):
            return True
    return False


def _is_gmm_calibrated_model_folder(model_folder: str) -> bool:
    if not isdir(model_folder):
        return False
    fold_folders = subdirs(model_folder, prefix='fold_', join=False)
    for fold_name in fold_folders:
        calibration_file = join(model_folder, fold_name, 'gmm_calibration', 'classifier_calibration.pth')
        if isfile(calibration_file):
            return True
    return False


def _is_fixed_bg_softmax_model_folder(model_folder: str) -> bool:
    folder_name = os.path.basename(os.path.normpath(model_folder))
    if (
        "nnUNetTrainer_foreground_distance_fixed_bg_softmax__" in folder_name
        or "nnUNetTrainer_multi_dot_fixed_bg_softmax__" in folder_name
    ):
        return True
    return False


def _build_predictor_for_model_folder(
    model_folder: str,
    device: torch.device,
    tile_step_size: float = 0.5,
    use_gaussian: bool = True,
    use_mirroring: bool = True,
    verbose: bool = False,
    verbose_preprocessing: bool = False,
    allow_tqdm: bool = True,
    metric: str = "seg",
):
    predictor_kwargs = {
        "tile_step_size": tile_step_size,
        "use_gaussian": use_gaussian,
        "use_mirroring": use_mirroring,
        "perform_everything_on_device": device.type == 'cuda',
        "device": device,
        "verbose": verbose,
        "verbose_preprocessing": verbose_preprocessing,
        "allow_tqdm": allow_tqdm,
    }

    if _is_gmm_calibrated_model_folder(model_folder):
        return _build_predictor_by_family("gmm_calibrated", predictor_kwargs, {"metric": metric})

    if _is_fixed_bg_softmax_model_folder(model_folder):
        return _build_predictor_by_family("fixed_bg_softmax", predictor_kwargs, {"metric": metric})

    if _is_tablea_model_folder(model_folder):
        return _build_predictor_by_family("tablea", predictor_kwargs, {"metric": metric})

    return nnUNetPredictor(**predictor_kwargs), "default"


def _build_predictor_by_family(
    predictor_family: str,
    shared_predictor_kwargs: dict,
    family_specific_kwargs: Optional[dict] = None,
):
    family_specific_kwargs = {} if family_specific_kwargs is None else dict(family_specific_kwargs)

    if predictor_family == "tablea":
        from nnunetv2.inference.nnUNetPredictor_unet import nnUNetPredictor_tableA
        predictor = nnUNetPredictor_tableA(**family_specific_kwargs, **shared_predictor_kwargs)
        return predictor, "tableA"

    if predictor_family == "gmm_calibrated":
        from nnunetv2.inference.nnUNetPredictor_gmm_calibrated import nnUNetPredictor_gmm_calibrated
        predictor = nnUNetPredictor_gmm_calibrated(**family_specific_kwargs, **shared_predictor_kwargs)
        return predictor, "gmm_calibrated"

    if predictor_family == "fixed_bg_softmax":
        from nnunetv2.inference.nnUNetPredictor_fixed_bg_softmax import nnUNetPredictor_fixed_bg_softmax
        predictor = nnUNetPredictor_fixed_bg_softmax(**family_specific_kwargs, **shared_predictor_kwargs)
        return predictor, "fixed_bg_softmax"

    if predictor_family == "unet":
        from nnunetv2.inference.nnUNetPredictor_unet import nnUNetPredictor_unet
        predictor = nnUNetPredictor_unet(**family_specific_kwargs, **shared_predictor_kwargs)
        return predictor, "unet"

    if predictor_family == "imbalance":
        from nnunetv2.inference.nnUNetPredictor_imbalance import nnUNetPredictor_imbalance
        predictor = nnUNetPredictor_imbalance(**family_specific_kwargs, **shared_predictor_kwargs)
        return predictor, "imbalance"

    if predictor_family == "swinunetr":
        from nnunetv2.inference.nnUNetPredictor_swinunetr import nnUNetPredictor_swinunetr
        predictor = nnUNetPredictor_swinunetr(**shared_predictor_kwargs)
        return predictor, "swinunetr"

    if predictor_family == "swinunetr_imbalance":
        from nnunetv2.inference.nnUNetPredictor_swinunetr import nnUNetPredictor_swinunetr_imbalance
        predictor = nnUNetPredictor_swinunetr_imbalance(**family_specific_kwargs, **shared_predictor_kwargs)
        return predictor, "swinunetr_imbalance"

    if predictor_family == "transunet":
        from nnunetv2.inference.nnUNetPredictor_transunet import nnUNetPredictor_transunet
        predictor = nnUNetPredictor_transunet(**shared_predictor_kwargs)
        return predictor, "transunet"

    if predictor_family == "transunet_imbalance":
        from nnunetv2.inference.nnUNetPredictor_transunet import nnUNetPredictor_transunet_imbalance
        predictor = nnUNetPredictor_transunet_imbalance(**family_specific_kwargs, **shared_predictor_kwargs)
        return predictor, "transunet_imbalance"

    if predictor_family == "swinumamba":
        from nnunetv2.inference.nnUNetPredictor_swinumamba import nnUNetPredictor_swinumamba
        predictor = nnUNetPredictor_swinumamba(**shared_predictor_kwargs)
        return predictor, "swinumamba"

    if predictor_family == "swinumamba_imbalance":
        from nnunetv2.inference.nnUNetPredictor_swinumamba import nnUNetPredictor_swinumamba_imbalance
        predictor = nnUNetPredictor_swinumamba_imbalance(**family_specific_kwargs, **shared_predictor_kwargs)
        return predictor, "swinumamba_imbalance"

    if predictor_family == "default":
        predictor = nnUNetPredictor(**shared_predictor_kwargs)
        return predictor, "default"

    raise ValueError(f"Unsupported predictor_family: {predictor_family}")


def _collect_predictor_family_kwargs(args) -> dict:
    family_specific_kwargs = {}
    option = getattr(args, "predictor_option", None)
    mode = getattr(args, "predictor_mode", None)
    transition_mode = getattr(args, "predictor_transition_mode", None)
    num_domains = getattr(args, "predictor_num_domains", None)
    source_trainer_name = getattr(args, "predictor_source_trainer_name", None)
    source_model_training_output_dir = getattr(args, "predictor_source_model_training_output_dir", None)
    metric = getattr(args, "predictor_metric", None)

    if metric is not None:
        family_specific_kwargs["metric"] = metric
    if option is not None:
        family_specific_kwargs["option"] = option
    if mode is not None:
        family_specific_kwargs["mode"] = mode
    if transition_mode is not None:
        family_specific_kwargs["transition_mode"] = transition_mode
    if num_domains is not None:
        family_specific_kwargs["num_domains"] = num_domains
    if source_trainer_name is not None:
        family_specific_kwargs["source_trainer_name"] = source_trainer_name
    if source_model_training_output_dir is not None:
        family_specific_kwargs["source_model_training_output_dir"] = source_model_training_output_dir
    return family_specific_kwargs


def _run_experiment_spec(
    experiment: dict,
    device_name: str = "cuda",
    results_root: Optional[str] = None,
    continue_prediction: bool = True,
    save_probabilities: bool = False,
    disable_tta: bool = False,
    verbose: bool = False,
    disable_progress_bar: bool = False,
    num_processes_preprocessing: int = 1,
    num_processes_segmentation_export: int = 1,
    run_postprocess: bool = True,
):
    device = _configure_inference_device(device_name)
    resolved = _resolve_registered_experiment_paths(experiment, results_root)

    model_folder = resolved["model_folder"]
    input_folder = resolved["input_folder"]
    output_folder = resolved["output_folder"]
    checkpoint_name = resolved["checkpoint_name"]
    checkpoint_file = resolved["checkpoint_file"]
    predictor_family_name = experiment.get("predictor_family", None)
    is_gmm_calibrated = predictor_family_name == "gmm_calibrated" or _is_gmm_calibrated_model_folder(model_folder)

    print(f'[Experiment] name={experiment["name"]}')
    print(f'[Experiment] status={experiment.get("status", "unknown")}')
    print(f'[Experiment] note={experiment.get("note", "")}')
    print(f'[Experiment] model_folder={model_folder}')
    print(f'[Experiment] input_folder={input_folder}')
    print(f'[Experiment] output_folder={output_folder}')

    if not isdir(input_folder):
        print(f'[Experiment] skip: input folder not found: {input_folder}')
        return
    if not is_gmm_calibrated and not isfile(checkpoint_file):
        print(f'[Experiment] skip: checkpoint not found: {checkpoint_file}')
        return

    shared_predictor_kwargs = {
        "tile_step_size": experiment.get("tile_step_size", 0.5),
        "use_gaussian": experiment.get("use_gaussian", True),
        "use_mirroring": not disable_tta if experiment.get("disable_tta") is None else not experiment["disable_tta"],
        "perform_everything_on_device": device.type == 'cuda',
        "device": device,
        "verbose": verbose,
        "verbose_preprocessing": verbose,
        "allow_tqdm": not disable_progress_bar,
    }
    predictor_family_kwargs = experiment.get("predictor_kwargs", None)
    if predictor_family_name is not None:
        predictor, predictor_family = _build_predictor_by_family(
            predictor_family_name,
            shared_predictor_kwargs,
            predictor_family_kwargs,
        )
    else:
        predictor, predictor_family = _build_predictor_for_model_folder(
            model_folder=model_folder,
            device=device,
            tile_step_size=shared_predictor_kwargs["tile_step_size"],
            use_gaussian=shared_predictor_kwargs["use_gaussian"],
            use_mirroring=shared_predictor_kwargs["use_mirroring"],
            verbose=shared_predictor_kwargs["verbose"],
            verbose_preprocessing=shared_predictor_kwargs["verbose_preprocessing"],
            allow_tqdm=shared_predictor_kwargs["allow_tqdm"],
            metric=experiment.get("predictor_metric", "seg"),
        )
    print(f'[Experiment] predictor_family={predictor_family}')

    predictor.initialize_from_trained_model_folder(
        model_folder,
        (experiment.get("fold", 4),),
        checkpoint_name=checkpoint_name,
    )
    predictor.predict_from_files(
        input_folder,
        output_folder,
        save_probabilities=save_probabilities,
        overwrite=not continue_prediction,
        num_processes_preprocessing=num_processes_preprocessing,
        num_processes_segmentation_export=num_processes_segmentation_export,
        folder_with_segs_from_prev_stage=None,
        num_parts=1,
        part_id=0,
    )

    if run_postprocess:
        _maybe_write_registered_postprocess(
            experiment=experiment,
            results_root=results_root,
            num_processes=1,
        )

    del predictor
    empty_cache(device)


def run_registered_experiment(
    experiment_name: str,
    device_name: str = "cuda",
    results_root: Optional[str] = None,
    continue_prediction: bool = True,
    save_probabilities: bool = False,
    disable_tta: bool = False,
    verbose: bool = False,
    disable_progress_bar: bool = False,
    num_processes_preprocessing: int = 1,
    num_processes_segmentation_export: int = 1,
):
    if experiment_name not in ACTIVE_EXPERIMENTS:
        raise KeyError(
            f"Unknown experiment: {experiment_name}. "
            f"Available experiments: {sorted(ACTIVE_EXPERIMENTS.keys())}"
        )
    experiment = ACTIVE_EXPERIMENTS[experiment_name]
    return _run_experiment_spec(
        experiment=experiment,
        device_name=device_name,
        results_root=results_root,
        continue_prediction=continue_prediction,
        save_probabilities=save_probabilities,
        disable_tta=disable_tta,
        verbose=verbose,
        disable_progress_bar=disable_progress_bar,
        num_processes_preprocessing=num_processes_preprocessing,
        num_processes_segmentation_export=num_processes_segmentation_export,
    )


def run_registered_suite(
    suite_name: str,
    device_name: str = "cuda",
    results_root: Optional[str] = None,
    continue_prediction: bool = True,
    save_probabilities: bool = False,
    disable_tta: bool = False,
    verbose: bool = False,
    disable_progress_bar: bool = False,
    num_processes_preprocessing: int = 1,
    num_processes_segmentation_export: int = 1,
):
    if suite_name not in EXPERIMENT_SUITES:
        raise KeyError(
            f"Unknown suite: {suite_name}. "
            f"Available suites: {sorted(EXPERIMENT_SUITES.keys())}"
        )

    experiment_names = EXPERIMENT_SUITES[suite_name]
    for job_idx, experiment_name in enumerate(experiment_names, start=1):
        print(f"[Suite] ({job_idx}/{len(experiment_names)}) {experiment_name}")
        run_registered_experiment(
            experiment_name=experiment_name,
            device_name=device_name,
            results_root=results_root,
            continue_prediction=continue_prediction,
            save_probabilities=save_probabilities,
            disable_tta=disable_tta,
            verbose=verbose,
            disable_progress_bar=disable_progress_bar,
            num_processes_preprocessing=num_processes_preprocessing,
            num_processes_segmentation_export=num_processes_segmentation_export,
        )


def _run_registered_experiment_postprocess(
    experiment: dict,
    results_root: Optional[str] = None,
    num_processes: int = 1,
):
    return _maybe_write_registered_postprocess(
        experiment=experiment,
        results_root=results_root,
        num_processes=num_processes,
    )


def run_registered_suite_postprocess(
    suite_name: str,
    results_root: Optional[str] = None,
    num_processes: int = 1,
):
    if suite_name not in EXPERIMENT_SUITES:
        raise KeyError(
            f"Unknown suite: {suite_name}. "
            f"Available suites: {sorted(EXPERIMENT_SUITES.keys())}"
        )

    experiment_names = EXPERIMENT_SUITES[suite_name]
    for job_idx, experiment_name in enumerate(experiment_names, start=1):
        print(f"[Suite postprocess] ({job_idx}/{len(experiment_names)}) {experiment_name}")
        experiment = ACTIVE_EXPERIMENTS[experiment_name]
        _run_registered_experiment_postprocess(
            experiment=experiment,
            results_root=results_root,
            num_processes=num_processes,
        )


def _registered_suite_predict_only_worker(
    experiment: dict,
    device_name: str,
    results_root: Optional[str],
    continue_prediction: bool,
    save_probabilities: bool,
    disable_tta: bool,
    verbose: bool,
    disable_progress_bar: bool,
    num_processes_preprocessing: int,
    num_processes_segmentation_export: int,
):
    _run_experiment_spec(
        experiment=experiment,
        device_name=device_name,
        results_root=results_root,
        continue_prediction=continue_prediction,
        save_probabilities=save_probabilities,
        disable_tta=disable_tta,
        verbose=verbose,
        disable_progress_bar=disable_progress_bar,
        num_processes_preprocessing=num_processes_preprocessing,
        num_processes_segmentation_export=num_processes_segmentation_export,
        run_postprocess=False,
    )
    return experiment["name"]


def run_registered_suite_predict_only_parallel(
    suite_name: str,
    device_name: str = "cuda",
    results_root: Optional[str] = None,
    continue_prediction: bool = True,
    save_probabilities: bool = False,
    disable_tta: bool = False,
    verbose: bool = False,
    disable_progress_bar: bool = False,
    num_processes_preprocessing: int = 1,
    num_processes_segmentation_export: int = 1,
    max_workers: int = 2,
):
    if suite_name not in EXPERIMENT_SUITES:
        raise KeyError(
            f"Unknown suite: {suite_name}. "
            f"Available suites: {sorted(EXPERIMENT_SUITES.keys())}"
        )

    experiment_names = EXPERIMENT_SUITES[suite_name]
    experiments = [ACTIVE_EXPERIMENTS[name] for name in experiment_names]
    results = {}

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_name = {
            executor.submit(
                _registered_suite_predict_only_worker,
                experiment,
                device_name,
                results_root,
                continue_prediction,
                save_probabilities,
                disable_tta,
                verbose,
                disable_progress_bar,
                num_processes_preprocessing,
                num_processes_segmentation_export,
            ): experiment["name"]
            for experiment in experiments
        }
        for future in as_completed(future_to_name):
            experiment_name = future_to_name[future]
            try:
                results[experiment_name] = {"status": "ok", "name": future.result()}
                print(f"[Suite predict-only] finished: {experiment_name}")
            except Exception as exc:
                results[experiment_name] = {"status": "error", "error": repr(exc)}
                print(f"[Suite predict-only] failed: {experiment_name}: {exc}")
    return results


def run_tablea_self_domain_suite(
    fold: int = 4,
    methods: Tuple[str, ...] = ("metric", "multi_proto", "slice_adaptive"),
    datasets: Tuple[str, ...] = ("002", "007", "009"),
    device_name: str = "cuda",
    plans_name: str = "nnUNetPlans",
    configuration: str = "2d",
    checkpoint_name: str = "checkpoint_final.pth",
    results_root: str = DEFAULT_RESULTS_ROOT,
    continue_prediction: bool = True,
    save_probabilities: bool = False,
    disable_tta: bool = False,
    verbose: bool = False,
    disable_progress_bar: bool = False,
    num_processes_preprocessing: int = 1,
    num_processes_segmentation_export: int = 1,
):
    jobs = [(method_name, dataset_id) for method_name in methods for dataset_id in datasets]

    for job_idx, (method_name, dataset_id) in enumerate(jobs, start=1):
        experiment = make_tablea_self_domain_experiment(method_name, dataset_id, fold=fold)
        experiment["plans_name"] = plans_name
        experiment["configuration"] = configuration
        experiment["checkpoint_name"] = checkpoint_name

        print(f"[TableA Suite] ({job_idx}/{len(jobs)}) method={method_name} dataset={dataset_id}")
        _run_experiment_spec(
            experiment=experiment,
            device_name=device_name,
            results_root=results_root,
            continue_prediction=continue_prediction,
            save_probabilities=save_probabilities,
            disable_tta=disable_tta,
            verbose=verbose,
            disable_progress_bar=disable_progress_bar,
            num_processes_preprocessing=num_processes_preprocessing,
            num_processes_segmentation_export=num_processes_segmentation_export,
        )

class nnUNetPredictor(object):
    def __init__(self,
                 tile_step_size: float = 0.5,
                 use_gaussian: bool = True,
                 use_mirroring: bool = True,
                 perform_everything_on_device: bool = True,
                 device: torch.device = torch.device('cuda'),
                 verbose: bool = False,
                 verbose_preprocessing: bool = False,
                 allow_tqdm: bool = True):
        self.verbose = verbose
        self.verbose_preprocessing = verbose_preprocessing
        self.allow_tqdm = allow_tqdm

        self.plans_manager, self.configuration_manager, self.list_of_parameters, self.network, self.dataset_json, \
        self.trainer_name, self.allowed_mirroring_axes, self.label_manager = None, None, None, None, None, None, None, None

        self.tile_step_size = tile_step_size
        self.use_gaussian = use_gaussian
        self.use_mirroring = use_mirroring
        if device.type == 'cuda':
            torch.backends.cudnn.benchmark = True
        else:
            print(f'perform_everything_on_device=True is only supported for cuda devices! Setting this to False')
            perform_everything_on_device = False
        self.device = device
        self.perform_everything_on_device = perform_everything_on_device

    def initialize_from_trained_model_folder(self, model_training_output_dir: str,
                                             use_folds: Union[Tuple[Union[int, str]], None],
                                             checkpoint_name: str = 'checkpoint_final.pth'):
        """
        This is used when making predictions with a trained model
        """
        if use_folds is None:
            use_folds = nnUNetPredictor.auto_detect_available_folds(model_training_output_dir, checkpoint_name)

        dataset_json = load_json(join(model_training_output_dir, 'dataset.json'))
        plans = load_json(join(model_training_output_dir, 'plans.json'))
        plans_manager = PlansManager(plans)

        if isinstance(use_folds, str):
            use_folds = [use_folds]

        parameters = []
        for i, f in enumerate(use_folds):
            f = int(f) if f != 'all' else f
            checkpoint = torch.load(join(model_training_output_dir, f'fold_{f}', checkpoint_name),
                                    map_location=torch.device('cpu'), weights_only=False)
            if i == 0:
                trainer_name = checkpoint['trainer_name']
                configuration_name = checkpoint['init_args']['configuration']
                inference_allowed_mirroring_axes = checkpoint['inference_allowed_mirroring_axes'] if \
                    'inference_allowed_mirroring_axes' in checkpoint.keys() else None

            parameters.append(checkpoint['network_weights'])

        configuration_manager = plans_manager.get_configuration(configuration_name)
        # restore network
        num_input_channels = determine_num_input_channels(plans_manager, configuration_manager, dataset_json)
        trainer_class = recursive_find_python_class(join(nnunetv2.__path__[0], "training", "nnUNetTrainer"),
                                                    trainer_name, 'nnunetv2.training.nnUNetTrainer')
        if trainer_class is None:
            raise RuntimeError(f'Unable to locate trainer class {trainer_name} in nnunetv2.training.nnUNetTrainer. '
                               f'Please place it there (in any .py file)!')
        network = trainer_class.build_network_architecture(
            configuration_manager.network_arch_class_name,
            configuration_manager.network_arch_init_kwargs,
            configuration_manager.network_arch_init_kwargs_req_import,
            num_input_channels,
            plans_manager.get_label_manager(dataset_json).num_segmentation_heads,
            enable_deep_supervision=False
        )

        self.plans_manager = plans_manager
        self.configuration_manager = configuration_manager
        self.list_of_parameters = parameters

        # initialize network with first set of parameters, also see https://github.com/MIC-DKFZ/nnUNet/issues/2520
        network.load_state_dict(parameters[0])

        self.network = network

        # initialize network with first set of parameters, also see https://github.com/MIC-DKFZ/nnUNet/issues/2520
        network.load_state_dict(parameters[0])

        self.dataset_json = dataset_json
        self.trainer_name = trainer_name
        self.allowed_mirroring_axes = inference_allowed_mirroring_axes
        self.label_manager = plans_manager.get_label_manager(dataset_json)
        if ('nnUNet_compile' in os.environ.keys()) and (os.environ['nnUNet_compile'].lower() in ('true', '1', 't')) \
                and not isinstance(self.network, OptimizedModule):
            print('Using torch.compile')
            self.network = torch.compile(self.network)

    def manual_initialization(self, network: nn.Module, plans_manager: PlansManager,
                              configuration_manager: ConfigurationManager, parameters: Optional[List[dict]],
                              dataset_json: dict, trainer_name: str,
                              inference_allowed_mirroring_axes: Optional[Tuple[int, ...]]):
        """
        This is used by the nnUNetTrainer to initialize nnUNetPredictor for the final validation
        """
        self.plans_manager = plans_manager
        self.configuration_manager = configuration_manager
        self.list_of_parameters = parameters
        self.network = network
        self.dataset_json = dataset_json
        self.trainer_name = trainer_name
        self.allowed_mirroring_axes = inference_allowed_mirroring_axes
        self.label_manager = plans_manager.get_label_manager(dataset_json)
        allow_compile = True
        allow_compile = allow_compile and ('nnUNet_compile' in os.environ.keys()) and (
                    os.environ['nnUNet_compile'].lower() in ('true', '1', 't'))
        allow_compile = allow_compile and not isinstance(self.network, OptimizedModule)
        if isinstance(self.network, DistributedDataParallel):
            allow_compile = allow_compile and isinstance(self.network.module, OptimizedModule)
        if allow_compile:
            print('Using torch.compile')
            self.network = torch.compile(self.network)

    @staticmethod
    def auto_detect_available_folds(model_training_output_dir, checkpoint_name):
        print('use_folds is None, attempting to auto detect available folds')
        fold_folders = subdirs(model_training_output_dir, prefix='fold_', join=False)
        fold_folders = [i for i in fold_folders if i != 'fold_all']
        fold_folders = [i for i in fold_folders if isfile(join(model_training_output_dir, i, checkpoint_name))]
        use_folds = [int(i.split('_')[-1]) for i in fold_folders]
        print(f'found the following folds: {use_folds}')
        return use_folds

    def _manage_input_and_output_lists(self, list_of_lists_or_source_folder: Union[str, List[List[str]]],
                                       output_folder_or_list_of_truncated_output_files: Union[None, str, List[str]],
                                       folder_with_segs_from_prev_stage: str = None,
                                       overwrite: bool = True,
                                       part_id: int = 0,
                                       num_parts: int = 1,
                                       save_probabilities: bool = False):
        if isinstance(list_of_lists_or_source_folder, str):
            list_of_lists_or_source_folder = create_lists_from_splitted_dataset_folder(list_of_lists_or_source_folder,
                                                                                       self.dataset_json['file_ending'])
        print(f'There are {len(list_of_lists_or_source_folder)} cases in the source folder')
        list_of_lists_or_source_folder = list_of_lists_or_source_folder[part_id::num_parts]
        caseids = [os.path.basename(i[0])[:-(len(self.dataset_json['file_ending']) + 5)] for i in
                   list_of_lists_or_source_folder]
        print(
            f'I am process {part_id} out of {num_parts} (max process ID is {num_parts - 1}, we start counting with 0!)')
        print(f'There are {len(caseids)} cases that I would like to predict')

        if isinstance(output_folder_or_list_of_truncated_output_files, str):
            output_filename_truncated = [join(output_folder_or_list_of_truncated_output_files, i) for i in caseids]
        else:
            output_filename_truncated = output_folder_or_list_of_truncated_output_files[part_id::num_parts]

        seg_from_prev_stage_files = [join(folder_with_segs_from_prev_stage, i + self.dataset_json['file_ending']) if
                                     folder_with_segs_from_prev_stage is not None else None for i in caseids]
        # remove already predicted files from the lists
        if not overwrite and output_filename_truncated is not None:
            tmp = [isfile(i + self.dataset_json['file_ending']) for i in output_filename_truncated]
            if save_probabilities:
                tmp2 = [isfile(i + '.npz') for i in output_filename_truncated]
                tmp = [i and j for i, j in zip(tmp, tmp2)]
            not_existing_indices = [i for i, j in enumerate(tmp) if not j]

            output_filename_truncated = [output_filename_truncated[i] for i in not_existing_indices]
            list_of_lists_or_source_folder = [list_of_lists_or_source_folder[i] for i in not_existing_indices]
            seg_from_prev_stage_files = [seg_from_prev_stage_files[i] for i in not_existing_indices]
            print(f'overwrite was set to {overwrite}, so I am only working on cases that haven\'t been predicted yet. '
                  f'That\'s {len(not_existing_indices)} cases.')
        return list_of_lists_or_source_folder, output_filename_truncated, seg_from_prev_stage_files

    def predict_from_files(self,
                           list_of_lists_or_source_folder: Union[str, List[List[str]]],
                           output_folder_or_list_of_truncated_output_files: Union[str, None, List[str]],
                           save_probabilities: bool = False,
                           overwrite: bool = True,
                           num_processes_preprocessing: int = default_num_processes,
                           num_processes_segmentation_export: int = default_num_processes,
                           folder_with_segs_from_prev_stage: str = None,
                           num_parts: int = 1,
                           part_id: int = 0):
        """
        This is nnU-Net's default function for making predictions. It works best for batch predictions
        (predicting many images at once).
        """
        assert part_id <= num_parts, ("Part ID must be smaller than num_parts. Remember that we start counting with 0. "
                                      "So if there are 3 parts then valid part IDs are 0, 1, 2")
        if isinstance(output_folder_or_list_of_truncated_output_files, str):
            output_folder = output_folder_or_list_of_truncated_output_files
        elif isinstance(output_folder_or_list_of_truncated_output_files, list):
            output_folder = os.path.dirname(output_folder_or_list_of_truncated_output_files[0])
        else:
            output_folder = None

        ########################
        # let's store the input arguments so that its clear what was used to generate the prediction
        if output_folder is not None:
            my_init_kwargs = {}
            for k in inspect.signature(self.predict_from_files).parameters.keys():
                my_init_kwargs[k] = locals()[k]
            my_init_kwargs = deepcopy(
                my_init_kwargs)  # let's not unintentionally change anything in-place. Take this as a
            recursive_fix_for_json_export(my_init_kwargs)
            maybe_mkdir_p(output_folder)
            save_json(my_init_kwargs, join(output_folder, 'predict_from_raw_data_args.json'))

            # we need these two if we want to do things with the predictions like for example apply postprocessing
            save_json(self.dataset_json, join(output_folder, 'dataset.json'), sort_keys=False)
            save_json(self.plans_manager.plans, join(output_folder, 'plans.json'), sort_keys=False)
        #######################

        # check if we need a prediction from the previous stage
        if self.configuration_manager.previous_stage_name is not None:
            assert folder_with_segs_from_prev_stage is not None, \
                f'The requested configuration is a cascaded network. It requires the segmentations of the previous ' \
                f'stage ({self.configuration_manager.previous_stage_name}) as input. Please provide the folder where' \
                f' they are located via folder_with_segs_from_prev_stage'

        # sort out input and output filenames
        list_of_lists_or_source_folder, output_filename_truncated, seg_from_prev_stage_files = \
            self._manage_input_and_output_lists(list_of_lists_or_source_folder,
                                                output_folder_or_list_of_truncated_output_files,
                                                folder_with_segs_from_prev_stage, overwrite, part_id, num_parts,
                                                save_probabilities)
        if len(list_of_lists_or_source_folder) == 0:
            return

        data_iterator = self._internal_get_data_iterator_from_lists_of_filenames(list_of_lists_or_source_folder,
                                                                                 seg_from_prev_stage_files,
                                                                                 output_filename_truncated,
                                                                                 num_processes_preprocessing)

        return self.predict_from_data_iterator(data_iterator, save_probabilities, num_processes_segmentation_export)

    def _internal_get_data_iterator_from_lists_of_filenames(self,
                                                            input_list_of_lists: List[List[str]],
                                                            seg_from_prev_stage_files: Union[List[str], None],
                                                            output_filenames_truncated: Union[List[str], None],
                                                            num_processes: int):
        return preprocessing_iterator_fromfiles(input_list_of_lists, seg_from_prev_stage_files,
                                                output_filenames_truncated, self.plans_manager, self.dataset_json,
                                                self.configuration_manager, num_processes, self.device.type == 'cuda',
                                                self.verbose_preprocessing)
        # preprocessor = self.configuration_manager.preprocessor_class(verbose=self.verbose_preprocessing)
        # # hijack batchgenerators, yo
        # # we use the multiprocessing of the batchgenerators dataloader to handle all the background worker stuff. This
        # # way we don't have to reinvent the wheel here.
        # num_processes = max(1, min(num_processes, len(input_list_of_lists)))
        # ppa = PreprocessAdapter(input_list_of_lists, seg_from_prev_stage_files, preprocessor,
        #                         output_filenames_truncated, self.plans_manager, self.dataset_json,
        #                         self.configuration_manager, num_processes)
        # if num_processes == 0:
        #     mta = SingleThreadedAugmenter(ppa, None)
        # else:
        #     mta = MultiThreadedAugmenter(ppa, None, num_processes, 1, None, pin_memory=pin_memory)
        # return mta

    def get_data_iterator_from_raw_npy_data(self,
                                            image_or_list_of_images: Union[np.ndarray, List[np.ndarray]],
                                            segs_from_prev_stage_or_list_of_segs_from_prev_stage: Union[None,
                                                                                                        np.ndarray,
                                                                                                        List[
                                                                                                            np.ndarray]],
                                            properties_or_list_of_properties: Union[dict, List[dict]],
                                            truncated_ofname: Union[str, List[str], None],
                                            num_processes: int = 3):

        list_of_images = [image_or_list_of_images] if not isinstance(image_or_list_of_images, list) else \
            image_or_list_of_images

        if isinstance(segs_from_prev_stage_or_list_of_segs_from_prev_stage, np.ndarray):
            segs_from_prev_stage_or_list_of_segs_from_prev_stage = [
                segs_from_prev_stage_or_list_of_segs_from_prev_stage]

        if isinstance(truncated_ofname, str):
            truncated_ofname = [truncated_ofname]

        if isinstance(properties_or_list_of_properties, dict):
            properties_or_list_of_properties = [properties_or_list_of_properties]

        num_processes = min(num_processes, len(list_of_images))
        pp = preprocessing_iterator_fromnpy(
            list_of_images,
            segs_from_prev_stage_or_list_of_segs_from_prev_stage,
            properties_or_list_of_properties,
            truncated_ofname,
            self.plans_manager,
            self.dataset_json,
            self.configuration_manager,
            num_processes,
            self.device.type == 'cuda',
            self.verbose_preprocessing
        )

        return pp

    def predict_from_list_of_npy_arrays(self,
                                        image_or_list_of_images: Union[np.ndarray, List[np.ndarray]],
                                        segs_from_prev_stage_or_list_of_segs_from_prev_stage: Union[None,
                                                                                                    np.ndarray,
                                                                                                    List[
                                                                                                        np.ndarray]],
                                        properties_or_list_of_properties: Union[dict, List[dict]],
                                        truncated_ofname: Union[str, List[str], None],
                                        num_processes: int = 3,
                                        save_probabilities: bool = False,
                                        num_processes_segmentation_export: int = default_num_processes):
        iterator = self.get_data_iterator_from_raw_npy_data(image_or_list_of_images,
                                                            segs_from_prev_stage_or_list_of_segs_from_prev_stage,
                                                            properties_or_list_of_properties,
                                                            truncated_ofname,
                                                            num_processes)
        return self.predict_from_data_iterator(iterator, save_probabilities, num_processes_segmentation_export)

    def predict_from_data_iterator(self,
                                   data_iterator,
                                   save_probabilities: bool = False,
                                   num_processes_segmentation_export: int = default_num_processes):
        """
        each element returned by data_iterator must be a dict with 'data', 'ofile' and 'data_properties' keys!
        If 'ofile' is None, the result will be returned instead of written to a file
        """
        with multiprocessing.get_context("spawn").Pool(num_processes_segmentation_export) as export_pool:
            worker_list = [i for i in export_pool._pool]
            r = []
            for preprocessed in data_iterator:
                data = preprocessed['data']
                if isinstance(data, str):
                    delfile = data
                    data = torch.from_numpy(np.load(data))
                    os.remove(delfile)

                ofile = preprocessed['ofile']
                if ofile is not None:
                    print(f'\nPredicting {os.path.basename(ofile)}:')
                else:
                    print(f'\nPredicting image of shape {data.shape}:')

                print(f'perform_everything_on_device: {self.perform_everything_on_device}')

                properties = preprocessed['data_properties']

                # let's not get into a runaway situation where the GPU predicts so fast that the disk has to b swamped with
                # npy files
                proceed = not check_workers_alive_and_busy(export_pool, worker_list, r, allowed_num_queued=2)
                while not proceed:
                    sleep(0.1)
                    proceed = not check_workers_alive_and_busy(export_pool, worker_list, r, allowed_num_queued=2)

                prediction = self.predict_logits_from_preprocessed_data(data).cpu()

                if ofile is not None:
                    # this needs to go into background processes
                    # export_prediction_from_logits(prediction, properties, self.configuration_manager, self.plans_manager,
                    #                               self.dataset_json, ofile, save_probabilities)
                    print('sending off prediction to background worker for resampling and export')
                    r.append(
                        export_pool.starmap_async(
                            export_prediction_from_logits,
                            ((prediction, properties, self.configuration_manager, self.plans_manager,
                              self.dataset_json, ofile, save_probabilities),)
                        )
                    )
                else:
                    # convert_predicted_logits_to_segmentation_with_correct_shape(
                    #             prediction, self.plans_manager,
                    #              self.configuration_manager, self.label_manager,
                    #              properties,
                    #              save_probabilities)

                    print('sending off prediction to background worker for resampling')
                    r.append(
                        export_pool.starmap_async(
                            convert_predicted_logits_to_segmentation_with_correct_shape, (
                                (prediction, self.plans_manager,
                                 self.configuration_manager, self.label_manager,
                                 properties,
                                 save_probabilities),)
                        )
                    )
                if ofile is not None:
                    print(f'done with {os.path.basename(ofile)}')
                else:
                    print(f'\nDone with image of shape {data.shape}:')
            ret = [i.get()[0] for i in r]

        if isinstance(data_iterator, MultiThreadedAugmenter):
            data_iterator._finish()

        # clear lru cache
        compute_gaussian.cache_clear()
        # clear device cache
        empty_cache(self.device)
        return ret

    def predict_single_npy_array(self, input_image: np.ndarray, image_properties: dict,
                                 segmentation_previous_stage: np.ndarray = None,
                                 output_file_truncated: str = None,
                                 save_or_return_probabilities: bool = False):
        """
        WARNING: SLOW. ONLY USE THIS IF YOU CANNOT GIVE NNUNET MULTIPLE IMAGES AT ONCE FOR SOME REASON.


        input_image: Make sure to load the image in the way nnU-Net expects! nnU-Net is trained on a certain axis
                     ordering which cannot be disturbed in inference,
                     otherwise you will get bad results. The easiest way to achieve that is to use the same I/O class
                     for loading images as was used during nnU-Net preprocessing! You can find that class in your
                     plans.json file under the key "image_reader_writer". If you decide to freestyle, know that the
                     default axis ordering for medical images is the one from SimpleITK. If you load with nibabel,
                     you need to transpose your axes AND your spacing from [x,y,z] to [z,y,x]!
        image_properties must only have a 'spacing' key!
        """
        ppa = PreprocessAdapterFromNpy([input_image], [segmentation_previous_stage], [image_properties],
                                       [output_file_truncated],
                                       self.plans_manager, self.dataset_json, self.configuration_manager,
                                       num_threads_in_multithreaded=1, verbose=self.verbose)
        if self.verbose:
            print('preprocessing')
        dct = next(ppa)

        if self.verbose:
            print('predicting')
        predicted_logits = self.predict_logits_from_preprocessed_data(dct['data']).cpu()

        if self.verbose:
            print('resampling to original shape')
        if output_file_truncated is not None:
            export_prediction_from_logits(predicted_logits, dct['data_properties'], self.configuration_manager,
                                          self.plans_manager, self.dataset_json, output_file_truncated,
                                          save_or_return_probabilities)
        else:
            ret = convert_predicted_logits_to_segmentation_with_correct_shape(predicted_logits, self.plans_manager,
                                                                              self.configuration_manager,
                                                                              self.label_manager,
                                                                              dct['data_properties'],
                                                                              return_probabilities=
                                                                              save_or_return_probabilities)
            if save_or_return_probabilities:
                return ret[0], ret[1]
            else:
                return ret

    @torch.inference_mode()
    def predict_logits_from_preprocessed_data(self, data: torch.Tensor) -> torch.Tensor:
        """
        IMPORTANT! IF YOU ARE RUNNING THE CASCADE, THE SEGMENTATION FROM THE PREVIOUS STAGE MUST ALREADY BE STACKED ON
        TOP OF THE IMAGE AS ONE-HOT REPRESENTATION! SEE PreprocessAdapter ON HOW THIS SHOULD BE DONE!

        RETURNED LOGITS HAVE THE SHAPE OF THE INPUT. THEY MUST BE CONVERTED BACK TO THE ORIGINAL IMAGE SIZE.
        SEE convert_predicted_logits_to_segmentation_with_correct_shape
        """
        n_threads = torch.get_num_threads()
        torch.set_num_threads(default_num_processes if default_num_processes < n_threads else n_threads)
        prediction = None

        for params in self.list_of_parameters:

            # messing with state dict names...
            # if not isinstance(self.network, OptimizedModule):
            #     self.network.load_state_dict(params)
            # else:
            #     self.network._orig_mod.load_state_dict(params)

            # why not leave prediction on device if perform_everything_on_device? Because this may cause the
            # second iteration to crash due to OOM. Grabbing that with try except cause way more bloated code than
            # this actually saves computation time
            if prediction is None:
                prediction = self.predict_sliding_window_return_logits(data).to('cpu')
            else:
                prediction += self.predict_sliding_window_return_logits(data).to('cpu')

        if len(self.list_of_parameters) > 1:
            prediction /= len(self.list_of_parameters)

        if self.verbose: print('Prediction done')
        torch.set_num_threads(n_threads)
        return prediction

    def _internal_get_sliding_window_slicers(self, image_size: Tuple[int, ...]):
        slicers = []
        if len(self.configuration_manager.patch_size) < len(image_size):
            assert len(self.configuration_manager.patch_size) == len(
                image_size) - 1, 'if tile_size has less entries than image_size, ' \
                                 'len(tile_size) ' \
                                 'must be one shorter than len(image_size) ' \
                                 '(only dimension ' \
                                 'discrepancy of 1 allowed).'
            steps = compute_steps_for_sliding_window(image_size[1:], self.configuration_manager.patch_size,
                                                     self.tile_step_size)
            if self.verbose: print(f'n_steps {image_size[0] * len(steps[0]) * len(steps[1])}, image size is'
                                   f' {image_size}, tile_size {self.configuration_manager.patch_size}, '
                                   f'tile_step_size {self.tile_step_size}\nsteps:\n{steps}')
            for d in range(image_size[0]):
                for sx in steps[0]:
                    for sy in steps[1]:
                        slicers.append(
                            tuple([slice(None), d, *[slice(si, si + ti) for si, ti in
                                                     zip((sx, sy), self.configuration_manager.patch_size)]]))
        else:
            steps = compute_steps_for_sliding_window(image_size, self.configuration_manager.patch_size,
                                                     self.tile_step_size)
            if self.verbose: print(
                f'n_steps {np.prod([len(i) for i in steps])}, image size is {image_size}, tile_size {self.configuration_manager.patch_size}, '
                f'tile_step_size {self.tile_step_size}\nsteps:\n{steps}')
            for sx in steps[0]:
                for sy in steps[1]:
                    for sz in steps[2]:
                        slicers.append(
                            tuple([slice(None), *[slice(si, si + ti) for si, ti in
                                                  zip((sx, sy, sz), self.configuration_manager.patch_size)]]))
        return slicers

    @torch.inference_mode()
    def _internal_maybe_mirror_and_predict(self, x: torch.Tensor) -> torch.Tensor:
        mirror_axes = self.allowed_mirroring_axes if self.use_mirroring else None
        prediction = self.network(x)

        if mirror_axes is not None:
            # check for invalid numbers in mirror_axes
            # x should be 5d for 3d images and 4d for 2d. so the max value of mirror_axes cannot exceed len(x.shape) - 3
            assert max(mirror_axes) <= x.ndim - 3, 'mirror_axes does not match the dimension of the input!'

            mirror_axes = [m + 2 for m in mirror_axes]
            axes_combinations = [
                c for i in range(len(mirror_axes)) for c in itertools.combinations(mirror_axes, i + 1)
            ]
            for axes in axes_combinations:
                prediction += torch.flip(self.network(torch.flip(x, axes)), axes)
            prediction /= (len(axes_combinations) + 1)
        return prediction

    @torch.inference_mode()
    def _internal_predict_sliding_window_return_logits(self,
                                                       data: torch.Tensor,
                                                       slicers,
                                                       do_on_device: bool = True,
                                                       ):
        predicted_logits = n_predictions = prediction = gaussian = workon = None
        results_device = self.device if do_on_device else torch.device('cpu')

        def producer(d, slh, q):
            for s in slh:
                q.put((torch.clone(d[s][None], memory_format=torch.contiguous_format).to(self.device), s))
            q.put('end')

        try:
            empty_cache(self.device)

            # move data to device
            if self.verbose:
                print(f'move image to device {results_device}')
            data = data.to(results_device)
            queue = Queue(maxsize=2)
            t = Thread(target=producer, args=(data, slicers, queue))
            t.start()

            # preallocate arrays
            if self.verbose:
                print(f'preallocating results arrays on device {results_device}')
            predicted_logits = torch.zeros((self.label_manager.num_segmentation_heads, *data.shape[1:]),
                                           dtype=torch.float32,
                                           device=results_device)
            n_predictions = torch.zeros(data.shape[1:], dtype=torch.float32, device=results_device)

            if self.use_gaussian:
                gaussian = compute_gaussian(tuple(self.configuration_manager.patch_size), sigma_scale=1. / 8,
                                            value_scaling_factor=10,
                                            dtype=predicted_logits.dtype,
                                            device=results_device)
            else:
                gaussian = 1

            if not self.allow_tqdm and self.verbose:
                print(f'running prediction: {len(slicers)} steps')

            with tqdm(desc=None, total=len(slicers), disable=not self.allow_tqdm) as pbar:
                while True:
                    item = queue.get()
                    if item == 'end':
                        queue.task_done()
                        break
                    workon, sl = item
                    prediction = self._internal_maybe_mirror_and_predict(workon)[0].to(
                        results_device,
                        dtype=predicted_logits.dtype,
                    )
                    if not torch.isfinite(prediction).all():
                        raise RuntimeError(
                            'Encountered non-finite tile prediction before sliding-window accumulation. '
                            'Inspect model logits for inf/nan in the gmm_anomaly inference path.'
                        )

                    if self.use_gaussian:
                        prediction *= gaussian
                    predicted_logits[sl] += prediction
                    n_predictions[sl[1:]] += gaussian
                    queue.task_done()
                    pbar.update()
            queue.join()

            # predicted_logits /= n_predictions
            torch.div(predicted_logits, n_predictions, out=predicted_logits)
            # check for infs
            if torch.any(torch.isinf(predicted_logits)):
                raise RuntimeError('Encountered inf in predicted array. Aborting... If this problem persists, '
                                   'reduce value_scaling_factor in compute_gaussian or increase the dtype of '
                                   'predicted_logits to fp32')
        except Exception as e:
            del predicted_logits, n_predictions, prediction, gaussian, workon
            empty_cache(self.device)
            empty_cache(results_device)
            raise e
        return predicted_logits

    @torch.inference_mode()
    def predict_sliding_window_return_logits(self, input_image: torch.Tensor) \
            -> Union[np.ndarray, torch.Tensor]:
        assert isinstance(input_image, torch.Tensor)
        self.network = self.network.to(self.device)
        self.network.eval()

        empty_cache(self.device)

        # Autocast can be annoying
        # If the device_type is 'cpu' then it's slow as heck on some CPUs (no auto bfloat16 support detection)
        # and needs to be disabled.
        # If the device_type is 'mps' then it will complain that mps is not implemented, even if enabled=False
        # is set. Whyyyyyyy. (this is why we don't make use of enabled=False)
        # So autocast will only be active if we have a cuda device.
        with torch.autocast(self.device.type, enabled=True) if self.device.type == 'cuda' else dummy_context():
            assert input_image.ndim == 4, 'input_image must be a 4D np.ndarray or torch.Tensor (c, x, y, z)'

            if self.verbose:
                print(f'Input shape: {input_image.shape}')
                print("step_size:", self.tile_step_size)
                print("mirror_axes:", self.allowed_mirroring_axes if self.use_mirroring else None)

            # if input_image is smaller than tile_size we need to pad it to tile_size.
            data, slicer_revert_padding = pad_nd_image(input_image, self.configuration_manager.patch_size,
                                                       'constant', {'value': 0}, True,
                                                       None)

            slicers = self._internal_get_sliding_window_slicers(data.shape[1:])

            if self.perform_everything_on_device and self.device != 'cpu':
                # we need to try except here because we can run OOM in which case we need to fall back to CPU as a results device
                try:
                    predicted_logits = self._internal_predict_sliding_window_return_logits(data, slicers,
                                                                                           self.perform_everything_on_device)
                except RuntimeError:
                    print(
                        'Prediction on device was unsuccessful, probably due to a lack of memory. Moving results arrays to CPU')
                    empty_cache(self.device)
                    predicted_logits = self._internal_predict_sliding_window_return_logits(data, slicers, False)
            else:
                predicted_logits = self._internal_predict_sliding_window_return_logits(data, slicers,
                                                                                       self.perform_everything_on_device)

            empty_cache(self.device)
            # revert padding
            predicted_logits = predicted_logits[(slice(None), *slicer_revert_padding[1:])]
        return predicted_logits

    def predict_from_files_sequential(self,
                           list_of_lists_or_source_folder: Union[str, List[List[str]]],
                           output_folder_or_list_of_truncated_output_files: Union[str, None, List[str]],
                           save_probabilities: bool = False,
                           overwrite: bool = True,
                           folder_with_segs_from_prev_stage: str = None):
        """
        Just like predict_from_files but doesn't use any multiprocessing. Slow, but sometimes necessary
        """
        if isinstance(output_folder_or_list_of_truncated_output_files, str):
            output_folder = output_folder_or_list_of_truncated_output_files
        elif isinstance(output_folder_or_list_of_truncated_output_files, list):
            output_folder = os.path.dirname(output_folder_or_list_of_truncated_output_files[0])
            if len(output_folder) == 0:  # just a file was given without a folder
                output_folder = os.path.curdir
        else:
            output_folder = None

        ########################
        # let's store the input arguments so that its clear what was used to generate the prediction
        if output_folder is not None:
            my_init_kwargs = {}
            for k in inspect.signature(self.predict_from_files_sequential).parameters.keys():
                my_init_kwargs[k] = locals()[k]
            my_init_kwargs = deepcopy(
                my_init_kwargs)  # let's not unintentionally change anything in-place. Take this as a
            recursive_fix_for_json_export(my_init_kwargs)
            save_json(my_init_kwargs, join(output_folder, 'predict_from_raw_data_args.json'))

            # we need these two if we want to do things with the predictions like for example apply postprocessing
            save_json(self.dataset_json, join(output_folder, 'dataset.json'), sort_keys=False)
            save_json(self.plans_manager.plans, join(output_folder, 'plans.json'), sort_keys=False)
        #######################

        # check if we need a prediction from the previous stage
        if self.configuration_manager.previous_stage_name is not None:
            assert folder_with_segs_from_prev_stage is not None, \
                f'The requested configuration is a cascaded network. It requires the segmentations of the previous ' \
                f'stage ({self.configuration_manager.previous_stage_name}) as input. Please provide the folder where' \
                f' they are located via folder_with_segs_from_prev_stage'

        # sort out input and output filenames
        list_of_lists_or_source_folder, output_filename_truncated, seg_from_prev_stage_files = \
            self._manage_input_and_output_lists(list_of_lists_or_source_folder,
                                                output_folder_or_list_of_truncated_output_files,
                                                folder_with_segs_from_prev_stage, overwrite, 0, 1,
                                                save_probabilities)
        if len(list_of_lists_or_source_folder) == 0:
            return

        label_manager = self.plans_manager.get_label_manager(self.dataset_json)
        preprocessor = self.configuration_manager.preprocessor_class(verbose=self.verbose)

        if output_filename_truncated is None:
            output_filename_truncated = [None] * len(list_of_lists_or_source_folder)
        if seg_from_prev_stage_files is None:
            seg_from_prev_stage_files = [None] * len(seg_from_prev_stage_files)

        ret = []
        for li, of, sps in zip(list_of_lists_or_source_folder, output_filename_truncated, seg_from_prev_stage_files):
            data, seg, data_properties = preprocessor.run_case(
                li,
                sps,
                self.plans_manager,
                self.configuration_manager,
                self.dataset_json
            )

            print(f'perform_everything_on_device: {self.perform_everything_on_device}')

            prediction = self.predict_logits_from_preprocessed_data(torch.as_tensor(np.asarray(data))).cpu()

            if of is not None:
                export_prediction_from_logits(prediction, data_properties, self.configuration_manager, self.plans_manager,
                  self.dataset_json, of, save_probabilities)
            else:
                ret.append(convert_predicted_logits_to_segmentation_with_correct_shape(prediction, self.plans_manager,
                     self.configuration_manager, self.label_manager,
                     data_properties,
                     save_probabilities))

        # clear lru cache
        compute_gaussian.cache_clear()
        # clear device cache
        empty_cache(self.device)
        return ret

    def save_stats(self, path, list_of_lists_or_source_folder, output_folder_or_list_of_truncated_output_files,folder_with_segs_from_prev_stage):
        overwrite = True
        part_id = 0
        num_parts = 1
        save_probabilities = False
        num_processes_preprocessing = 1
        num_processes_segmentation_export = 1
        list_of_lists_or_source_folder, output_filename_truncated, seg_from_prev_stage_files = \
            self._manage_input_and_output_lists(list_of_lists_or_source_folder,
                                                output_folder_or_list_of_truncated_output_files,
                                                folder_with_segs_from_prev_stage, overwrite, part_id, num_parts,
                                                save_probabilities)
        if len(list_of_lists_or_source_folder) == 0:
            return
        data_iterator = self._internal_get_data_iterator_from_lists_of_filenames(list_of_lists_or_source_folder,
                                                                                 seg_from_prev_stage_files,
                                                                                 output_filename_truncated,
                                                                                 num_processes_preprocessing)
        from nnunetv2.utilities.additional.visualize import SegmentationFeatureVisualizer
        extractor = SegmentationFeatureVisualizer(self.network)
        for preprocessed in data_iterator:
            data = preprocessed['data']
            self.patient_name = preprocessed['ofile'].split("/")[-1]
            if isinstance(data, str):
                delfile = data
                data = torch.from_numpy(np.load(data))
                os.remove(delfile)
            params = self.list_of_parameters[0]
            if not isinstance(self.network, OptimizedModule):
                self.network.load_state_dict(params)
            else:
                self.network._orig_mod.load_state_dict(params)

            input_image = data
            self.network = self.network.to(self.device)
            self.network.eval()

            empty_cache(self.device)
            context_autocast = torch.autocast(self.device.type, enabled=True) if self.device.type == 'cuda' else dummy_context()
            with torch.no_grad(), context_autocast:
                assert input_image.ndim == 4, 'input_image must be a 4D np.ndarray or torch.Tensor (c, x, y, z)'

                # if input_image is smaller than tile_size we need to pad it to tile_size.
                data, slicer_revert_padding = pad_nd_image(input_image, self.configuration_manager.patch_size,
                                                        'constant', {'value': 0}, True,
                                                        None)
                slicers = self._internal_get_sliding_window_slicers(data.shape[1:])

                with tqdm(desc=None, total=len(slicers), disable=not self.allow_tqdm) as pbar:
                    for s in slicers:
                        data_s = torch.clone(data[s][None], memory_format=torch.contiguous_format).to(self.device)
                        input = data_s[:,0:1]
                        target = data_s[:,1:]
                        prediction = self.network(input)
                        new_target = torch.zeros_like(target[:,0])
                        new_target = target[:,0] * 1 + target[:,1] * 2
                        new_target = new_target.to(torch.long)
                        extractor.masks = new_target[None]
                        feats, labels = extractor.extract_features()["decoder_last_stage_feat"]
                        self.evaluator.partial_fit(feats, labels)
                        del data_s, input, target, prediction, feats, new_target
                        pbar.update(1)

        self.evaluator.finalize()
        self.evaluator.save(path)
        empty_cache(self.device)
    

def predict_entry_point_modelfolder():
    import argparse
    parser = argparse.ArgumentParser(description='Use this to run inference with nnU-Net. This function is used when '
                                                 'you want to manually specify a folder containing a trained nnU-Net '
                                                 'model. This is useful when the nnunet environment variables '
                                                 '(nnUNet_results) are not set.')
    parser.add_argument('-i', type=str, required=True,
                        help='input folder. Remember to use the correct channel numberings for your files (_0000 etc). '
                             'File endings must be the same as the training dataset!')
    parser.add_argument('-o', type=str, required=True,
                        help='Output folder. If it does not exist it will be created. Predicted segmentations will '
                             'have the same name as their source images.')
    parser.add_argument('-m', type=str, required=True,
                        help='Folder in which the trained model is. Must have subfolders fold_X for the different '
                             'folds you trained')
    parser.add_argument('-f', nargs='+', type=str, required=False, default=(0, 1, 2, 3, 4),
                        help='Specify the folds of the trained model that should be used for prediction. '
                             'Default: (0, 1, 2, 3, 4)')
    parser.add_argument('-step_size', type=float, required=False, default=0.5,
                        help='Step size for sliding window prediction. The larger it is the faster but less accurate '
                             'the prediction. Default: 0.5. Cannot be larger than 1. We recommend the default.')
    parser.add_argument('--disable_tta', action='store_true', required=False, default=False,
                        help='Set this flag to disable test time data augmentation in the form of mirroring. Faster, '
                             'but less accurate inference. Not recommended.')
    parser.add_argument('--verbose', action='store_true', help="Set this if you like being talked to. You will have "
                                                               "to be a good listener/reader.")
    parser.add_argument('--save_probabilities', action='store_true',
                        help='Set this to export predicted class "probabilities". Required if you want to ensemble '
                             'multiple configurations.')
    parser.add_argument('--continue_prediction', '--c', action='store_true',
                        help='Continue an aborted previous prediction (will not overwrite existing files)')
    parser.add_argument('-chk', type=str, required=False, default='checkpoint_final.pth',
                        help='Name of the checkpoint you want to use. Default: checkpoint_final.pth')
    parser.add_argument('-npp', type=int, required=False, default=3,
                        help='Number of processes used for preprocessing. More is not always better. Beware of '
                             'out-of-RAM issues. Default: 3')
    parser.add_argument('-nps', type=int, required=False, default=3,
                        help='Number of processes used for segmentation export. More is not always better. Beware of '
                             'out-of-RAM issues. Default: 3')
    parser.add_argument('-prev_stage_predictions', type=str, required=False, default=None,
                        help='Folder containing the predictions of the previous stage. Required for cascaded models.')
    parser.add_argument('-device', type=str, default='cuda', required=False,
                        help="Use this to set the device the inference should run with. Available options are 'cuda' "
                             "(GPU), 'cpu' (CPU) and 'mps' (Apple M1/M2). Do NOT use this to set which GPU ID! "
                             "Use CUDA_VISIBLE_DEVICES=X nnUNetv2_predict [...] instead!")
    parser.add_argument('--disable_progress_bar', action='store_true', required=False, default=False,
                        help='Set this flag to disable progress bar. Recommended for HPC environments (non interactive '
                             'jobs)')
    parser.add_argument('--predictor_family', type=str, required=False, default=None,
                        help='Optional predictor family override, for example gmm_calibrated.')
    parser.add_argument('--predictor_metric', type=str, required=False, default='seg',
                        help='Predictor metric/task selector. Default: seg')
    parser.add_argument('--predictor_option', type=int, required=False, default=None,
                        help='Optional gmm_calibrated inference option override.')
    parser.add_argument('--predictor_mode', type=str, required=False, default=None,
                        help='Optional gmm_calibrated classifier mode override.')
    parser.add_argument('--predictor_transition_mode', type=str, required=False, default=None,
                        help='Optional gmm_calibrated transition mode override.')
    parser.add_argument('--predictor_num_domains', type=int, required=False, default=None,
                        help='Optional gmm_calibrated num_domains override.')
    parser.add_argument('--predictor_source_trainer_name', type=str, required=False, default=None,
                        help='Optional gmm_calibrated source trainer name override.')
    parser.add_argument('--predictor_source_model_training_output_dir', type=str, required=False, default=None,
                        help='Optional explicit source model folder for gmm_calibrated predictor.')

    print(
        "\n#######################################################################\nPlease cite the following paper "
        "when using nnU-Net:\n"
        "Isensee, F., Jaeger, P. F., Kohl, S. A., Petersen, J., & Maier-Hein, K. H. (2021). "
        "nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation. "
        "Nature methods, 18(2), 203-211.\n#######################################################################\n")

    args = parser.parse_args()
    args.f = [i if i == 'all' else int(i) for i in args.f]

    if not isdir(args.o):
        maybe_mkdir_p(args.o)

    device = _configure_inference_device(args.device)

    shared_predictor_kwargs = {
        "tile_step_size": args.step_size,
        "use_gaussian": True,
        "use_mirroring": not args.disable_tta,
        "perform_everything_on_device": device.type == 'cuda',
        "device": device,
        "verbose": args.verbose,
        "verbose_preprocessing": args.verbose,
        "allow_tqdm": not args.disable_progress_bar,
    }
    if args.predictor_family is not None:
        predictor, predictor_family = _build_predictor_by_family(
            args.predictor_family,
            shared_predictor_kwargs,
            _collect_predictor_family_kwargs(args),
        )
    else:
        predictor, predictor_family = _build_predictor_for_model_folder(
            model_folder=args.m,
            device=device,
            tile_step_size=args.step_size,
            use_gaussian=True,
            use_mirroring=not args.disable_tta,
            verbose=args.verbose,
            verbose_preprocessing=args.verbose,
            allow_tqdm=not args.disable_progress_bar,
            metric=args.predictor_metric,
        )
    print(f'Using predictor family: {predictor_family}')
    predictor.initialize_from_trained_model_folder(args.m, args.f, args.chk)
    predictor.predict_from_files(args.i, args.o, save_probabilities=args.save_probabilities,
                                 overwrite=not args.continue_prediction,
                                 num_processes_preprocessing=args.npp,
                                 num_processes_segmentation_export=args.nps,
                                 folder_with_segs_from_prev_stage=args.prev_stage_predictions,
                                 num_parts=1, part_id=0)


def predict_entry_point():
    import argparse
    parser = argparse.ArgumentParser(description='Use this to run inference with nnU-Net. This function is used when '
                                                 'you want to manually specify a folder containing a trained nnU-Net '
                                                 'model. This is useful when the nnunet environment variables '
                                                 '(nnUNet_results) are not set.')
    parser.add_argument('-i', type=str, required=True,
                        help='input folder. Remember to use the correct channel numberings for your files (_0000 etc). '
                             'File endings must be the same as the training dataset!')
    parser.add_argument('-o', type=str, required=True,
                        help='Output folder. If it does not exist it will be created. Predicted segmentations will '
                             'have the same name as their source images.')
    parser.add_argument('-d', type=str, required=True,
                        help='Dataset with which you would like to predict. You can specify either dataset name or id')
    parser.add_argument('-p', type=str, required=False, default='nnUNetPlans',
                        help='Plans identifier. Specify the plans in which the desired configuration is located. '
                             'Default: nnUNetPlans')
    parser.add_argument('-tr', type=str, required=False, default='nnUNetTrainer',
                        help='What nnU-Net trainer class was used for training? Default: nnUNetTrainer')
    parser.add_argument('-c', type=str, required=True,
                        help='nnU-Net configuration that should be used for prediction. Config must be located '
                             'in the plans specified with -p')
    parser.add_argument('-f', nargs='+', type=str, required=False, default=(0, 1, 2, 3, 4),
                        help='Specify the folds of the trained model that should be used for prediction. '
                             'Default: (0, 1, 2, 3, 4)')
    parser.add_argument('-step_size', type=float, required=False, default=0.5,
                        help='Step size for sliding window prediction. The larger it is the faster but less accurate '
                             'the prediction. Default: 0.5. Cannot be larger than 1. We recommend the default.')
    parser.add_argument('--disable_tta', action='store_true', required=False, default=False,
                        help='Set this flag to disable test time data augmentation in the form of mirroring. Faster, '
                             'but less accurate inference. Not recommended.')
    parser.add_argument('--verbose', action='store_true', help="Set this if you like being talked to. You will have "
                                                               "to be a good listener/reader.")
    parser.add_argument('--save_probabilities', action='store_true',
                        help='Set this to export predicted class "probabilities". Required if you want to ensemble '
                             'multiple configurations.')
    parser.add_argument('--continue_prediction', action='store_true',
                        help='Continue an aborted previous prediction (will not overwrite existing files)')
    parser.add_argument('-chk', type=str, required=False, default='checkpoint_final.pth',
                        help='Name of the checkpoint you want to use. Default: checkpoint_final.pth')
    parser.add_argument('-npp', type=int, required=False, default=3,
                        help='Number of processes used for preprocessing. More is not always better. Beware of '
                             'out-of-RAM issues. Default: 3')
    parser.add_argument('-nps', type=int, required=False, default=3,
                        help='Number of processes used for segmentation export. More is not always better. Beware of '
                             'out-of-RAM issues. Default: 3')
    parser.add_argument('-prev_stage_predictions', type=str, required=False, default=None,
                        help='Folder containing the predictions of the previous stage. Required for cascaded models.')
    parser.add_argument('-num_parts', type=int, required=False, default=1,
                        help='Number of separate nnUNetv2_predict call that you will be making. Default: 1 (= this one '
                             'call predicts everything)')
    parser.add_argument('-part_id', type=int, required=False, default=0,
                        help='If multiple nnUNetv2_predict exist, which one is this? IDs start with 0 can end with '
                             'num_parts - 1. So when you submit 5 nnUNetv2_predict calls you need to set -num_parts '
                             '5 and use -part_id 0, 1, 2, 3 and 4. Simple, right? Note: You are yourself responsible '
                             'to make these run on separate GPUs! Use CUDA_VISIBLE_DEVICES (google, yo!)')
    parser.add_argument('-device', type=str, default='cuda', required=False,
                        help="Use this to set the device the inference should run with. Available options are 'cuda' "
                             "(GPU), 'cpu' (CPU) and 'mps' (Apple M1/M2). Do NOT use this to set which GPU ID! "
                             "Use CUDA_VISIBLE_DEVICES=X nnUNetv2_predict [...] instead!")
    parser.add_argument('--disable_progress_bar', action='store_true', required=False, default=False,
                        help='Set this flag to disable progress bar. Recommended for HPC environments (non interactive '
                             'jobs)')
    parser.add_argument('--predictor_family', type=str, required=False, default=None,
                        help='Optional predictor family override, for example gmm_calibrated.')
    parser.add_argument('--predictor_metric', type=str, required=False, default='seg',
                        help='Predictor metric/task selector. Default: seg')
    parser.add_argument('--predictor_option', type=int, required=False, default=None,
                        help='Optional gmm_calibrated inference option override.')
    parser.add_argument('--predictor_mode', type=str, required=False, default=None,
                        help='Optional gmm_calibrated classifier mode override.')
    parser.add_argument('--predictor_transition_mode', type=str, required=False, default=None,
                        help='Optional gmm_calibrated transition mode override.')
    parser.add_argument('--predictor_num_domains', type=int, required=False, default=None,
                        help='Optional gmm_calibrated num_domains override.')
    parser.add_argument('--predictor_source_trainer_name', type=str, required=False, default=None,
                        help='Optional gmm_calibrated source trainer name override.')
    parser.add_argument('--predictor_source_model_training_output_dir', type=str, required=False, default=None,
                        help='Optional explicit source model folder for gmm_calibrated predictor.')

    print(
        "\n#######################################################################\nPlease cite the following paper "
        "when using nnU-Net:\n"
        "Isensee, F., Jaeger, P. F., Kohl, S. A., Petersen, J., & Maier-Hein, K. H. (2021). "
        "nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation. "
        "Nature methods, 18(2), 203-211.\n#######################################################################\n")

    args = parser.parse_args()
    args.f = [i if i == 'all' else int(i) for i in args.f]

    model_folder = get_output_folder(args.d, args.tr, args.p, args.c)

    if not isdir(args.o):
        maybe_mkdir_p(args.o)

    # slightly passive aggressive haha
    assert args.part_id < args.num_parts, 'Do you even read the documentation? See nnUNetv2_predict -h.'

    device = _configure_inference_device(args.device)

    shared_predictor_kwargs = {
        "tile_step_size": args.step_size,
        "use_gaussian": True,
        "use_mirroring": not args.disable_tta,
        "perform_everything_on_device": device.type == 'cuda',
        "device": device,
        "verbose": args.verbose,
        "verbose_preprocessing": args.verbose,
        "allow_tqdm": not args.disable_progress_bar,
    }
    if args.predictor_family is not None:
        predictor, predictor_family = _build_predictor_by_family(
            args.predictor_family,
            shared_predictor_kwargs,
            _collect_predictor_family_kwargs(args),
        )
    else:
        predictor, predictor_family = _build_predictor_for_model_folder(
            model_folder=model_folder,
            device=device,
            tile_step_size=args.step_size,
            use_gaussian=True,
            use_mirroring=not args.disable_tta,
            verbose=args.verbose,
            verbose_preprocessing=args.verbose,
            allow_tqdm=not args.disable_progress_bar,
            metric=args.predictor_metric,
        )
    print(f'Using predictor family: {predictor_family}')
    predictor.initialize_from_trained_model_folder(
        model_folder,
        args.f,
        checkpoint_name=args.chk
    )
    predictor.predict_from_files(args.i, args.o, save_probabilities=args.save_probabilities,
                                 overwrite=not args.continue_prediction,
                                 num_processes_preprocessing=args.npp,
                                 num_processes_segmentation_export=args.nps,
                                 folder_with_segs_from_prev_stage=args.prev_stage_predictions,
                                 num_parts=args.num_parts,
                                 part_id=args.part_id)
    # r = predict_from_raw_data(args.i,
    #                           args.o,
    #                           model_folder,
    #                           args.f,
    #                           args.step_size,
    #                           use_gaussian=True,
    #                           use_mirroring=not args.disable_tta,
    #                           perform_everything_on_device=True,
    #                           verbose=args.verbose,
    #                           save_probabilities=args.save_probabilities,
    #                           overwrite=not args.continue_prediction,
    #                           checkpoint_name=args.chk,
    #                           num_processes_preprocessing=args.npp,
    #                           num_processes_segmentation_export=args.nps,
    #                           folder_with_segs_from_prev_stage=args.prev_stage_predictions,
    #                           num_parts=args.num_parts,
    #                           part_id=args.part_id,
    #                           device=device)


if __name__ == '__main__':
    ########################## predict a bunch of files
    # Default to one explicit registered experiment so this file can be run directly
    # without having to remember the Python one-liner used during manual batches.
    run_mode = "registered_experiment"
    # Common TableA experiments:
    #   tableA_metric_007_self_fold4
    #   tableA_multi_proto_007_self_fold4
    #   tableA_metric_009_self_fold4
    #   tableA_multi_proto_009_self_fold4
    #   tableA_slice_adaptive_009_self_fold4
    registered_experiment_name = "multi_dot_fixed_bg_softmax_test_007_test_fold4"
    registered_suite_name = "tablea_self_domain_fold4_all"

    if run_mode == "registered_suite":
        print(f"[Registry] running suite: {registered_suite_name}")
        run_registered_suite(
            suite_name=registered_suite_name,
            device_name="cuda",
            results_root=DEFAULT_RESULTS_ROOT,
            continue_prediction=True,
            save_probabilities=False,
            disable_tta=False,
            verbose=False,
            disable_progress_bar=False,
            num_processes_preprocessing=1,
            num_processes_segmentation_export=1,
        )
        raise SystemExit(0)

    if run_mode == "registered_experiment":
        print(f"[Registry] running experiment: {registered_experiment_name}")
        run_registered_experiment(
            experiment_name=registered_experiment_name,
            device_name="cuda",
            results_root=DEFAULT_RESULTS_ROOT,
            continue_prediction=True,
            save_probabilities=False,
            disable_tta=False,
            verbose=False,
            disable_progress_bar=False,
            num_processes_preprocessing=1,
            num_processes_segmentation_export=1,
        )
        raise SystemExit(0)

    if run_mode == "tablea_suite":
        run_tablea_self_domain_suite(
            fold=4,
            methods=("metric", "multi_proto", "slice_adaptive"),
            datasets=("002", "007", "009"),
            device_name="cuda",
            configuration="2d",
            checkpoint_name="checkpoint_final.pth",
            results_root=DEFAULT_RESULTS_ROOT,
            continue_prediction=True,
            save_probabilities=False,
            disable_tta=False,
            verbose=False,
            disable_progress_bar=False,
            num_processes_preprocessing=1,
            num_processes_segmentation_export=1,
        )
        raise SystemExit(0)
