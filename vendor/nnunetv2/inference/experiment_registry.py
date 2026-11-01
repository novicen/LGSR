"""Optional experiment registry placeholder for the standalone LGSR release."""

import os

DEFAULT_RESULTS_ROOT = os.environ.get("LGSR_RESULTS_ROOT", os.environ.get("nnUNet_results", "./nnUNet_results"))
ACTIVE_EXPERIMENTS = {}
EXPERIMENT_SUITES = {}


def make_tablea_self_domain_experiment(*args, **kwargs):
    raise NotImplementedError("The standalone release does not ship experiment registries")
