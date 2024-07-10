import glob
import os
from argparse import Namespace

import mrunner
from mrunner.helpers import client_helper
from mrunner.helpers.client_helper import get_configuration

from src.train import main


mrunner.settings.NEPTUNE_USE_NEW_API = True

if __name__ == "__main__":
    params = get_configuration(
        print_diagnostics=True, with_neptune=True, inject_parameters_to_gin=True
    )
    params.pop("experiment_id")
    # verity params: combine_with_defaults(params, defaults=vars(parse_namespace([])))
    namespace = Namespace(**params)
    try:
        client_helper.experiment_["neptune_path"] = os.environ["NEPTUNEPWD"]
    except KeyError:
        pass

    if not namespace.cpdag and namespace.sampling:
        pass  # add constraints here

    main(namespace)
