import argparse
from copy import deepcopy


def combine_with_defaults(config, defaults):
    res = deepcopy(defaults)
    for k, v in config.items():
        assert k in res.keys(), "{} not in default values".format(k)
        res[k] = v
    return res


def parse_args(args=None):
    parser = get_parser()
    return parser.parse_args(args=args)


def get_parser():
    """Parses command line arguments."""
    parser = argparse.ArgumentParser(description="Validate model")
    main_parser(parser)
    train_parser(parser)
    return parser.parse_args()


def main_parser(parser):  # Check if needs to return parser
    parser.add_argument(
        "--method",
        choices=["omivae", "babel", "advae", "vae"],
        help="Name of the method to use.",
    )
    parser.add_argument(
        "--cv-seed", default=42, help="Seed used to make k folds for cross validation."
    )
    parser.add_argument(
        "--n-folds", default=5, type=int, help="Number of folds in cross validation."
    )
    parser.add_argument(
        "--plus-iid-holdout",
        action="store_true",
        help="Whether to include the iid_holdout data in the anndata object.",
    )
    parser.add_argument(
        "--config",
        default="standard",
        help="Name of a configuration in src/config/{method}.yaml.",
    )
    parser.add_argument(
        "--preload-subsample-frac",
        default=None,
        type=float,
        help="Fraction of the data to load. If None, use all data. Don't use subsample-frac with this option.",
    )
    parser.add_argument(
        "--subsample-frac",
        default=None,
        type=float,
        help="Fraction of the data to use for cross validation. If None, use all data.",
    )
    parser.add_argument(
        "--retrain", default=True, help="Retrain a model using the whole dataset."
    )


def train_parser(parser):
    # Training arguments
    parser.add_argument("--lr", default=1e-3, type=float, help="Learning rate.")
    parser.add_argument("--batch-size", default=64, type=int, help="Batch size.")
    parser.add_argument("--model-name", type=str, help="Name of a current model.")
    parser.add_argument("--modalities", type=str, help="Name of a current model.")
    parser.add_argument(
        "--log-every-n-steps",
        default=2,
        type=int,
        help="How often do logging during training.",
    )
    parser.add_argument(
        "--max-epochs", default=10, type=int, help="Number of training epochs."
    )
    parser.add_argument(
        "--early-stopping", default=False, type=bool, help="If to use EarlyStopping."
    )
    parser.add_argument(  # prevoisly min_delta
        "--early-stopping-delta",
        default=1e-2,
        type=float,
        help="Delta used in EarlyStopping.",
    )
    parser.add_argument(
        "--patience", default=2, type=int, help="Patience for EarlyStopping."
    )
    parser.add_argument(
        "--kld-weight",
        default=1,
        type=float,
        help="Weight for the KL divergence term in loss..",
    )
    parser.add_argument(
        "--batch-norm", default=False, type=bool, help="If to use BatchNorm in network."
    )
    parser.add_argument("--hidden-dim", default=12, type=int, help="TODO.")  # TO CHANGE
    parser.add_argument("--dim", default=12, type=int, help="TODO.")  # TO CHANGE
    parser.add_argument("--embedding-dim", type=int, help="TODO.")
    parser.add_argument("--laatent-dim", default=12, type=int, help="TODO.")
    parser.add_argument("--latent-hidden-dim", default=12, type=int, help="TODO.")

    # modality name?

    # Babel

    # Omivae

    # Transformer
