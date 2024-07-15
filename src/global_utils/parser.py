import argparse
from copy import deepcopy


def combine_with_defaults(config, modalities_draft_config):
    modalities_names = modalities_draft_config.keys()
    defaults = vars(_parse_args(modalities_names, []))
    res = deepcopy(defaults)

    for modalitiy_name, modality_config in modalities_draft_config.items():
        modality_dict = {
            f"{modalitiy_name}_{key}": value for key, value in modality_config.items()
        }
        config = config | modality_dict

    for k, v in config.items():
        assert k in res.keys(), "{} not in default values".format(k)
        res[k] = v
    return res


def _parse_args(modalities_names, args=None):
    parser = _get_parser(modalities_names)
    return parser.parse_args(args=args)


def _get_parser(modalities_names):
    """Parses command line arguments."""
    parser = argparse.ArgumentParser(modalities_namesdescription="Validate model")
    _main_parser(parser)
    _data_parser(parser)
    _train_parser(parser)
    _modality_parser(parser, modalities_names)
    return parser.parse_args()


def _main_parser(parser):  # Check if needs to return parser
    parser.add_argument(
        "--method",
        choices=["omivae", "babel", "advae", "vae"],
        help="Name of the method to use.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["train", "test"],
        default="train",
        help="Mode to run the model.",
    )
    parser.add_argument(
        "--cv-seed", default=42, help="Seed used to make k folds for cross validation."
    )
    parser.add_argument(
        "--n-folds", default=5, type=int, help="Number of folds in cross validation."
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


def _data_parser(parser):
    parser.add_argument(
        "--path-to-dataset",
        help="Path to the dataset.",
    )

    parser.add_argument(
        "--dataset-name",
        choices=["cite-seq"],
        help="Name of the dataset. Used for adding hierarchies and creating correct dataloader.",
    )

    parser.add_argument(
        "--plus-iid-holdout",
        action="store_true",
        help="Whether to include the iid_holdout data in the anndata object.",
    )
    parser.add_argument(
        "--data_normalization",
        type=str,
        choices=["log1p", "standardize", "pearson_residuals", None],
        default="standardize",
        help="Normalization method for the data.",
    )
    parser.add_argument(
        "--remove_batch_effect",
        type=bool,
        default=True,
        help="Whether to remove batch effects.",
    )
    parser.add_argument(
        "--target_hierarchy_level",
        type=int,
        default=-1,
        help="Target hierarchy level for classification.",
    )


def _train_parser(parser):
    parser.add_argument("--lr", default=1e-3, type=float, help="Learning rate.")
    parser.add_argument("--batch-size", default=64, type=int, help="Batch size.")
    parser.add_argument("--model-name", type=str, help="Name of a current model.")
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
        default=0.001,
        type=float,
        help="Delta used in EarlyStopping.",
    )
    parser.add_argument(  # previously patience
        "--early-stopping-patience",
        default=2,
        type=int,
        help="Patience for EarlyStopping.",
    )


def _test_parser(parser):
    parser.add_argument(  # previously patience
        "--model-path", type=str, help="Path of the model."
    )


def _modality_parser(parser, modalities_names):
    for modality_name in modalities_names:
        prefix = f"{modality_name}_"
        parser.add_argument(
            f"--{prefix}modality_name",
            type=str,
            default=modality_name.upper(),
            help=f"Modality name for {modality_name.upper()}.",
        )
        parser.add_argument(
            f"--{prefix}dim",
            type=int,
            default=13953 if modality_name == "gex" else 134,
            help=f"Dimension of the {modality_name.upper()} modality.",
        )
        parser.add_argument(
            f"--{prefix}hidden_dim",
            type=int,
            default=128,
            help=f"Hidden dimension for the {modality_name.upper()} modality.",
        )
        parser.add_argument(
            f"--{prefix}encoder_hidden_dim",
            type=int,
            default=128,
            help=f"Encoder hidden dimension for the {modality_name.upper()} modality.",
        )
        parser.add_argument(
            f"--{prefix}encoder_out_dim",
            type=int,
            default=32,
            help=f"Encoder output dimension for the {modality_name.upper()} modality.",
        )
        parser.add_argument(
            f"--{prefix}latent_dim",
            type=int,
            default=16,
            help=f"Latent dimension for the {modality_name.upper()} modality.",
        )
        parser.add_argument(
            f"--{prefix}decoder_hidden_dim",
            type=int,
            default=128,
            help=f"Decoder hidden dimension for the {modality_name.upper()} modality.",
        )
        parser.add_argument(
            f"--{prefix}batch_norm",
            type=bool,
            default=False,
            help=f"Whether to use batch normalization for the {modality_name.upper()} modality.",
        )
        parser.add_argument(
            f"--{prefix}dropout_rate",
            type=float,
            default=0.1,
            help=f"Dropout rate for the {modality_name.upper()} modality.",
        )
        parser.add_argument(
            f"--{prefix}recon_loss_coef",
            type=float,
            default=1,
            help=f"Reconstruction loss coefficient for the {modality_name.upper()} modality.",
        )
        parser.add_argument(
            f"--{prefix}kld_loss_coef",
            type=float,
            default=0.1,
            help=f"KLD loss coefficient for the {modality_name.upper()} modality.",
        )


# Babel

# Omivae

# Transformer
