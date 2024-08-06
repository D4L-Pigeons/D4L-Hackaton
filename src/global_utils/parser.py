import argparse
from copy import deepcopy

# from typing import Any, Callable


def combine_with_defaults(config):
    res = deepcopy(vars(parse_args([])))
    for k, v in config.items():
        assert k in res.keys(), "{} not in default values".format(k)
        res[k] = v
    return res


def get_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--method",
        choices=["omivae", "babel", "advae", "vae"],
        help="Name of the method to use.",
    )
    parser.add_argument("--model-name", type=str, help="Name of a current model.")
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
    parser.add_argument(
        "--neptune-logger", default=False, help="If to log experiment to neptune."
    )

    parser.add_argument(
        "--path-to-dataset",
        default="",
        help="Path to the dataset. If you want to run locally (with path repo/data/raw/dataset_name) leave it default.",
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
        choices=[-1, -2],
        help="Target hierarchy level for classification.",
    )

    parser.add_argument("--lr", default=1e-3, type=float, help="Learning rate.")
    parser.add_argument("--batch-size", default=64, type=int, help="Batch size.")
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
    parser.add_argument(
        f"--hidden_dim",
        type=int,
        default=128,
        help=f"Hidden dimension .",
    )
    parser.add_argument(
        f"--encoder_hidden_dim",
        type=int,
        default=128,
        help=f"Encoder hidden dimension.",
    )
    parser.add_argument(
        f"--encoder_out_dim",
        type=int,
        default=32,
        help=f"Encoder output dimension.",
    )
    parser.add_argument(
        f"--latent_dim",
        type=int,
        default=16,
        help=f"Latent dimension.",
    )
    parser.add_argument(
        f"--decoder_hidden_dim",
        type=int,
        default=128,
        help=f"Decoder hidden dimension.",
    )
    parser.add_argument(
        f"--batch_norm",
        type=bool,
        default=False,
        help=f"Whether to use batch normalization.",
    )
    parser.add_argument(
        f"--dropout_rate",
        type=float,
        default=0.1,
        help=f"Dropout rate.",
    )
    parser.add_argument(
        f"--recon_loss_coef",
        type=float,
        default=1,
        help=f"Reconstruction loss coefficient.",
    )
    parser.add_argument(
        f"--kld_loss_coef",
        type=float,
        default=0.1,
        help=f"KLD loss coefficient.",
    )

    for modality_name in ["gex", "adt"]:
        prefix = f"{modality_name}_"
        parser.add_argument(
            f"--{prefix}_modality_name",
            type=str,
            default=modality_name.upper(),
            help=f"Modality name for {modality_name.upper()}.",
        )
        parser.add_argument(
            f"--{prefix}_dim",
            type=int,
            default=13953 if modality_name == "gex" else 134,
            help=f"Dimension of the {modality_name.upper()} modality.",
        )

    return parser


def parse_args(args=None):
    parser = get_parser()
    return parser.parse_args(args=args)


# # def combine_with_defaults(config, modalities_draft_config):
# #     modalities_names = list(modalities_draft_config.keys())
# #     defaults = vars(_parse_args(modalities_names, []))
# #     res = deepcopy(defaults)

# #     print(modalities_names)

# #     for modalitiy_name, modality_config in modalities_draft_config.items():
# #         modality_dict = {
# #             f"{modalitiy_name}_{key}": value for key, value in modality_config.items()
# #         }
# #         config   = config | modality_dict

# #     for k, v in config.items():
# #         assert k in res.keys(), "{} not in default values".format(k)
# #         res[k] = v

# #     res["modalities_names"] = modalities_names
# #     return res


# def parse_args(args=None):
#     parser = _get_parser()
#     print(parser)
#     return parser.parse_args()


# def _get_parser():
#     """Parses command line arguments."""
#     parser = argparse.ArgumentParser()
#     # _main_parser(parser)
#     # _data_parser(parser)
#     # _train_parser(parser)
#     # _modality_parser(parser, modalities_names)
#     return parser


# def _main_parser(parser):  # Check if needs to return parser
#     # parser.add_argument(
#     #     "--ex", type=str, default="/exp", help="Path to the experiment."
#     # )
#     parser.add_argument(
#         "--method",
#         choices=["omivae", "babel", "advae", "vae"],
#         help="Name of the method to use.",
#     )
#     parser.add_argument(
#         "--mode",
#         type=str,
#         choices=["train", "test"],
#         default="train",
#         help="Mode to run the model.",
#     )
#     parser.add_argument(
#         "--cv-seed", default=42, help="Seed used to make k folds for cross validation."
#     )
#     parser.add_argument(
#         "--n-folds", default=5, type=int, help="Number of folds in cross validation."
#     )
#     parser.add_argument(
#         "--preload-subsample-frac",
#         default=None,
#         type=float,
#         help="Fraction of the data to load. If None, use all data. Don't use subsample-frac with this option.",
#     )
#     parser.add_argument(
#         "--subsample-frac",
#         default=None,
#         type=float,
#         help="Fraction of the data to use for cross validation. If None, use all data.",
#     )
#     parser.add_argument(
#         "--retrain", default=True, help="Retrain a model using the whole dataset."
#     )
#     parser.add_argument(
#         "--neptune-logger", default=False, help="If to log experiment to neptune."
#     )


# def _data_parser(parser):
#     parser.add_argument(
#         "--path-to-dataset",
#         default="",
#         help="Path to the dataset. If you want to run locally (with path repo/data/raw/dataset_name) leave it default.",
#     )

#     parser.add_argument(
#         "--dataset-name",
#         choices=["cite-seq"],
#         help="Name of the dataset. Used for adding hierarchies and creating correct dataloader.",
#     )

#     parser.add_argument(
#         "--plus-iid-holdout",
#         action="store_true",
#         help="Whether to include the iid_holdout data in the anndata object.",
#     )
#     parser.add_argument(
#         "--data_normalization",
#         type=str,
#         choices=["log1p", "standardize", "pearson_residuals", None],
#         default="standardize",
#         help="Normalization method for the data.",
#     )
#     parser.add_argument(
#         "--remove_batch_effect",
#         type=bool,
#         default=True,
#         help="Whether to remove batch effects.",
#     )
#     parser.add_argument(
#         "--target_hierarchy_level",
#         type=int,
#         default=-1,
#         choices=[-1, -2],
#         help="Target hierarchy level for classification.",
#     )


# def _train_parser(parser):
#     parser.add_argument("--lr", default=1e-3, type=float, help="Learning rate.")
#     parser.add_argument("--batch-size", default=64, type=int, help="Batch size.")
#     parser.add_argument("--model-name", type=str, help="Name of a current model.")
#     parser.add_argument(
#         "--log-every-n-steps",
#         default=2,
#         type=int,
#         help="How often do logging during training.",
#     )
#     parser.add_argument(
#         "--max-epochs", default=10, type=int, help="Number of training epochs."
#     )
#     parser.add_argument(
#         "--early-stopping", default=False, type=bool, help="If to use EarlyStopping."
#     )
#     parser.add_argument(  # prevoisly min_delta
#         "--early-stopping-delta",
#         default=0.001,
#         type=float,
#         help="Delta used in EarlyStopping.",
#     )
#     parser.add_argument(  # previously patience
#         "--early-stopping-patience",
#         default=2,
#         type=int,
#         help="Patience for EarlyStopping.",
#     )
#     parser.add_argument(
#         f"--hidden_dim",
#         type=int,
#         default=128,
#         help=f"Hidden dimension .",
#     )
#     parser.add_argument(
#         f"--encoder_hidden_dim",
#         type=int,
#         default=128,
#         help=f"Encoder hidden dimension.",
#     )
#     parser.add_argument(
#         f"--encoder_out_dim",
#         type=int,
#         default=32,
#         help=f"Encoder output dimension.",
#     )
#     parser.add_argument(
#         f"--latent_dim",
#         type=int,
#         default=16,
#         help=f"Latent dimension.",
#     )
#     parser.add_argument(
#         f"--decoder_hidden_dim",
#         type=int,
#         default=128,
#         help=f"Decoder hidden dimension.",
#     )
#     parser.add_argument(
#         f"--batch_norm",
#         type=bool,
#         default=False,
#         help=f"Whether to use batch normalization.",
#     )
#     parser.add_argument(
#         f"--dropout_rate",
#         type=float,
#         default=0.1,
#         help=f"Dropout rate.",
#     )
#     parser.add_argument(
#         f"--recon_loss_coef",
#         type=float,
#         default=1,
#         help=f"Reconstruction loss coefficient.",
#     )
#     parser.add_argument(
#         f"--kld_loss_coef",
#         type=float,
#         default=0.1,
#         help=f"KLD loss coefficient.",
#     )


#     for modality_name in ["gex", "adt"]:
#         prefix = f"{modality_name}_"
#         parser.add_argument(
#             f"--{prefix}_modality_name",
#             type=str,
#             default=modality_name.upper(),
#             help=f"Modality name for {modality_name.upper()}.",
#         )
#         parser.add_argument(
#             f"--{prefix}_dim",
#             type=int,
#             default=13953 if modality_name == "gex" else 134,
#             help=f"Dimension of the {modality_name.upper()} modality.",
#         )


# def _test_parser(parser):
#     parser.add_argument(  # previously patience
#         "--model-path", type=str, help="Path of the model."
#     )


# def _modality_parser(parser, modalities_names):
#     for modality_name in modalities_names:
#         prefix = f"{modality_name}_"
#         parser.add_argument(
#             f"--{prefix}modality_name",
#             type=str,
#             default=modality_name.upper(),
#             help=f"Modality name for {modality_name.upper()}.",
#         )
#         parser.add_argument(
#             f"--{prefix}dim",
#             type=int,
#             default=13953 if modality_name == "gex" else 134,
#             help=f"Dimension of the {modality_name.upper()} modality.",
#         )
#         # parser.add_argument(
#         #     f"--{prefix}hidden_dim",
#         #     type=int,
#         #     default=128,
#         #     help=f"Hidden dimension for the {modality_name.upper()} modality.",
#         # )
#         # parser.add_argument(
#         #     f"--{prefix}encoder_hidden_dim",
#         #     type=int,
#         #     default=128,
#         #     help=f"Encoder hidden dimension for the {modality_name.upper()} modality.",
#         # )
#         # parser.add_argument(
#         #     f"--{prefix}encoder_out_dim",
#         #     type=int,
#         #     default=32,
#         #     help=f"Encoder output dimension for the {modality_name.upper()} modality.",
#         # )
#         # parser.add_argument(
#         #     f"--{prefix}latent_dim",
#         #     type=int,
#         #     default=16,
#         #     help=f"Latent dimension for the {modality_name.upper()} modality.",
#         # )
#         # parser.add_argument(
#         #     f"--{prefix}decoder_hidden_dim",
#         #     type=int,
#         #     default=128,
#         #     help=f"Decoder hidden dimension for the {modality_name.upper()} modality.",
#         # )
#         # parser.add_argument(
#         #     f"--{prefix}batch_norm",
#         #     type=bool,
#         #     default=False,
#         #     help=f"Whether to use batch normalization for the {modality_name.upper()} modality.",
#         # )
#         # parser.add_argument(
#         #     f"--{prefix}dropout_rate",
#         #     type=float,
#         #     default=0.1,
#         #     help=f"Dropout rate for the {modality_name.upper()} modality.",
#         # )
#         # parser.add_argument(
#         #     f"--{prefix}recon_loss_coef",
#         #     type=float,
#         #     default=1,
#         #     help=f"Reconstruction loss coefficient for the {modality_name.upper()} modality.",
#         # )
#         # parser.add_argument(
#         #     f"--{prefix}kld_loss_coef",
#         #     type=float,
#         #     default=0.1,
#         #     help=f"KLD loss coefficient for the {modality_name.upper()} modality.",
#         # )


# # Babel

# # Transformer


# def apply_to_all_modalities(config: dict, field_name: str, func: Callable) -> Any:
#     """Applies a function to values from all modalities in the config.

#     Args:
#         config: A dictionary containing configuration data.
#         field_name: The name of the field to extract from each modality.
#         func: The function to apply to the extracted values.

#     Returns:
#         The result of applying the function to the list of extracted values.

#     Raises:
#         TypeError: If the function cannot be applied to the extracted data types.
#     """
#     config_dict = vars(config)
#     values = [
#         config_dict[f"{modality_name}_{field_name}"]
#         for modality_name in config.modalities_names
#     ]
#     try:
#         return func(values)
#     except TypeError:
#         # Handle the case where func cannot handle the data types
#         raise TypeError(
#             f"Function '{func.__name__}' cannot be applied to the extracted data types."
#         )
