from models.babel import BabelModel
from models.ModelBase import ModelBase
from models.omivae import OmiModel
from models.vae import VAE
from utils.data_utils import load_anndata
from utils.train_and_validate_utils import (
    parse_arguments,
    load_config,
    create_results_dir,
    save_model_and_latent,
    CrossValidator,
)
from utils.neptune_utils import get_metrics_callback


def create_model(args, config) -> ModelBase:
    if args.method == "omivae":
        model = OmiModel(config)
        return model
    elif args.method == "babel":
        model = BabelModel(config)
        return model
    elif args.method == "advae":
        raise NotImplementedError(f"{args.method} method not implemented.")
    elif args.method == "vae":
        return VAE(config)
    else:
        raise NotImplementedError(f"{args.method} method not implemented.")


def train(config):
    metrics_callback = get_metrics_callback(config)

    config = load_config(args)
    model = create_model(args, config)

    train_data = load_anndata(
        mode="train",
        normalize=config.normalize,
        remove_batch_effect=config.remove_batch_effect,
        target_hierarchy_level=config.target_hierarchy_level,
        preload_subsample_frac=args.preload_subsample_frac,
    )
    val_data = load_anndata(
        mode="test",
        normalize=config.normalize,
        remove_batch_effect=config.remove_batch_effect,
        target_hierarchy_level=config.target_hierarchy_level,
        preload_subsample_frac=args.preload_subsample_frac,
    )

    model.fit(train_data)

    cross_validator = CrossValidator(
        model, args.cv_seed, args.n_folds, args.subsample_frac
    )

    cross_validation_metrics = cross_validator.perform_cross_validation(
        train_data, val_data
    )

    latent_representation = model.predict(train_data)

    results_path = create_results_dir(args)
    # compute_latent_and_save_model(model, latent_representation, results_path)
    # compute_and_save_metrics(model)


if __name__ == "__main__":
    train()
