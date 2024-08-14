from src.data.dataloader_todo import load_anndata
from src.global_utils.train_and_validate_utils import (
    CrossValidator,
    create_model,
    create_results_dir,
    save_model_and_latent,
)
from src.utils.old.data_utils import load_anndata
from src.utils.paths import CONFIG_PATH, RESULTS_PATH


def train(config):
    model = create_model(config)

    train_data = load_anndata(
        mode="train",
        normalize=config.data_normalization,
        remove_batch_effect=config.remove_batch_effect,
        target_hierarchy_level=config.target_hierarchy_level,
        preload_subsample_frac=config.preload_subsample_frac,
    )
    val_data = load_anndata(
        mode="test",
        normalize=config.data_normalization,
        remove_batch_effect=config.remove_batch_effect,
        target_hierarchy_level=config.target_hierarchy_level,
        preload_subsample_frac=config.preload_subsample_frac,
    )

    model.fit(train_data)

    cross_validator = CrossValidator(
        model, config.cv_seed, config.n_folds, config.subsample_frac
    )

    cross_validation_metrics = cross_validator.cross_validate(train_data, val_data)

    latent_representation = model.predict(train_data)

    results_path = create_results_dir(config)
    # compute_latent_and_save_model(model, latent_representation, results_path)
    # compute_and_save_metrics(model)


if __name__ == "__main__":
    train()
