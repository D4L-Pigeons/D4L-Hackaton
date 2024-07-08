def main():

    cross_validation_metrics.to_json(results_path / "metrics.json", indent=4)
    print(f"Metrics saved to: {results_path / 'metrics.json'}")


if __name__ == "__main__":
    main()
