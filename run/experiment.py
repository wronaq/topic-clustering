import argparse
import importlib
import json
import torch


def run_experiment(experiment_config, save_embeddings=True):

    # load data
    data_module_ = importlib.import_module("topic_clustering.data_loader")
    dataloader_class_ = getattr(data_module_, experiment_config["dataloader"])
    dataloader_args_ = experiment_config.get("dataloader_args", {})
    dataset = dataloader_class_(**dataloader_args_)

    # generate sentence embeddings
    model_module_ = importlib.import_module("sentence_transformers")
    model_class_ = getattr(model_module_, "SentenceTransformer")
    model_args_ = {
        k: v
        for k, v in experiment_config.get(
            "model_args",
            {
                "model_name_or_path": "sentence-transformers/xlm-r-100langs-bert-base-nli-stsb-mean-tokens"
            },
        ).items()
        if k in ["model_name_or_path", "device"]
    }
    encode_args_ = {
        k: v
        for k, v in experiment_config.get("model_args", {}).items()
        if k not in ["model_name_or_path", "max_seq_length"]
    }
    max_seq_length_ = experiment_config.get("model_args", {}).get("max_seq_length", {})
    model = model_class_(**model_args_)
    model.max_seq_length = max_seq_length_
    embeddings = model.encode(
        dataset.load_data(), show_progress_bar=True, **encode_args_
    )

    if save_embeddings:
        torch.save(embeddings, "sentence_embeddings.pt")


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "experiment_config",
        type=str,
        help='Path to experiment JSON like: \'{"dataloader": "DataLoader", "model_args": {"model_name_or_path": "sentence-transformers/xlm-r-100langs-bert-base-nli-stsb-mean-tokens"}}\'',
    )
    parser.add_argument(
        "--save_embeddings",
        action="store_true",
        help="Whether to save embeddings",
    )
    args = parser.parse_args()
    return args


def main():
    args = _parse_args()

    with open(args.experiment_config, "r") as f:
        config = f.read()
    experiment_config = json.loads(config)
    run_experiment(experiment_config, args.save_embeddings)


if __name__ == "__main__":
    main()
