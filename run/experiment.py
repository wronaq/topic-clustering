import argparse
import importlib
import json
import torch
import logging

logging.basicConfig(level=logging.INFO)


def run_experiment(experiment_config, n_topics, n_words, save_embeddings=True):

    try:
        embeddings = torch.load("sentence_embeddings.pt")
    except:
        logging.info("Creating sentence embeddings")
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
        max_seq_length_ = experiment_config.get("model_args", {}).get(
            "max_seq_length", {}
        )
        model = model_class_(**model_args_)
        model.max_seq_length = max_seq_length_
        embeddings = model.encode(
            dataset.load_data()[:32], show_progress_bar=True, **encode_args_
        )

        if save_embeddings:
            torch.save(embeddings, "sentence_embeddings.pt")
    else:
        logging.info("Sentence embeddings loaded from file")

    # dimensionality reduction
    logging.info("Dimensionality reduction")
    umap_module_ = importlib.import_module("umap")
    umap_class_ = getattr(umap_module_, "UMAP")
    umap_args_ = experiment_config.get("umap_args", {})
    umap = umap_class_(**umap_args_)
    low_dim_embeddings = umap.fit_transform(embeddings)

    # clustering
    logging.info("Clustering")
    hdbscan_module_ = importlib.import_module("hdbscan")
    hdbscan_class_ = getattr(hdbscan_module_, "HDBSCAN")
    hdbscan_args_ = experiment_config.get("hdbscan_args", {})
    hdbscan = hdbscan_class_(**hdbscan_args_)
    clusters = hdbscan.fit(low_dim_embeddings)

    # class-based TF-IDF
    logging.info("Calculate class-based TF-IDF")
    topic_module_ = importlib.import_module("topic_clustering.topics")
    topic_class_ = getattr(topic_module_, "Topic")
    topic_args_ = experiment_config.get("topic_args", {})
    topic = topic_class_(dataset, clusters, n_topics, **topic_args_)
    topic.top_n_words(n_words)


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "experiment_config",
        type=str,
        help='Path to experiment JSON like: \'{"dataloader": "DataLoader", "model_args": {"model_name_or_path": "sentence-transformers/xlm-r-100langs-bert-base-nli-stsb-mean-tokens"}}\'',
    )
    parser.add_argument(
        "n_topics",
        type=int,
        help="Number of topics to find. If zero then no reduction is made.",
    )
    parser.add_argument(
        "n_words",
        type=int,
        help="Top n words that describe each topic.",
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
    run_experiment(experiment_config, args.n_topics, args.n_words, args.save_embeddings)


if __name__ == "__main__":
    main()
