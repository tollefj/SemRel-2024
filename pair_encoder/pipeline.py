from datasets import load_dataset
from fire import Fire
from scipy.stats import spearmanr

from pair_encoder.evaluation import CorrelationEvaluator
from pair_encoder.train import train_encoder
from pair_encoder.util import PairInput


def create_pair(row):
    return PairInput(pair=(row["sentence1"], row["sentence2"]), label=row["score"])


def process_sts(dataset, _drop_duplicates=True, _normalize=True, return_pairs=True):
    df = dataset.to_pandas()
    df = df[["score", "sentence1", "sentence2"]]
    if _normalize:
        # normalize scores
        df.score = normalize(df.score)
    # remove duplicate sentence1, sentence2 pairs
    if _drop_duplicates:
        df = df.drop_duplicates(subset=["sentence1", "sentence2"])

    if return_pairs:
        df = df.apply(create_pair, axis=1).tolist()
    return df


def run(
    model_name: str,
    similarity_model: str = None,
    batch_size: int = 32,
    learning_rate: float = 2e-5,
    epochs: int = 3,
    eval_steps: int = 0,
    k: int = 0,
    verbose: bool = False,
    save_to: str = "output",
    device: str = "cuda",
    dataset_name: str = "mteb/stsbenchmark-sts",
):
    dataset = load_dataset(dataset_name)
    train = process_sts(dataset["train"])
    test = process_sts(dataset["test"])
    dev = process_sts(dataset["validation"])

    evaluator = CorrelationEvaluator.load(dev)

    encoder = train_encoder(
        train_samples=train,
        evaluator=evaluator,
        model_name=model_name,
        similarity_model=similarity_model,
        batch_size=batch_size,
        learning_rate=learning_rate,
        epochs=epochs,
        eval_steps=eval_steps,
        k=k,
        verbose=verbose,
        save_to=save_to,
        device=device,
    )

    test_data = [(data.pair[0], data.pair[1]) for data in test]
    test_scores = [p.label for p in test]

    preds = encoder.predict(test_data)
    corr = spearmanr(test_scores, preds).correlation
    print(f"Spearman correlation on test set: {corr}")


if __name__ == "__main__":
    Fire(run)
