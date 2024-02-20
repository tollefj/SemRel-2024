import datasets
import numpy as np
import pandas as pd

sources = {
    "stsb": "mteb/stsbenchmark-sts",
    "sickr": "mteb/sickr-sts",
}


def normalize(scores):
    if isinstance(scores, list):
        scores = np.array(scores)
    scores = scores / 5.0
    scores = np.round(scores, 3)
    return scores


def process_sts(dataset):
    df = dataset.to_pandas()
    df = df[["score", "sentence1", "sentence2"]]
    df.score = normalize(df.score)
    df = df.rename(
        columns={"score": "Score", "sentence1": "s1", "sentence2": "s2"}
    )
    return df


if __name__ == "__main__":
    combined = []
    for dataset_name in "sickr stsb".split():
        dataset = datasets.load_dataset(sources[dataset_name])
        splits = list(dataset.keys())
        if len(splits) > 1:
            for split in splits:
                split_df = process_sts(dataset[split])
                split_df.to_csv(f"{dataset_name}_{split}.csv", index=False)
        dataset = datasets.concatenate_datasets([dataset[s] for s in splits])
        df = process_sts(dataset)
        df["source"] = dataset
        df.to_csv(f"{dataset_name}.csv", index=False)
        combined.append(df)
    combined_df = pd.concat(combined)
    combined_df.to_csv("all_sts.csv", index=False)
