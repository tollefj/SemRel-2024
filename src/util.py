import json
import os
import re
import zipfile
from argparse import Namespace
from datetime import datetime
from typing import List, Tuple

import numpy as np
import torch
from pandas import DataFrame, concat, read_csv
from scipy import stats
from sentence_transformers import InputExample, SentenceTransformer
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from sklearn.metrics.pairwise import paired_cosine_distances
from sklearn.model_selection import train_test_split
from torch import Tensor

from pair_encoder import PairEncoder
from pair_encoder.util import PairInput

PATH = "../Semantic_Relatedness_SemEval2024/Track A"

palette = [
    "#e60049",
    "#0bb4ff",
    "#50e991",
    "#9b19f5",
    "#ffa300",
    "#dc0ab4",
    "#b3d4ff",
    "#00bfa0",
]


def get_model(model_id: str = "intfloat/multilingual-e5-", size: str = "base"):
    model = SentenceTransformer(f"{model_id}{size}")
    return model


def get_langs():
    return [l for l in sorted(os.listdir(PATH)) if len(l) == 3]


def get_all_langs(
    train: bool = True,
    test: bool = False,
    prefix: str = "",
    verbose: bool = False,
    language_col: bool = False,
) -> DataFrame:
    langs = get_langs()
    data = []
    for lang in langs:
        if verbose:
            print(f"Processing {lang}...")
        tmp = get_data(lang=lang, train=train, test=test, prefix=prefix)
        if language_col:
            tmp["language"] = lang
        data.append(tmp)
    df = concat(data)
    return df


def get_data_from_path(path: str, prefix: str = "") -> DataFrame:
    df = read_csv(path)
    # check if there's a "\n" in the first line, otherwise the split should be on \t
    sep = "\n"
    if "\\n" in df["Text"].iloc[0]:
        sep = "\\n"
    if "\t" in df["Text"].iloc[0]:
        sep = "\t"
    split_text = [sent.split(sep) for sent in df.Text.tolist()]
    df["s1"] = [sent[0] for sent in split_text]
    df["s2"] = [sent[1] for sent in split_text]

    if len(prefix) > 0:
        add_prefix = lambda x: f"{prefix} {x}"
        df = df.assign(s1=df.s1.apply(add_prefix))
        df = df.assign(s2=df.s2.apply(add_prefix))

    df = df.drop(columns=["Text"])
    return df


def get_data(
    lang: str = "eng",
    train: bool = True,
    test=False,
    prefix: str = "",
) -> DataFrame:
    if lang == "all":
        return get_all_langs(train, prefix)
    if not train and not test:
        split = "dev_with_labels"
    elif test:
        # workaround for lacking Spanish test :(
        if lang == "esp":
            split = "test"
        else:
            split = "test_with_labels"
    else:
        split = "train"
    path = f"{PATH}/{lang}/{lang}_{split}.csv"
    return get_data_from_path(path, prefix)


def get_and_split(
    lang: str = "eng",
    seed: int = 42,
    test_size: float = 0.2,
    prefix: str = "",
) -> Tuple[DataFrame, DataFrame]:
    if lang == "all":
        print("Getting all languages...")
        df = get_all_langs(train=True, prefix=prefix)
    else:
        df = get_data(lang, prefix=prefix)

    train, test = train_test_split(df, test_size=test_size, random_state=seed)
    return train, test


def get_all_splits(test_size=0.0, prefix=""):
    dfs = {}

    for lang in get_langs():
        train, test = None, None
        if test_size > 0.0:
            train, test = get_and_split(lang=lang, prefix=prefix)
        else:
            train = get_data(lang=lang, train=True, prefix=prefix)

        dfs[lang] = {
            "train": train,
            "test": test,
            "dev": get_data(lang=lang, train=False, prefix=prefix),
        }
    return dfs


def get_pairs(df) -> List[PairInput]:
    examples = []
    s1 = df.s1.tolist()
    s2 = df.s2.tolist()

    if "Score" not in df.columns:
        score = [-1] * len(s1)
    else:
        score = df.Score.tolist()

    # for _s1, _s2, _score in zip(df.s1, df.s2, df.Score):
    for _s1, _s2, _score in zip(s1, s2, score):
        examples.append(PairInput(pair=(_s1, _s2), label=_score))

    return examples


def get_examples(df) -> List[InputExample]:
    # this assumes a cosine-similarity loss
    examples = []
    for _s1, _s2, _score in zip(df.s1, df.s2, df.Score):
        examples.append(InputExample(texts=[_s1, _s2], label=_score))

    return examples


def get_evaluator(test_df: DataFrame) -> EmbeddingSimilarityEvaluator:
    evaluator = EmbeddingSimilarityEvaluator(
        test_df.s1.tolist(), test_df.s2.tolist(), test_df.Score.tolist()
    )
    return evaluator


def embed(model, df, batch_size=16):
    e1 = model.encode(df.s1.tolist(), batch_size=batch_size, convert_to_numpy=True)
    e2 = model.encode(df.s2.tolist(), batch_size=batch_size, convert_to_numpy=True)
    return e1, e2


def get_cosine_dist(e1, e2):
    return 1 - paired_cosine_distances(e1, e2)


def get_spearman(gold_scores, pred_scores):
    spearmanr = stats.spearmanr(gold_scores, pred_scores)
    return spearmanr.correlation


def get_current_timestamp() -> str:
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def make_submission(
    df,
    preds,
    lang,
    timestamp=None,
    info=None,
    model_name=None,
    submissions_folder="submissions",
):
    if not timestamp:
        timestamp = get_current_timestamp()

    submission = DataFrame({"PairID": df["PairID"], "Pred_Score": preds})
    csv_name = f"pred_{lang}_a.csv"
    # if lang == "esp":
    #     csv_name = f"pred_dev_a.csv"  # workaround for an error on codalab

    submission_file = os.path.join(submissions_folder, timestamp, csv_name)
    print(f"Saving submission to {submission_file}")
    os.makedirs(os.path.dirname(submission_file), exist_ok=True)
    submission.to_csv(submission_file, index=False)

    zip_name = os.path.join(submissions_folder, timestamp)
    if model_name:
        zip_name = os.path.join(
            submissions_folder, model_name.replace("/", "-"), timestamp
        )
    os.makedirs(zip_name, exist_ok=True)

    zipfile_path = os.path.join(zip_name, f"pred_{lang}_a.zip")
    with zipfile.ZipFile(zipfile_path, "w") as zf:
        zf.write(submission_file, arcname=csv_name)
    os.remove(submission_file)
    os.rmdir(os.path.dirname(submission_file))


def eval_and_submit(
    pair_encoder: PairEncoder,
    lang: str,
    model_name: str,
    timestamp: str,
    evaluation_phase: bool = False,
) -> List[float]:
    if evaluation_phase:
        dev_df = get_data(lang=lang, test=True)
    else:
        dev_df = get_data(lang=lang, train=False)
    dev_data = []
    for s1, s2 in zip(dev_df.s1, dev_df.s2):
        dev_data.append((s1, s2))
    preds = pair_encoder.predict(dev_data)
    submissions_folder = "submissions" if evaluation_phase else "submissions_dev"
    make_submission(
        dev_df,
        preds,
        lang,
        model_name=model_name,
        timestamp=timestamp,
        submissions_folder=submissions_folder,
    )
    return preds


def do_evaluation(
    model: SentenceTransformer,
    lang: str,
    df: DataFrame,
    timestamp=None,  # a timestamp defines the folder and unique id
    submit=True,
    info=None,
    model_name=None,
) -> np.ndarray:
    if not model:
        raise ValueError("model must be specified")

    if submit and not timestamp:
        timestamp = get_current_timestamp()

    e1, e2 = embed(model=model, df=df)
    preds = get_cosine_dist(e1, e2)

    if submit:
        make_submission(df, preds, lang, timestamp, info, model_name)

    return preds


def evaluate_local_model_on_holdout(
    model_path: str, lang: str, return_preds=False, test_size: float = 0.2
):
    model = SentenceTransformer(model_path)
    _, test = get_and_split(lang=lang, test_size=test_size)
    preds = do_evaluation(model, lang, test, submit=False)
    if return_preds:
        return preds
    return get_spearman(test["Score"], preds)


def get_log_path(
    model_name: str,
    lang: str,
    starttime: str,
    k: int,
    learning_rate: float,
    epochs: int,
    evaluation_phase: bool = False,
) -> str:
    model_id = model_name.replace("/", "-")
    main_folder = "logs" if evaluation_phase else "logs_dev"
    logfile = os.path.join(
        main_folder,
        lang,
        model_id,
        f"{starttime}--k={k}-lr={learning_rate}-epochs={epochs}.txt",
    )
    return logfile


def get_log_path_from_args(
    args: Namespace, starttime: str, evaluation_phase: bool = False
) -> str:
    return get_log_path(
        model_name=args.model_name,
        lang=args.lang,
        starttime=starttime,
        k=args.k,
        learning_rate=args.learning_rate,
        epochs=args.epochs,
        evaluation_phase=evaluation_phase,
    )
