import os
from typing import List, Union

import pandas as pd

from argparse_config import get_args
from pair_encoder import train_encoder
from pair_encoder.evaluation import CorrelationEvaluator, get_correlation
from util import (
    eval_and_submit,
    get_current_timestamp,
    get_data,
    get_log_path_from_args,
    get_pairs,
)

EVALUATION_PHASE = False


def run(args) -> Union[List[float], float]:
    starttime = get_current_timestamp()
    logfile = get_log_path_from_args(args, starttime, evaluation_phase=EVALUATION_PHASE)
    os.makedirs(os.path.dirname(logfile), exist_ok=True)

    train_df = get_data(lang=args.lang, train=True, clean=args.clean)
    if args.lang == "esp":
        # workaround for missing data, use dev set for evaluating
        dev_samples = get_pairs(get_data(lang=args.lang, train=False, clean=args.clean))
    else:
        # otherwise, use test set
        dev_samples = get_pairs(get_data(lang=args.lang, test=True, clean=args.clean))

    train_samples = get_pairs(train_df)
    evaluator = CorrelationEvaluator.load(dev_samples)

    pair_encoder, history = train_encoder(
        train_samples=train_samples,
        evaluator=evaluator,
        model_name=args.model_name,
        similarity_model=args.similarity_model,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        max_grad_norm=args.max_grad_norm,
        epochs=args.epochs,
        eval_steps=args.eval_steps,
        max_length=args.max_length,
        k=args.k,
        weak_training_epochs=args.weak_training_epochs,
        weak_sampling_percentage=args.weak_sampling_percentage,
        seed=args.seed,
        save_to=args.save_to,
        verbose=args.verbose,
        device=args.device,
    )

    scores = eval_and_submit(
        pair_encoder,
        args.lang,
        args.model_name,
        starttime,
        evaluation_phase=EVALUATION_PHASE,
    )
    return (
        get_correlation(test=dev_samples, pair_encoder=pair_encoder),
        scores,
    )


if __name__ == "__main__":
    args = get_args()
    args.device = "cpu"
    args.epochs = 1
    args.learning_rate = 2e-5
    for k in [0, 1, 2, 3]:
        print(k)
        args.k = k
        run(args)
