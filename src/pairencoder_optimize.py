import logging
import os

import optuna

from argparse_config import get_args
from models import models
from pairencoder_train import run
from util import get_current_timestamp, get_langs


def objective_large_explore(trial, args):
    lr_from = 1e-06
    lr_to = 1e-4
    args.learning_rate = trial.suggest_float("learning_rate", lr_from, lr_to, log=True)
    args.k = trial.suggest_int("k", 0, 3)
    args.epochs = trial.suggest_int("epochs", 1, 5)
    args.weak_training_epochs = trial.suggest_int("weak_training", 0, 2)
    args.max_grad_norm = trial.suggest_float("max_grad_norm", 0.1, 1.0)
    args.weak_sampling_percentage = 100
    args.batch_size = 32
    corr, _ = run(args)
    return corr


if __name__ == "__main__":
    logger = optuna.logging.get_logger("optuna")
    logger.propagate = False
    logger.setLevel(logging.INFO)
    print(f"Starting optimization...")
    for lang in get_langs():
        print(lang)
        folder = f"logs_optuna/{lang}"
        print(f"Exists already? {os.path.exists(folder)}")
        if os.path.exists(folder):
            continue
        os.makedirs(folder, exist_ok=True)
        logger.addHandler(
            logging.FileHandler(f"{folder}/{get_current_timestamp()}.txt")
        )
        logger.info(f"Optimizing for {lang}...")
        args = get_args()
        args.model_name = models[lang]
        args.similarity_model = "intfloat/multilingual-e5-base"
        args.lang = lang
        study = optuna.create_study(direction="maximize")
        obj = lambda trial: objective_large_explore(trial, args=args)
        study.optimize(obj, n_trials=20)
