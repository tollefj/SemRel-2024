import logging
import random
from typing import Dict, List, Tuple, Union

import numpy as np
import torch
from pair_encoder.evaluation import CorrelationEvaluator
from pair_encoder.util import get_scheduler, init_optimizer
from torch import nn
from torch.utils.data import DataLoader
from tqdm.autonotebook import tqdm, trange
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)

logger = logging.getLogger(__name__)


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


class PairEncoder:
    def __init__(
        self,
        model_name: str,
        max_length: int = 512,
        device: str = None,
        seed: int = None,
    ):
        self.config = AutoConfig.from_pretrained(model_name)
        self.config.num_labels = 1
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name, config=self.config
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.max_length = max_length

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = torch.device(device)

        if isinstance(seed, int):
            set_seed(seed)

    def fit(
        self,
        dataloader: DataLoader,
        evaluator: CorrelationEvaluator = None,
        default_optimizer=torch.optim.AdamW,
        epochs: int = 1,
        learning_rate: float = 2e-5,
        warmup: str = "linear",
        warmup_steps: int = 10000,
        weight_decay: float = 0.01,  # default, AdamW
        evaluation_steps: int = 0,
        max_grad_norm: float = 1,
        verbose: bool = True,
        output_path: str = None,
    ) -> None:
        self.model.to(self.device)
        dataloader.collate_fn = lambda x: self.batching(x)

        optimizer = init_optimizer(
            optimizer=default_optimizer,
            model=self.model,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
        )
        scheduler = get_scheduler(
            warmup_steps=warmup_steps,
            train_steps=int(len(dataloader) * epochs),
            optimizer=optimizer,
            warmup_type=warmup,
        )

        best_score = -1

        for epoch in trange(epochs, desc="Training...", disable=not verbose):
            _steps = 0
            self.model.zero_grad()
            self.model.train()

            for features, labels in tqdm(
                dataloader,
                desc="Epoch {}".format(epoch + 1),
                smoothing=0.05,
                disable=not verbose,
            ):
                loss = nn.BCEWithLogitsLoss()(
                    self.model(**features, return_dict=True).logits.view(-1),
                    labels,
                )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
                _steps += 1

                if (
                    evaluator is not None
                    and evaluation_steps > 0
                    and _steps % evaluation_steps == 0
                ):
                    score = evaluator(model=self, epoch=epoch, steps=_steps)
                    self.model.zero_grad()
                    self.model.train()

                    if score > best_score:
                        logger.info(
                            f"New best score: {score} for epoch {epoch} (step {_steps})"
                        )
                        best_score = score
                        if output_path is not None:
                            logger.info(f"Saving model to {output_path}")
                            self.save(output_path)

            if evaluator is not None:
                _ = evaluator(model=self, epoch=epoch, steps=-1)

        if output_path is not None:
            logger.info(f"Saving final model to {output_path}")
            self.save(output_path)

    def batching(
        self, batch, train=True
    ) -> Union[Dict[str, torch.Tensor], Tuple[Dict[str, torch.Tensor], torch.Tensor]]:
        pairs = [[] for _ in range(len(batch[0].pair if train else batch[0]))]
        labels = []

        for i in batch:
            for idx, text in enumerate(i.pair if train else i):
                pairs[idx].append(text.strip())
            if train:
                labels.append(i.label)

        tokenized = self.tokenizer(
            *pairs,
            padding=True,
            truncation="longest_first",
            return_tensors="pt",
            max_length=self.max_length,
        )

        if train:
            labels = torch.tensor(labels, dtype=torch.float).to(self.device)

        for name in tokenized:
            tokenized[name] = tokenized[name].to(self.device)

        return (tokenized, labels) if train else tokenized

    def predict(self, sentences: List[List[str]], batch_size=32) -> np.ndarray:
        if isinstance(sentences[0], str):
            raise ValueError(
                "Input must be a list of lists of strings (i.e. a list of sentence pairs)"
            )

        inp_dataloader = DataLoader(
            sentences,
            batch_size=batch_size,
            collate_fn=lambda x: self.batching(x, train=False),
            shuffle=False,
        )

        preds = []
        self.model.eval()
        self.model.to(self.device)
        with torch.no_grad():
            for features in inp_dataloader:
                preds.extend(
                    nn.Sigmoid()(self.model(**features, return_dict=True).logits)
                )
        return np.asarray([score[0].cpu().detach().numpy() for score in preds])

    def save(self, path: str) -> None:
        if path is None:
            return
        logger.info("Save model to {}".format(path))
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
