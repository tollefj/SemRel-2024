from dataclasses import dataclass
from typing import Tuple, Union

import torch
from transformers import (
    PreTrainedModel,
    get_cosine_schedule_with_warmup,
    get_linear_schedule_with_warmup,
)


@dataclass
class PairInput:
    """
    Represents a pair input with a pair of strings and a label.

    Attributes:
        pair (Tuple[str]): The pair of strings.
        label (Union[int, float]): The label associated with the pair.
    """

    pair: Tuple[str]
    label: Union[int, float]


def init_optimizer(
    optimizer: torch.optim.Optimizer,
    model: PreTrainedModel,
    learning_rate: float,
    weight_decay: float,
) -> torch.optim.Optimizer:
    """
    Initializes the optimizer for the given model with the specified learning rate and weight decay.

    Args:
        optimizer (torch.optim.Optimizer): The optimizer to be initialized.
        model (PreTrainedModel): The model for which the optimizer is being initialized.
        learning_rate (float): The learning rate for the optimizer.
        weight_decay (float): The weight decay for the optimizer.

    Returns:
        torch.optim.Optimizer: The initialized optimizer.
    """

    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p for n, p in param_optimizer if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": weight_decay,
        },
        {
            "params": [
                p for n, p in param_optimizer if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]

    return optimizer(
        params=optimizer_grouped_parameters,
        lr=learning_rate,
        weight_decay=weight_decay,
    )


warmups = {
    "linear": get_linear_schedule_with_warmup,
    "cosine": get_cosine_schedule_with_warmup,
}


def get_scheduler(
    warmup_steps: int,
    train_steps: int,
    optimizer=torch.optim.AdamW,
    warmup_type: str = "linear",
) -> torch.optim.lr_scheduler.LambdaLR:
    """
    Returns a scheduler for adjusting the learning rate during training.

    Parameters:
        warmup_steps (int): The number of warmup steps.
        train_steps (int): The total number of training steps.
        optimizer (torch.optim.Optimizer): The optimizer used for training. Default is torch.optim.AdamW.
        cosine (bool): Whether to use a cosine scheduler. Default is False.

    Returns:
        torch.optim.lr_scheduler.LambdaLR: The scheduler for adjusting the learning rate during training.
    """
    warmup_fn = warmups[warmup_type]
    return warmup_fn(
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=train_steps,
    )

def cos_sim(a: torch.Tensor, b: torch.Tensor):
    """
    Computes the cosine similarity cos_sim(a[i], b[j]) for all i and j.
    :return: Matrix with res[i][j]  = cos_sim(a[i], b[j])
    """
    if not isinstance(a, torch.Tensor):
        a = torch.tensor(a)

    if not isinstance(b, torch.Tensor):
        b = torch.tensor(b)

    if len(a.shape) == 1:
        a = a.unsqueeze(0)

    if len(b.shape) == 1:
        b = b.unsqueeze(0)

    a_norm = torch.nn.functional.normalize(a, p=2, dim=1)
    b_norm = torch.nn.functional.normalize(b, p=2, dim=1)
    return torch.mm(a_norm, b_norm.transpose(0, 1))