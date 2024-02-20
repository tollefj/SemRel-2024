import os
from math import ceil

from sentence_transformers import SentenceTransformer, models
from sentence_transformers.evaluation import (
    EmbeddingSimilarityEvaluator,
    SimilarityFunction,
)
from sentence_transformers.losses import CosineSimilarityLoss
from torch import nn
from torch.utils.data import DataLoader

from util import get_data, get_examples


def train_callback(score, epoch, steps):
    print(f"Epoch: {epoch} | Steps: {steps} | Score: {score}")


def evaluator_from_df(df, name="eval"):
    data = get_examples(df)
    return EmbeddingSimilarityEvaluator.from_input_examples(
        data,
        name=name,
        main_similarity=SimilarityFunction.COSINE,
    )


def train_from_base_model(model_name, train_samples, config, lang="eng", batch_size=32):
    word_embedding_model = models.Transformer(model_name, max_seq_length=256)
    pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
    dense_model = models.Dense(
        in_features=pooling_model.get_sentence_embedding_dimension(),
        out_features=256,
        activation_function=nn.Tanh(),
    )
    model = SentenceTransformer(
        modules=[word_embedding_model, pooling_model, dense_model],
        device="cuda:1",
    )

    dev_samples = get_examples(get_data(lang=lang, train=False))
    evaluator = EmbeddingSimilarityEvaluator.from_input_examples(
        dev_samples,
        name=f"{lang}-dev",
        main_similarity=SimilarityFunction.COSINE,
    )

    train_loss = CosineSimilarityLoss(model)
    train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=batch_size)

    save_path = f"trained_sentence_transformers/{lang}"
    os.makedirs(save_path, exist_ok=True)

    _epochs = config.get("epochs", 1)
    warmup_steps = ceil(len(train_dataloader) * _epochs * 0.1)

    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        evaluator=evaluator,
        output_path=save_path,
        callback=train_callback,
        warmup_steps=warmup_steps,
        **config,
    )

    return evaluator(model, output_path=None, epoch=-1, steps=-1)


def train_on_df(model, df, config, batch_size=32, model_name="", evaluator=None):
    if isinstance(model, str):
        model = SentenceTransformer(model)
    data = get_examples(df)
    loader = DataLoader(data, shuffle=True, batch_size=batch_size)
    train_loss = CosineSimilarityLoss(model)
    warmup_steps = ceil(len(loader) * config.get("epochs", 1) * 0.1)

    model.fit(
        train_objectives=[(loader, train_loss)],
        output_path=f"trained-models/trained-bi-encoders/{model_name}",
        callback=train_callback,
        warmup_steps=warmup_steps,
        evaluator=evaluator,
        **config,
    )
    return model
