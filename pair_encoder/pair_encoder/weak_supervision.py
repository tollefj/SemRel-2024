from typing import List

import torch

from pair_encoder.model import PairEncoder
from pair_encoder.util import PairInput, cos_sim


def label_sentences(
    bi_encoder,
    encoder: PairEncoder,
    train: PairInput,
    top_k: int,
    batch_size: int,
    verbose: bool = True,
) -> List[PairInput]:
    """
    Label sentences using weak supervision.

    Args:
        bi_encoder (Any): A bi-encoder to produce sentence embeddings. Must implement an `encode` method.
        encoder (PairEncoder): The pair encoder model.
        train (PairInput): The training data.
        top_k (int): The number of top similar sentences to consider.
        batch_size (int): The batch size for encoding sentences.
        verbose (bool, optional): Whether to show progress bar. Defaults to True.

    Returns:
        List[PairInput]: The labeled pair inputs.
    """
    sentences = set()
    for sample in train:
        sentences.update(sample.pair)
    sentences = list(sentences)
    sent2idx = {sentence: idx for idx, sentence in enumerate(sentences)}
    duplicates = set((sent2idx[data.pair[0]], sent2idx[data.pair[1]]) for data in train)

    sentences = list(sentences)
    model_name = bi_encoder.tokenizer.name_or_path
    if "e5" in model_name:
        train_sents = [f"query: {s}" for s in sentences]
    else:
        train_sents = sentences

    embeddings = bi_encoder.encode(
        train_sents,
        batch_size=batch_size,
        convert_to_tensor=True,
        show_progress_bar=verbose,
    )
    weak_data = []
    for idx in range(len(sentences)):
        sentence_embedding = embeddings[idx]
        cos_scores = cos_sim(sentence_embedding, embeddings)[0]
        top_results = torch.topk(cos_scores, k=top_k + 1)

        for _, _id in zip(top_results[0], top_results[1]):
            if _id != idx and (_id, idx) not in duplicates:
                weak_data.append((sentences[idx], sentences[_id]))
                duplicates.add((idx, _id))

    weak_scores = encoder.predict(weak_data)
    assert all(0.0 <= score <= 1.0 for score in weak_scores)

    return list(
        PairInput(pair=(data[0], data[1]), label=score)
        for data, score in zip(weak_data, weak_scores)
    )
