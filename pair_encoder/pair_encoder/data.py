import numpy as np

from pair_encoder.util import PairInput

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
