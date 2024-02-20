import argparse


def get_args(return_namespace: bool = False) -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_name",
        default="xlm-roberta-base",
        type=str,
        help="Name of the model",
    )
    parser.add_argument(
        "--similarity_model",
        default="intfloat/multilingual-e5-base",
        type=str,
        help="Similarity model to use for data augmentation. See `k` argument below.",
    )
    parser.add_argument(
        "--batch_size", default=32, type=int, help="Batch size for training"
    )
    parser.add_argument(
        "--lang", default="eng", type=str, help="Language of the data"
    )
    parser.add_argument(
        "--mock_eval",
        default=False,
        action="store_true",
        help="Whether to perform mock evaluation",
    )
    parser.add_argument(
        "--learning_rate",
        default=2e-5,
        type=float,
        help="Learning rate for the optimizer",
    )

    parser.add_argument(
        "--max_grad_norm",
        default=1.0,
        type=float,
        help="Maximum gradient norm for gradient clipping",
    )

    parser.add_argument(
        "--epochs", default=5, type=int, help="Number of epochs for training"
    )
    parser.add_argument(
        "--eval_steps",
        default=200,
        type=int,
        help="Number of steps for evaluation",
    )
    parser.add_argument(
        "--k",
        default=0,
        type=int,
        help="Top-k for augmented similar training pairs",
    )
    parser.add_argument(
        "--weak_training_epochs",
        default=2,
        type=int,
        help="Number of epochs for weak training",
    )
    parser.add_argument(
        "--verbose",
        default=False,
        action="store_true",
        help="Whether to print verbose output",
    )
    parser.add_argument(
        "--save_to",
        default=None,
        type=str,
        help="Path to save the trained model",
    )
    parser.add_argument(
        "--device", default="cuda", type=str, help="Device to run the model on"
    )
    parser.add_argument(
        "--max_length",
        default=128,
        type=int,
        help="Maximum length for the input sequences",
    )
    parser.add_argument(
        "--seed", default=42, type=int, help="Random seed for reproducibility"
    )

    parser.add_argument(
        "--clean",
        default=False,
        action="store_true",
        help="Whether to clean the data before training",
    )

    parser.add_argument(
        "--adversarial",
        default=False,
        action="store_true",
        help="Whether to perform adversarial validation (English data only)",
    )

    if return_namespace:
        namespace, _ = parser.parse_known_args()
        return namespace

    return parser.parse_args()


def get_namespace() -> argparse.Namespace:
    return get_args(return_namespace=True)
