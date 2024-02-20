import os

from sklearn.model_selection import train_test_split
from util import get_all_langs


def generate_data():
    os.makedirs("data", exist_ok=True)
    df = get_all_langs(train=True, language_col=True)
    df_eval = get_all_langs(train=False, language_col=True)
    df_test = get_all_langs(test=True, language_col=True)

    df.to_csv("data/train.csv", index=False)
    df_eval.to_csv("data/eval.csv", index=False)
    df_test.to_csv("data/test.csv", index=False)

    train, holdout = train_test_split(df, test_size=0.2, random_state=42)
    train.to_csv("data/holdout_train.csv", index=False)
    holdout.to_csv("data/holdout_test.csv", index=False)

    print(
        f"Created {len(df)} train, {len(df_eval)} eval and {len(df_test)} test samples."
    )
    print(
        f"Created {len(train)} holdout train and {len(holdout)} holdout test samples."
    )


if __name__ == "__main__":
    generate_data()
