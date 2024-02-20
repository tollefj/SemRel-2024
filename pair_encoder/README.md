# Augmented Pair Encoder
A pipeline for training a binary encoder on similarity-based data with weak labeling.


### Example
- With weak labeling:
  - `python pipeline.py bert-base-uncased --similarity_model="intfloat/multilingual-e5-base" --learning_rate=2e-5 --epochs=3 --k=2 --verbose`
- Without:
  - `python pipeline.py bert-base-uncased --learning_rate=2e-5 --epochs=3 --verbose`