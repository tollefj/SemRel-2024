# Submissions

## Team name
Team triplet

## Method name
top-k pair encoding
<!-- small encoders with top-k augmented sentence-pair fine-tuning -->

# Modeling notes
we use STS-based data for fine-tuning a baseline bert-based model, then data is augmented by sampling up to k=4 similar sentences from the embedding space of the training data, using a pre-trained small sentence embedding model and train our encoder for 1 epoch prior to weak labeling.
Due to observed data distribution variances (train-test), we only train for a single epoch after weak labeling with a low gradient norm.

## Models
- e5-multilingual base + large gives overall decent results on all languages
## To consider:
- setu4993/LEALLA-large
    - https://aclanthology.org/2023.eacl-main.138.pdf


## Lang-specific models:
- amh: Amharic
    - base: Davlan/xlm-roberta-base-finetuned-amharic
- arq: Algerian Arabic
    - Abdou/arabert-large-algerian
    - Abdou/arabert-base-algerian
    - alger-ia/dziribert
        - probably the best!
- ary: Moroccan Arabic
    - base: SI2M-Lab/DarijaBERT
- eng: English
    - intfloat/multilingual-e5-large
- esp: Spanish
    - hackathon-pln-es/paraphrase-spanish-distilroberta
- hau: Hausa
    - base: Davlan/xlm-roberta-base-finetuned-hausa
- kin: Kinyarwanda
    - base: Davlan/xlm-roberta-base-finetuned-kinyarwanda
- mar: Marathi
    - base: deepampatel/roberta-mlm-marathi
    - base: DarshanDeshpande/marathi-distilbert
- tel: Telugu
    - base: l3cube-pune/telugu-bert


## Notes on existing parallel data corpora:
TED2020
- Has all languages except for MOROCCAN ARABIC and KINYARWANDA



# NEW AS OF 17.01.24
Just train cross-encoders, dammit.

Data sources:
- english, arabic, spanish:
    - https://huggingface.co/datasets/mteb/sts17-crosslingual-sts
    - https://huggingface.co/datasets/mteb/sts22-crosslingual-sts
    - mteb/sts17-crosslingual-sts
    - mteb/sts22-crosslingual-sts
- english


## Data expansion:
- embedding-data/coco_captions_quintets -- 80k samples!
    - 5 sentences of similar meaning
        - weakly label each one to create pseudolabels
            - retrain!
    - https://huggingface.co/datasets/embedding-data/coco_captions_quintets
- embedding-data/altlex -- 113k samples of sentence-pairs (parallel from wiki)
- mteb/sickr-sts -- 10k samples of pairs with label (1-5, just normalize it)

# Other notes
- Hausa and Telugu are well represented in MassiveSumm: https://aclanthology.org/2021.emnlp-main.797.pdf
