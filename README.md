# PEAR: Pair Encoding with Augmented Re-sampling for Semantic Textual Relatedness

A system submitted to SemEval-2024 Task 1
___

- Figures and experiments are found under the [`sections`](sections)
- Source code in [`src`](src).
- The SemRel shared task github repo is added as a submodule. To initialize:
- The simplified pair encoder model and related pipelines is added as a separate folder (installable with `pip install .`) in the [`pair_encoder`](pair_encoder) directory.

```bash
git submodule update --init --recursive
```

The final values are obtained through `src/pairencoder_optimize.py` with the following multi/monolingual model selection (see `src/models.py`)

```python
models = {
    "multilingual": "FacebookAI/xlm-roberta-base",
    "arq": "CAMeL-Lab/bert-base-arabic-camelbert-da",
    "amh": "Davlan/xlm-roberta-base-finetuned-amharic",
    "eng": "FacebookAI/roberta-base",
    "hau": "Davlan/xlm-roberta-base-finetuned-hausa",
    "kin": "Davlan/xlm-roberta-base-finetuned-kinyarwanda",
    "mar": "l3cube-pune/marathi-roberta",
    "ary": "CAMeL-Lab/bert-base-arabic-camelbert-da",
    "esp": "PlanTL-GOB-ES/roberta-base-bne",
    "tel": "l3cube-pune/telugu-bert",
}
```
