# Mushroom AI Assistant

An integrated university semester project that combines three AI blocks:

1. Computer vision transfer learning for mushroom image classification.
2. Structured-data machine learning for edible vs poisonous prediction.
3. An LLM layer that explains the combined result and applies a safety-first policy.

## Project Layout

- `app.py` - Gradio frontend that ties all three blocks together.
- `train_cv.py` - transfer-learning training and CV inference helpers.
- `train_numeric.py` - tabular training, evaluation, and numeric inference helpers.
- `llm.py` - OpenAI integration plus offline fallback responses.
- `utils.py` - shared paths, logging, and artifact helpers.
- `models/` - saved artifacts such as `mushroom_cv.pt` and `mushroom_rf.pkl`.
- `data/` - structured CSV and image folders.

## Expected Data

- `data/mushroom_numeric.csv` for the structured mushroom dataset.
- `data/images/` for the image dataset.
- For CV training, use either `data/images/train/<class-name>/...` and `data/images/val/<class-name>/...` or a single `data/images/<class-name>/...` folder structure.

## Install

```bash
pip install -r requirements.txt
```

## Train the numeric model

```bash
python train_numeric.py --data-path data/mushroom_numeric.csv
```

## Train the CV model

```bash
python train_cv.py --data-dir data/images
```

## Run the app

```bash
python app.py
```

## Hugging Face Spaces

Set the Space to Python and expose `app.py` as the entry point. Provide `OPENAI_API_KEY` as a Space secret if you want live LLM generation.
