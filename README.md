# Mushroom AI Assistant — Final Status

This repository contains the final code and documentation for the Mushroom AI Assistant (semester project). The project integrates three cooperating AI blocks to assess mushroom images and structured observations, provide explainable outputs and enforce a safety-first policy.

Key components

- **Computer Vision (species recognition):** transfer-learning model for mushroom species classification (final model: ResNet18, 169 classes).
- **Structured numeric ML model:** tabular classifier trained on the UCI Mushroom Classification CSV for edible/poisonous predictions from structured features.
- **LLM/NLP layer:** OpenAI-based explanation and safety layer that receives both CV and numeric signals, explains results and blocks unsafe recommendations when models disagree.

Datasets used

1. UCI Mushroom Classification Dataset
	- Purpose: Structured feature prediction (Numeric ML)

2. Edible & Poisonous Mushroom Classification Dataset
	- Purpose: Initial binary computer vision baseline

3. Mushroom Species Recognition Dataset
	- Purpose: Fine-grained species recognition (169 classes) — final computer vision model

Final species model (report)

- Model architecture: ResNet18 (transfer learning)
- Number of classes: 169
- Training samples used (approx): 30,000
- Validation samples: 6,000
- Test samples: 6,000
- Training epochs (final run): 15
- Test metrics:
	- Top-1 accuracy: 53.45%
	- Top-3 accuracy: 74.40%
	- Top-5 accuracy: 82.17%

Notes about artifacts and deployment

- The repository does not contain large datasets; these must be provided separately according to the data layout used by the scripts.
- Trained model checkpoints are referenced in `models/` metadata files but large checkpoint files are not committed here by default.
- When deploying to Hugging Face Spaces, do NOT commit API keys or `.env` files. Instead configure runtime secrets as described below.

Quick start

1. Install requirements:

```bash
pip install -r requirements.txt
```

2. Provide required artifacts (models + numeric pipeline) and datasets under `models/` and `data/` as needed.

3. Run the app locally:

```bash
py app.py
```

Hugging Face Spaces deployment notes

- Set `OPENAI_API_KEY` as a Space secret (do not commit it into the repo).
- Large datasets are not included in the Space; use external storage or dataset downloads during build if required.
- Only deploy the minimal runtime files (app, models metadata, small model artifacts you are comfortable hosting). See `documentation.md` for more details.

License / Attribution

See project header and documentation for dataset attributions and licenses. If you add third-party checkpoints, ensure compliance with their licenses.
