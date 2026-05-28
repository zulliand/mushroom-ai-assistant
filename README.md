# Mushroom AI Assistant

The Mushroom AI Assistant is an educational AI system that combines Computer Vision, Numeric Machine Learning and NLP to analyse mushroom images and structured mushroom attributes.

The application predicts mushroom species, estimates edibility from structured observations, generates explanations via a Large Language Model (LLM) and applies safety logic to avoid unsafe recommendations.

---

## Project Overview

This repository contains the final implementation and documentation for the Mushroom AI Assistant semester project.

The project integrates three cooperating AI components:

### 1. Computer Vision (Species Recognition)

Transfer learning model for mushroom species recognition from images.

Final model:

- Architecture: EfficientNet-B0
- Fine-grained classification
- 169 mushroom species classes
- Top-k candidate predictions
- Species confidence estimation

### 2. Structured Numeric Machine Learning

Tabular classification model trained on structured mushroom attributes.

Purpose:

- Predict edible vs poisonous mushrooms
- Use structured mushroom observations
- Provide independent validation signal
- Enable conflict detection against image predictions

### 3. NLP / LLM Safety Layer

OpenAI-based explanation and safety component.

Responsibilities:

- Explain predictions
- Compare CV and numeric model outputs
- Detect conflicting predictions
- Block unsafe recommendations
- Enforce conservative safety policies

![CV and NLP inference without structured numeric inputs](docs/screenshots/CV_NLP.png)
*Figure 2. CV and NLP inference without structured numeric features.*

---

## System Architecture

```
Image
 ↓
Computer Vision Model
 ↓
Species prediction + confidence
 ↓
Conflict Detection
 ↑
Numeric ML Model ← Structured Features JSON
 ↓
LLM Explanation Layer
 ↓
Safety Gating
 ↓
Final Output
```

![Full multimodal pipeline interface](docs/screenshots/CV_NLP_Numeric_Model_Input.png)
*Figure 1. Full multimodal evaluation with manual structured inputs.*

---

## Datasets Used

| Dataset | Purpose | Data Type |
|----------|----------|------------|
| UCI Mushroom Classification Dataset (kaggle.com/datasets/uciml/mushroom-classification) | Exploratory structured analysis | Numeric CSV |
| Mushroom Edibility Classification (kaggle.com/datasets/devzohaib/mushroom-edibility-classification) | Structured edible / poisonous prediction — final numeric model | Numeric CSV |
| Edible & Poisonous Mushroom Classification (kaggle.com/datasets/benedictusjason/edible-and-poisonous-mushroom-classification) | Initial binary computer vision baseline | Images |
| Mushroom Species Recognition (kaggle.com/datasets/zlatan599/mushroom1) | Fine-grained species recognition (final CV model) | Images |

The project intentionally uses datasets that were not part of the semester exercises.

---

## Final Species Recognition Model

### Model Configuration

- Architecture: EfficientNet-B0 (transfer learning)
- Number of classes: 169
- Training samples used: ~40,000
- Validation samples: ~13,500
- Test samples: ~13,500
- Final training epochs: 20

### Final Metrics

| Metric | Result |
|---------|---------|
| Top-1 Accuracy | 55.13% |
| Top-3 Accuracy | 74.40% |
| Top-5 Accuracy | 80.67% |

The species recognition task contains 169 visually similar mushroom classes. Therefore, Top-k metrics provide more informative performance indicators than Top-1 accuracy alone.

---

## Safety Design

The application intentionally blocks unsafe outputs.

Cooking recommendations are blocked when:

- Model confidence is too low
- Computer Vision and Numeric ML predictions disagree
- Potentially poisonous species are detected
- Safety thresholds are not met

This conservative design prioritises safety over aggressive prediction behaviour.

![Conservative safety handling under uncertainty](docs/screenshots/Description.png)
*Figure 3. Conservative safety handling under low-confidence conditions.*

---

## Features

### Computer Vision

- Mushroom species recognition
- Top-k predictions
- Confidence estimation
- Transfer learning pipeline

### Numeric Machine Learning

- Structured feature classification
- Edible vs poisonous prediction
- Probability estimation

### NLP Layer

- Prediction explanation
- Safety messaging
- Prompt comparison
- Conflict analysis

### Integrated Safety Logic

- Conflict detection
- Recommendation blocking
- Educational disclaimers

---

## Quick Start

### Install dependencies

```bash
pip install -r requirements.txt
```

### Required artifacts

Provide:

- Model files
- Numeric ML pipeline
- Dataset files (if training locally)

Expected structure:

```
models/
data/
```

### Run locally

```bash
py app.py
```

Open:

```
http://localhost:7860
```

---

## Hugging Face Deployment

For deployment to Hugging Face Spaces:

### Required Secret

Configure:

```
OPENAI_API_KEY
```

Training vs Inference

Training was performed using dedicated scripts (`train_cv.py`, `train_numeric.py`, `train_species_cv.py`) and produced the model artifacts stored in the `models/` directory. Inference and the deployed app rely on these pre-trained artifacts and are executed via `app.py`; the deployment does not retrain models at runtime.

Acknowledgements / Submission

- Jasmin Heierli — jasminh
- Benjamin Kühnis — bkuehnis

Do NOT commit:

- API keys
- .env files
- Large datasets
- Training folders

Large datasets are intentionally excluded from the repository.

---

## Repository Notes

The repository excludes:

- Raw datasets
- Species image folders
- Large checkpoints
- Temporary training artifacts

Model metadata remains included for reproducibility.

---

## Limitations

- Top-1 species accuracy remains challenging due to visual similarity between mushroom species.
- Field photographs introduce domain shift caused by lighting, background clutter, camera quality and viewpoint variation.
- The numeric model relies solely on structured attributes and may disagree with image-based predictions.
- Confidence calibration remains limited.
- The system is intended for educational purposes only.

This application must NOT be used as the sole basis for real-world mushroom foraging decisions.

---

## Future Improvements

Potential future work:

- Confidence calibration and uncertainty estimation
- Expanded dataset coverage for underrepresented species
- Additional field-photo evaluation
- Improved species balancing
- User feedback loop for correction labels
- Taxonomy refinement
- Model ensemble approaches
- Additional safety verification layers

---

## License and Attribution

Please refer to dataset providers and source repositories for licensing information.

Datasets used remain attributed to their original creators.

---

## Educational Disclaimer

This project was developed for educational purposes as part of a university semester project.

Predictions may be incorrect.

Wild mushrooms can be dangerous.

Always consult qualified experts before consuming wild mushrooms.
