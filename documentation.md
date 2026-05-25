# Technical Documentation

## Architecture

The assistant is split into three cooperating stages:

1. Computer vision classifies the uploaded mushroom image with a transfer-learning model.
2. A structured-data classifier predicts edible vs poisonous from mushroom attributes.
3. The LLM receives both outputs and produces a concise explanation, a safety warning, optional cooking suggestions, and a mandatory educational disclaimer.

## Integration Rule

The final answer is not produced by any one model alone. The LLM step explicitly combines the CV result and the structured-data result, and cooking advice is blocked unless both signals are highly confident.

## Numeric Model

- Preprocessing handles categorical and numeric columns.
- Logistic Regression and Random Forest are compared.
- The best model is selected by F1 score.
- A confusion matrix and metadata JSON are saved with the trained artifact.

## CV Model

- ResNet18 transfer learning is used.
- The backbone is frozen when pretrained weights are available.
- The trained checkpoint stores class names, label mapping, image size, and training metrics.

## Deployment Notes

- The app expects `models/mushroom_rf.pkl` and `models/mushroom_cv.pt` to exist.
- OpenAI is optional at runtime; if no key is available, the app falls back to a deterministic safety-focused response.
- Hugging Face Spaces deployment should expose `app.py` and include `requirements.txt`.
