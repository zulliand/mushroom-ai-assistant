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

## Iterative development

- **Iteration 1 — binary CV baseline:** a binary edible/poisonous image classifier to validate pipeline and inference integration.
- **Iteration 2 — 46-class species model:** expanded species labels to a mid-sized taxonomy to validate multi-class training and data pipeline.
- **Iteration 3 — 169-class species model:** final species model trained on the Kaggle species dataset with top-k evaluation (top-1, top-3, top-5 metrics).

### Why top-k metrics matter

Fine-grained species recognition is inherently ambiguous from single images: many mushroom species share visual traits and intra-class variability (growth stages, lighting, occlusion). Top-3 and top-5 accuracy measure whether the correct species appears among the top candidates returned by the model — a practical and useful signal for downstream human-in-the-loop workflows and LLM-guided explanations.

### Error analysis and safety rationale

- **Visual similarity:** Several species are visually similar (color variants, subtle cap/gill differences) which reduces single-image top-1 accuracy even when the model retains useful information in higher-ranked predictions.
- **Domain shift:** Field photos from users or external sources often differ from the training distribution (camera, lighting, background), which can degrade performance.
- **Numeric model scope:** The structured-data model operates purely on JSON features (cap shape, odor, gill color, etc.) and does not inspect the image; it provides a complementary signal that may agree or disagree with CV.
- **Safety layer behavior:** When the CV and numeric model disagree (for example CV predicts a poisonous species while structured features strongly indicate edible), the safety layer intentionally withholds cooking suggestions and blocks the high-trust path to avoid giving dangerous advice.

### Final example (illustrative)

- CV: `Amanita_muscaria` detected with high confidence (e.g., 79.5%).
- Numeric: structured features converted from CSV indicate `edible` with 76.0% probability.
- Result: conflict detected by the safety logic and cooking suggestions are blocked.

## Deployment Notes

- The app expects `models/mushroom_rf.pkl` and `models/mushroom_cv.pt` to exist.
- OpenAI is optional at runtime; if no key is available, the app falls back to a deterministic safety-focused response.
- Hugging Face Spaces deployment should expose `app.py` and include `requirements.txt`.
