# Mushroom AI Assistant - Documentation

## Project Metadata

- Project title: Mushroom AI Assistant
- Student: Andres Zulliger
- GitHub repository URL: https://github.com/zulliand/mushroom-ai-assistant
- Deployment URL: https://huggingface.co/spaces/zulliand/mushroom-ai-assistant
- Submission date: 07 June 2026

### Mandatory Setup Checks

- [x] At least 2 blocks selected
- [x] Multiple and different data sources used
- [x] Deployment URL provided
- [x] Required GitHub users added to repository (`jasminh`, `bkuehnis`)

## Selected AI Blocks

- [x] ML Numeric Data
- [x] NLP
- [x] Computer Vision

Primary blocks used for core solution (choose 2):
- Primary block 1: Computer Vision
- Primary block 2: ML Numeric Data

---

## 1. Project Foundation (Short)

### 1.1 Problem Definition

- Problem statement: Correct mushroom identification is difficult because many species are visually similar, and misidentification can cause severe health risks.
- Goal: Build a multimodal assistant that combines Computer Vision, Numeric ML, and NLP to provide explainable, conservative, safety-first outputs.
- Success criteria: Reliable species prediction pipeline, optional structured-feature risk signal, clear uncertainty communication, and conservative final safety behavior.

**Scope:**
- The system accepts a single mushroom image and optional structured trait inputs.
- It predicts species from a fixed set of 169 known species; unknown or out-of-distribution species are not identified.
- The NLP component requires an OpenAI API key; without it the system falls back to a deterministic safety response.
- The system is strictly educational and must not be used as the sole basis for consuming wild mushrooms.
- Training is out of scope at runtime; the deployment performs inference only.

**Assumptions:**
- The user provides a reasonably clear, close-up image of a single mushroom.
- Manually entered structured traits (cap shape, habitat, etc.) reflect the actual specimen observed.
- Internet connectivity is available for the Hugging Face deployment and the OpenAI API calls.

### 1.2 Integration Logic

- How the selected blocks interact: CV predicts species and confidence; Numeric ML provides an auxiliary edible/poisonous signal when structured mushroom traits are available; NLP explains outputs and uncertainty; safety logic combines these signals conservatively.
- Data and output flow between blocks:
  1. User uploads mushroom image.
  2. Computer Vision predicts species and confidence.
  3. Structured mushroom traits (manual or LLM-extracted) optionally feed Numeric ML.
  4. Safety logic combines CV, Numeric ML (if available), and species mapping.
  5. NLP generates explanation and warning text.

See `run_assistant()` in [`app.py`](app.py#L604).

**Technical implementation details:**

| Component | Libraries / Frameworks |
| --- | --- |
| Computer Vision | PyTorch 2.2, torchvision 0.17 (ResNet18, EfficientNet-B0 via `torchvision.models`) |
| Numeric ML | scikit-learn 1.4 (LogisticRegression, RandomForestClassifier, Pipeline, ColumnTransformer, OneHotEncoder) |
| NLP / LLM | OpenAI SDK 1.30 (chat completions, JSON mode, structured extraction) |
| Web UI | Gradio 4.x (image upload, dropdowns, tabbed output panels) |
| Data handling | pandas 2.2, numpy 1.26 |
| Serialization | joblib 1.3 (numeric pipeline), torch.save/load (CV checkpoints) |
| Deployment | Hugging Face Spaces (Gradio runtime, Git LFS for model artifacts) |

Project structure: `app.py` (UI + inference orchestration), `train_cv.py` / `train_species_cv.py` / `train_numeric.py` (training scripts), `llm.py` (NLP layer), `utils.py` (shared paths and helpers), `species_info.py` (species metadata).

![Full multimodal pipeline interface](docs/screenshots/CV_NLP_Numeric_Model_Input.png)
*Figure 1. Full multimodal evaluation with manual structured inputs.*

---

## 2. Block Documentation

### 2A. ML Numeric Data (If selected)

#### 2A.1 Data Source(s)

| Entry | Source (file / location) | Type | Size | Role in this block |
| --- | --- | --- | --- | --- |
| 1 | `data/Mushroom data.csv` — UCI Mushroom Classification Dataset (Kaggle, UCI ML Repository, CC0 Public Domain) | Structured CSV (categorical) | 8,124 rows, 23 columns, 374 kB | Exploratory dataset used for early experiments, preprocessing validation and feature exploration. |
| 2 | `data/secondary_data.csv` — Mushroom Edibility Classification (Kaggle, DevZohaib, CC BY-NC-SA 4.0) | Structured CSV (numerical + categorical traits) | 61,069 rows, 21 columns, 2.9 MB | Final dataset used to train and evaluate the deployed Numeric ML pipeline (deployed Random Forest). |
| 3 | Runtime structured inputs | User-provided manual dropdowns and LLM-extracted JSON features (per inference) | per request | Optional inference-time input source that enables the Numeric ML component when available. |

Data references: `data/Mushroom data.csv`, `data/secondary_data.csv`, and evaluation artifacts in `models/numeric_metrics.json`.

Note: `data/Mushroom data.csv` functioned primarily as an exploratory resource; `data/secondary_data.csv` is the structured dataset used in the final numeric pipeline. Runtime user inputs (manual traits or LLM-extracted traits) are distinct from both and are only used at inference time when provided.

#### 2A.2 Preprocessing and Features

- Cleaning steps: categorical normalization, alias mapping, feature ordering.
- Preprocessing steps: tabular feature encoding and pipeline-based transformation before model inference.
- Feature engineering and selection: structured mushroom traits such as cap shape/color, gill attributes, stem features, habitat, season, bruises/bleeding.

See `predict_numeric_sample()` in [`app.py`](app.py#L296).

**Exploratory Data Analysis (EDA)**

*Dataset overview:* `secondary_data.csv` contains 61,069 rows × 21 columns (20 features + 1 target). Target distribution: `p` (poisonous) = 33,888 rows (55.5%), `e` (edible) = 27,181 rows (44.5%) — moderate class imbalance favouring the poisonous class.

*Missing data:* Several features have extremely high missing rates, which constrained feature selection and required pipeline-level imputation handling:

| Feature | Missing (%) |
| --- | --- |
| veil-type | 94.8 |
| spore-print-color | 89.6 |
| veil-color | 87.9 |
| stem-root | 84.4 |
| stem-surface | 62.4 |

*Feature observation 1 — Habitat:* The `habitat` feature contains values with near-perfect class separation. Paths (`p`) map exclusively to poisonous samples; urban (`u`) and waste (`w`) habitats contain exclusively edible samples. The most frequent habitat, woodland (`d`), accounts for 72.4% of all rows with a 46/54 edible-poisonous split. This structural property of the dataset directly explains the perfect test metrics reported in section 2A.5.

*Feature observation 2 — Cap diameter:* The only continuous numeric feature, `cap-diameter`, shows a distributional shift between classes: edible mushrooms have a mean diameter of 7.80 cm (SD 6.37) versus 5.88 cm (SD 3.97) for poisonous ones. While the distributions overlap substantially and cap diameter alone is not a reliable separator, the difference is consistent and contributes marginal discriminative signal alongside categorical features.

*Interpretation:* The dataset is a synthetically generated resource constructed from rule-based morphological patterns, which results in highly separable class boundaries for several categorical features. This is consistent with the known properties of the UCI extended mushroom dataset and is addressed in detail in section 2A.5.

#### 2A.3 Model Selection

- Models tested: Logistic Regression, Random Forest.
- Why these models were chosen: Logistic Regression as linear baseline; Random Forest for nonlinear interactions common in categorical mushroom traits.

#### 2A.4 Model Comparison and Iterations

| Iteration | Objective | Key changes | Models used | Main metric | Change vs previous |
| --- | --- | --- | --- | --- | --- |
| 1 | Establish baseline | Basic OHE preprocessing, linear classifier | Logistic Regression | acc 0.865, precision 0.876, recall 0.880, F1 0.878 | Baseline — strong linear separability but leaves nonlinear structure unexploited |
| 2 | Improve nonlinear performance | Tree ensemble on same feature set | Random Forest (100 trees) | acc 1.000, precision 1.000, recall 1.000, F1 1.000 | Exploits deterministic categorical splits in synthetic dataset; perfect separation achieved |
| 3 | Integrate in multimodal pipeline | Optional runtime use with manual/LLM traits | Random Forest (deployed) | Safety-consistent auxiliary signal | Avoids over-trusting standalone numeric output; treated as supporting evidence only |

#### 2A.5 Evaluation and Error Analysis

- Metrics used: accuracy, precision, recall, F1 (single 80/20 train/test split).
- Final results: accuracy=1.0, precision=1.0, recall=1.0, F1=1.0 on the 20% held-out test split.

**Discussion of perfect metrics:**

The perfect scores are best explained by the structure of the dataset rather than by overfitting in the conventional sense. As shown in the EDA (section 2A.2), `secondary_data.csv` is a synthetic dataset with rule-based construction: several categorical features contain values that map deterministically to the target class. Examples include `habitat='p'` (path) → 100% poisonous; `spore-print-color='n'` (brown) → 100% poisonous; `spore-print-color='g'` (green) → 100% edible. A Random Forest trivially exploits these exact categorical thresholds and achieves zero misclassifications on held-out data drawn from the same distribution.

**Key limitations of this evaluation:**

- *No cross-validation:* Only a single 80/20 split was used. Without k-fold CV, there is no estimate of variance in model performance and no guard against lucky splits.
- *No out-of-domain test:* The held-out test split is drawn from the same synthetic distribution as training data. The model has not been evaluated against real-world mushroom trait observations, which are noisier and may include combinations not represented in the dataset.
- *Inference-time distribution shift:* Features with the strongest predictive signal in the dataset (e.g., `spore-print-color`, `veil-type`) have missing rates above 89% and are therefore rarely available at inference time. At runtime, users provide manual trait inputs, which may differ substantially from the training distribution and may not trigger the same decision paths.
- *Separability ≠ real-world reliability:* Perfect separability on a synthetic dataset does not imply reliable predictions on ambiguous, partial, or noisy real inputs.

**Mitigation in the multimodal system:** the Numeric ML component is deliberately treated as an auxiliary supporting signal, not a standalone decision-maker. The safety logic requires corroboration from the CV prediction, and conflicting signals between the two models trigger a conservative block rather than any committed recommendation. This design explicitly acknowledges the limited real-world generalizability of the numeric model.

Source: [`models/numeric_metrics.json`](models/numeric_metrics.json#L1)

#### 2A.6 Integration with Other Block(s)

- Inputs received from other block(s): optional structured traits extracted from user text via NLP; species context from CV pipeline state.
- Outputs provided to other block(s): edible/poisonous probability signal used by safety logic and summarized by NLP explanation output.

See `extract_mushroom_features()` in [`llm.py`](llm.py#L134), `generate_mushroom_advice()` in [`llm.py`](llm.py#L294), and `run_assistant()` in [`app.py`](app.py#L604).

### 2B. NLP (If selected)

#### 2B.1 Data Source(s)

| Entry | Source name or link | Type | Size | Role in this block |
| --- | --- | --- | --- | --- |
| 1 | User text descriptions and uploaded images | Runtime text input + runtime images | per request | Explanation context and optional LLM-based feature extraction; uploaded images are used by CV inference (see 2C) |
| 2 | CV result payload | Structured model output | per inference | Species/confidence context for explanation (CV models trained on local image datasets under `data/`) |
| 3 | Numeric ML result payload (optional) | Structured model output | per inference | Auxiliary risk signal in explanation and safety wording |

#### 2B.2 Preprocessing and Prompt Design

- Text preprocessing: normalize user description text, pass concise structured context fields.
- Prompt design or retrieval setup: strict prompt instructions, safety framing, JSON-oriented extraction for structured features, conservative fallback behavior.

The NLP layer itself is not trained from scratch; it uses OpenAI models with prompt engineering and structured context injection.

#### 2B.3 Approach Selection

- Approach used: prompt engineering with OpenAI chat models for explanation generation and optional structured feature extraction.
- Alternatives considered: prompt variants A and B for explanation quality and safety consistency.

#### 2B.4 Comparison and Iterations

| Iteration | Objective | Key changes | Model or prompt setup | Main metric or qualitative check | Change vs previous |
| --- | --- | --- | --- | --- | --- |
| 1 | Initial explanation behavior | Generic instruction: "Explain the mushroom prediction briefly." No safety constraints; outputs `explanation`, `safety_warning`, `disclaimer` only. | Prompt A | Qualitative safety/readability review | Baseline — readable but inconsistent safety language; cooking suggestions not gated on confidence. |
| 2 | Improve consistency and safety language | Stricter instruction: explicit confidence reporting, conditional cooking gate (`allow_cooking` flag + 0.85 threshold), required disclaimer, suppressed numeric model output when not evaluated. | Prompt B | Qualitative comparison across 5 sample scenarios; heuristic scoring (structure, safety, cooking-gate correctness) | Prompt B scored higher on safety and cooking-gate dimensions in all tested scenarios. |
| 3 | Add feature extraction role | Separate extraction prompt with JSON schema for 20 structured mushroom traits; null-preference over guessing; robust fallback to defaults on parse failure. | Prompt B + extraction schema | Valid extraction fields + graceful fallback | Enables optional Numeric ML input path without requiring fully manual feature entry. |

Prompt A vs B comparison is implemented in `compare_prompt_strategies()` in [`llm.py`](llm.py#L354) using a heuristic scorer (`heuristic_prompt_score()` in [`llm.py`](llm.py#L232)) that evaluates structure completeness, safety language presence, and cooking-gate correctness.

#### 2B.5 Evaluation and Error Analysis

- Evaluation strategy: qualitative side-by-side comparison of Prompt A and Prompt B outputs across representative scenarios (confident edible prediction, confident poisonous prediction, low-confidence case, conflict case, no numeric input); heuristic scoring on structure, safety, and cooking-gate dimensions.
- Results: Prompt B consistently scored equal or higher overall, driven by the explicit safety constraints and the conditional cooking gate that Prompt A lacks.
- Error patterns and likely causes: possible hallucinations or overconfident phrasing in Prompt A mitigated by Prompt B's stricter instruction wording; remaining risk addressed by deterministic fallback response when the API is unavailable (see `fallback_response()` in [`llm.py`](llm.py#L251)).

#### 2B.6 Integration with Other Block(s)

- Inputs received from other block(s): CV prediction/confidence and optional Numeric ML probabilities.
- Outputs provided to other block(s): explanation text, uncertainty communication, and optional structured feature extraction feeding Numeric ML.

### 2C. Computer Vision (If selected)

#### 2C.1 Data Source(s)

| Entry | Source (file / location) | Type | Size | Role in this block |
| --- | --- | --- | --- | --- |
| 1 | Edible & Poisonous Mushroom Classification Dataset (Kaggle, Benedictus Jason, MIT License) — local folders `data/train`, `data/val`, `data/test` | Image dataset (binary labels) | 2,820 images; 25 edible + 22 poisonous species; 80/10/10 split | Used for initial binary CV training and baseline evaluation (edible vs poisonous). |
| 2 | Mushroom Species Recognition Dataset (Kaggle, Leonardo Cofone / zlatan599, MIT License, kaggle.com/datasets/zlatan599/mushroom1) — `data/species_images` and `data/raw/mushroom_species_recognition/merged_dataset` | Image dataset (fine-grained species folders) | ~67,000 images used (169 classes, max 300/class); full dataset 104k files / 12.18 GB; ~40k train, ~13k val, ~13k test | Final species-recognition training and evaluation (multi-class, used for deployed species CV model). |
| 3 | Local curated mushroom images | Runtime evaluation images | 3 example images | Small curated set used for manual evaluation and deployment checks |

Data references: `data/species_images/`, `data/train/`, `data/val/`, `data/test/`, and evaluation artifacts in `models/species_cv_metrics.json` and `models/cv_metrics.json`.

Note: The CV blocks use locally stored image folders for training and evaluation; at inference, user-uploaded images are processed by the same CV models trained on the folders listed above.

#### 2C.2 Preprocessing and Augmentation

- Image preprocessing: resizing, normalization, tensor conversion.
- Augmentation strategy: random crops, horizontal flips, color jitter.

#### 2C.3 Model Selection

- Vision model(s) used: ResNet18 (binary baseline, Iteration 1), EfficientNet-B0 (deployed species model, Iteration 3).
- Why these model(s) were chosen: ResNet18 as a lightweight transfer-learning baseline; EfficientNet-B0 for the fine-grained species task due to its stronger feature extraction at comparable compute cost.

#### 2C.4 Model Comparison and Iterations

| Iteration | Objective | Key changes | Model(s) used | Main metric | Change vs previous |
| --- | --- | --- | --- | --- | --- |
| 1 | Binary classification baseline | 2-class (edible vs poisonous) setup, image size 160 | ResNet18 | val acc 80.1%, test acc 79.1% | Established transfer-learning baseline |
| 2 | First species-level experiments | Scaled to 169-class task; multiple runs with varying epoch counts (5, 10, 20, 30 epochs); explored different fine-tuning depths | ResNet18 | Top-1 ~40–48% across runs | Showed fine-grained classification is feasible but ResNet18 plateaued; longer training gave diminishing returns |
| 3 | Architecture switch to EfficientNet-B0 | Replaced backbone with EfficientNet-B0 for stronger feature extraction at similar compute; image size 224 | EfficientNet-B0 | Top-1 improved over ResNet18 runs | Better generalisation on visually similar species; motivated full training run |
| 4 | Final training with extended epochs | 20 epochs, early stopping (patience 3), top-k evaluation strategy adopted | EfficientNet-B0 (deployed) | Top-1 55.1% / Top-3 74.4% / Top-5 80.7% | Best result across all CV iterations; deployed as final species model |

See `predict_image()` in [`train_cv.py`](train_cv.py#L304).

#### 2C.5 Evaluation and Error Analysis

- Metrics and/or visual checks: accuracy for binary model; top-1, top-3, top-5 accuracy for species model; qualitative review of low-confidence examples.
- Results by model:

| Model | Task | Val Acc | Test Acc | Top-3 | Top-5 |
| --- | --- | --- | --- | --- | --- |
| ResNet18 (Iteration 1) | Binary edible/poisonous (2 classes) | 80.1% | 79.1% | — | — |
| EfficientNet-B0 (deployed) | Species recognition (169 classes) | 55.9% | 55.1% | 74.4% | 80.7% |

- Error patterns and limitations: the binary model shows acceptable accuracy for a 2-class task but is not used in the final pipeline. The species model at 55.1% top-1 reflects the difficulty of 169-class fine-grained classification with visually similar species; top-3 and top-5 scores (74.4% / 80.7%) are more representative of practical retrieval quality. Main error sources: visually similar species pairs, domain shift between training photos and user-uploaded images, and sensitivity to lighting and clutter.

Top-k metrics are emphasized because the correct species appears among the top candidates even when top-1 is uncertain, which is the relevant signal for the safety logic.

Sources: [`models/species_cv_metrics.json`](models/species_cv_metrics.json#L1), [`models/cv_metrics.json`](models/cv_metrics.json#L1)

#### 2C.6 Integration with Other Block(s)

- Inputs received from other block(s): none required for base CV inference.
- Outputs provided to other block(s): species prediction and confidence feed safety logic and NLP explanation layer.

Safety mapping reference: [`species_info.py`](species_info.py#L1).

---

## 3. Deployment

- Deployment URL: [https://huggingface.co/spaces/zulliand/mushroom-ai-assistant](https://huggingface.co/spaces/zulliand/mushroom-ai-assistant)
- Platform: Hugging Face Spaces (Gradio runtime)
- Framework: Gradio 4.x — provides the web UI with image upload, structured input dropdowns, and formatted output panels
- Model artifacts: pre-trained model files (`mushroom_cv.pt`, `mushroom_species_cv.pt`, `mushroom_numeric_pipeline.pkl`) stored in the repository via Git LFS and loaded at startup
- API key: an OpenAI API key must be provided by the user in the UI or set as a Hugging Face Space secret (`OPENAI_API_KEY`); without it the system uses a deterministic fallback response
- Separation of training and inference: all training was performed locally using dedicated scripts (`train_cv.py`, `train_species_cv.py`, `train_numeric.py`); the deployed app (`app.py`) performs inference only and does not retrain models at runtime

**Main user flow:**
1. User uploads a mushroom image
2. CV model predicts species and confidence (top-k candidates shown)
3. User optionally enters structured mushroom traits manually or provides a text description for LLM-based extraction
4. If structured traits are available, Numeric ML provides an auxiliary edible/poisonous signal
5. Safety logic combines CV confidence, species edibility mapping, and Numeric ML signal
6. NLP layer generates explanation and safety messaging
7. Result displayed with prediction badges, confidence scores, and educational disclaimer

**Screenshots demonstrating key functionality:**

![Full multimodal pipeline interface](docs/screenshots/CV_NLP_Numeric_Model_Input.png)
*Figure 1. Full multimodal evaluation: image + manual structured inputs + NLP explanation.*

![CV and NLP inference without structured numeric inputs](docs/screenshots/CV_NLP.png)
*Figure 2. CV and NLP inference without structured features — numeric model skipped.*

![Conservative safety handling under uncertain conditions](docs/screenshots/Description.png)
*Figure 3. Conservative safety handling: low-confidence case with safety block active.*

---

## 4. Execution Instructions

- Requirements: Python 3.10+, pip, GPU recommended for training (CPU sufficient for inference).

- Environment setup:

```bash
pip install -r requirements.txt
```

- API key setup (required for NLP):
  - An OpenAI API key is required for the NLP explanation and feature extraction components.
  - Set the key via the UI field in the Gradio app, or export it before running:

```bash
export OPENAI_API_KEY=your_key_here   # Linux/macOS
set OPENAI_API_KEY=your_key_here      # Windows
```

  - Without an API key the system still runs but falls back to a deterministic safety response (no LLM explanation).

- Data setup (for local training only):
  - Model artifacts are pre-trained and stored in `models/` — no download needed for inference.
  - To reproduce training, download the datasets from Kaggle and place them as follows:

| Dataset | Kaggle URL | Local path |
| --- | --- | --- |
| UCI Mushroom Classification | kaggle.com/datasets/uciml/mushroom-classification | `data/Mushroom data.csv` |
| Mushroom Edibility Classification | kaggle.com/datasets/devzohaib/mushroom-edibility-classification | `data/secondary_data.csv` |
| Edible & Poisonous Mushroom Classification | kaggle.com/datasets/benedictusjason/edible-and-poisonous-mushroom-classification | `data/train/`, `data/val/`, `data/test/` |
| Mushroom Species Recognition | kaggle.com/datasets/zlatan599/mushroom1 | `data/raw/mushroom_species_recognition/` |

- Training command(s):

```bash
python train_numeric.py
python train_cv.py
python train_species_cv.py
```

- Inference/run command(s):

```bash
python app.py
```

- Reproducibility notes:
  - Models are pre-trained and stored in the `models/` directory via Git LFS.
  - Large image datasets are excluded from the repository due to size; download links above.
  - The deployment on Hugging Face performs inference only and does not retrain models.

---

## 5. Optional Bonus Evidence

- [x] Third selected block implemented with strong quality
- [x] More than two data sources used with clear added value
- [x] Extended evaluation
- [x] Ethics, bias, or fairness analysis
- [ ] Creative or exceptional use case

Evidence for selected bonus items:
- Multimodal integration of CV, Numeric ML, and NLP in one coherent safety-oriented pipeline.
- Multiple external datasets used across structured and visual tasks.
- Extended evaluation with top-k CV metrics, prompt comparison, and low-confidence safety-case analysis.
- Ethics/safety focus through conservative decision logic, uncertainty communication, and explicit non-consumption guidance.
