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

### 1.2 Integration Logic

- How the selected blocks interact: CV predicts species and confidence; Numeric ML provides an auxiliary edible/poisonous signal when structured mushroom traits are available; NLP explains outputs and uncertainty; safety logic combines these signals conservatively.
- Data and output flow between blocks:
  1. User uploads mushroom image.
  2. Computer Vision predicts species and confidence.
  3. Structured mushroom traits (manual or LLM-extracted) optionally feed Numeric ML.
  4. Safety logic combines CV, Numeric ML (if available), and species mapping.
  5. NLP generates explanation and warning text.

See `run_assistant()` in [`app.py`](app.py#L604).

![Full multimodal pipeline interface](docs/screenshots/CV_NLP_Numeric_Model_Input.png)
*Figure 1. Full multimodal evaluation with manual structured inputs.*

---

## 2. Block Documentation

### 2A. ML Numeric Data (If selected)

#### 2A.1 Data Source(s)

| Entry | Source (file / location) | Type | Role in this block |
| --- | --- | --- | --- |
| 1 | `data/Mushroom data.csv` (UCI Mushroom Dataset) | Structured CSV (categorical) | Exploratory dataset used for early experiments, preprocessing validation and feature exploration. |
| 2 | `data/secondary_data.csv` (Extended Mushroom Edibility Dataset) | Structured CSV (numerical + categorical traits) | Final dataset used to train and evaluate the deployed Numeric ML pipeline (deployed Random Forest). |
| 3 | Runtime structured inputs | User-provided manual dropdowns and LLM-extracted JSON features (per inference) | Optional inference-time input source that enables the Numeric ML component when available. |

Data references: `data/Mushroom data.csv`, `data/secondary_data.csv`, and evaluation artifacts in `models/numeric_metrics.json`.

Note: `data/Mushroom data.csv` functioned primarily as an exploratory resource; `data/secondary_data.csv` is the structured dataset used in the final numeric pipeline. Runtime user inputs (manual traits or LLM-extracted traits) are distinct from both and are only used at inference time when provided.

#### 2A.2 Preprocessing and Features

- Cleaning steps: categorical normalization, alias mapping, feature ordering.
- Preprocessing steps: tabular feature encoding and pipeline-based transformation before model inference.
- Feature engineering and selection: structured mushroom traits such as cap shape/color, gill attributes, stem features, habitat, season, bruises/bleeding.

See `predict_numeric_sample()` in [`app.py`](app.py#L296).

#### 2A.3 Model Selection

- Models tested: Logistic Regression, Random Forest.
- Why these models were chosen: Logistic Regression as linear baseline; Random Forest for nonlinear interactions common in categorical mushroom traits.

#### 2A.4 Model Comparison and Iterations

| Iteration | Objective | Key changes | Models used | Main metric | Change vs previous |
| --- | --- | --- | --- | --- | --- |
| 1 | Establish baseline | Basic preprocessing and baseline classifier | Logistic Regression | Accuracy / F1 | Baseline reference |
| 2 | Improve nonlinear performance | Improved feature handling and tree ensemble | Random Forest | Accuracy / F1 | Stronger class separation |
| 3 | Integrate in multimodal pipeline | Optional runtime use with manual/LLM traits | Random Forest (deployed) | Safety-consistent auxiliary signal | Avoids over-trusting standalone numeric output |

#### 2A.5 Evaluation and Error Analysis

- Metrics used: accuracy, precision, recall, f1.
- Final results: accuracy=1.0, precision=1.0, recall=1.0, f1=1.0 on the available evaluation split.
- Error patterns and likely causes: near-perfect metrics suggest very separable data but also potential overfitting or limited generalization.

The Numeric ML component is therefore treated as a supportive signal within the multimodal system, not as a standalone safety-critical classifier.

Source: [`models/numeric_metrics.json`](models/numeric_metrics.json#L1)

#### 2A.6 Integration with Other Block(s)

- Inputs received from other block(s): optional structured traits extracted from user text via NLP; species context from CV pipeline state.
- Outputs provided to other block(s): edible/poisonous probability signal used by safety logic and summarized by NLP explanation output.

See `extract_mushroom_features()` in [`llm.py`](llm.py#L134), `generate_mushroom_advice()` in [`llm.py`](llm.py#L294), and `run_assistant()` in [`app.py`](app.py#L604).

### 2B. NLP (If selected)

#### 2B.1 Data Source(s)

| Entry | Source name or link | Type | Size | Role in this block |
| --- | --- | --- | --- |
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
| 1 | Initial explanation behavior | Baseline prompt style | Prompt A | Qualitative safety/readability review | Baseline |
| 2 | Improve consistency and safety language | Stricter instruction wording | Prompt B | Qualitative comparison across sample cases | More consistent and safety-focused output |
| 3 | Add feature extraction role | Structured JSON extraction prompt | Prompt B + extraction schema | Valid extraction fields + robust fallback | Enables optional Numeric ML input without manual-only dependency |

See `generate_mushroom_advice()` and prompt comparison in [`llm.py`](llm.py#L294), [`llm.py`](llm.py#L376).

#### 2B.5 Evaluation and Error Analysis

- Evaluation strategy: qualitative comparison of prompt variants, safety review, and manual scenario testing.
- Results: Prompt B produced more consistent and safety-focused explanations during qualitative evaluation.
- Error patterns and likely causes: possible hallucinations or overconfident phrasing mitigated by strict prompts and conservative fallback messaging.

#### 2B.6 Integration with Other Block(s)

- Inputs received from other block(s): CV prediction/confidence and optional Numeric ML probabilities.
- Outputs provided to other block(s): explanation text, uncertainty communication, and optional structured feature extraction feeding Numeric ML.

![CV and NLP inference without structured numeric inputs](docs/screenshots/CV_NLP.png)
*Figure 2. CV and NLP inference without structured numeric features.*

### 2C. Computer Vision (If selected)

### 2C.1 Data Source(s)

| Entry | Source (file / location) | Type | Role in this block |
| --- | --- | --- | --- |
| 1 | Binary edible/poisonous image dataset — local folders `data/train`, `data/val`, `data/test` | Image dataset (binary labels) | Used for initial binary CV training and baseline evaluation (edible vs poisonous). |
| 2 | Species-level image dataset — `data/species_images` and `data/raw/mushroom_species_recognition/merged_dataset` | Image dataset (fine-grained species folders) | Final species-recognition training and evaluation (multi-class, used for deployed species CV model). |
| 3 | Local curated mushroom images | Runtime evaluation images | Small curated set used for manual evaluation and deployment checks |

Data references: `data/species_images/`, `data/train/`, `data/val/`, `data/test/`, and evaluation artifacts in `models/species_cv_metrics.json` and `models/cv_metrics.json`.

Note: The CV blocks use locally stored image folders for training and evaluation; at inference, user-uploaded images are processed by the same CV models trained on the folders listed above.

#### 2C.2 Preprocessing and Augmentation

- Image preprocessing: resizing, normalization, tensor conversion.
- Augmentation strategy: random crops, horizontal flips, color jitter.

#### 2C.3 Model Selection

- Vision model(s) used: ResNet18 (deployed), EfficientNet-B0 (experimental comparison/iteration path).
- Why these model(s) were chosen: strong transfer-learning baseline with manageable compute and stable fine-tuning behavior.

#### 2C.4 Model Comparison and Iterations

| Iteration | Objective | Key changes | Model(s) used | Main metric | Change vs previous |
| --- | --- | --- | --- | --- | --- |
| 1 | Binary classification baseline | Edible vs poisonous setup | ResNet18 | Accuracy | Established baseline |
| 2 | Species-level expansion | Multi-class species task | ResNet18 / EfficientNet path | Top-1 | Improved granularity |
| 3 | Final fine-grained model | 169-class training with top-k evaluation | Final species CV model | Top-1/Top-3/Top-5 | Better practical retrieval via top-k |

See `predict_image()` in [`train_cv.py`](train_cv.py#L304).

#### 2C.5 Evaluation and Error Analysis

- Metrics and/or visual checks: top-1, top-3, top-5 accuracy; qualitative review of low-confidence examples.
- Final results:
  - top1_accuracy: 0.5513333333333333
  - top3_accuracy: 0.744
  - top5_accuracy: 0.8066666666666666
  - test_loss: 1.9632646945317587
- Error patterns and limitations: visually similar species, domain shift in user photos, and confidence sensitivity under clutter/lighting changes.

Top-k metrics are emphasized because visually similar species often appear among top candidates even when top-1 is uncertain.

Sources: [`models/species_cv_metrics.json`](models/species_cv_metrics.json#L1), [`models/cv_metrics.json`](models/cv_metrics.json#L1)

#### 2C.6 Integration with Other Block(s)

- Inputs received from other block(s): none required for base CV inference.
- Outputs provided to other block(s): species prediction and confidence feed safety logic and NLP explanation layer.

Safety mapping reference: [`species_info.py`](species_info.py#L1).

![Conservative safety handling under uncertain conditions](docs/screenshots/Description.png)
*Figure 3. Conservative safety handling under low-confidence conditions.*

---

## 3. Deployment

- Deployment URL: [Hugging Face Deployment](https://huggingface.co/spaces/zulliand/mushroom-ai-assistant)
- Main user flow: upload image -> CV species prediction -> optional structured traits (manual or LLM-extracted) -> optional Numeric ML signal -> safety logic -> NLP explanation.
- Screenshot or short demo: see Figures 1-3 above.

Guidance alignment: deployment is inference-only and usable through Gradio UI.

---

## 4. Execution Instructions

- Environment setup:

```bash
pip install -r requirements.txt
```

- Data setup:
  - Place required model artifacts in `models/`.
  - Keep datasets under `data/` when running training locally.

- Training command(s):

```bash
python train_cv.py
python train_species_cv.py
python train_numeric.py
```

- Inference/run command(s):

```bash
python app.py
```

- Reproducibility notes:
  - Models are pre-trained and stored in the `models/` directory.
  - Large datasets are excluded from the repository due to size limitations.
  - The deployment performs inference only and does not retrain models.

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
