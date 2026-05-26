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

- ResNet18 transfer learning is used.
- The backbone is frozen when pretrained weights are available.

- **Iteration 1 — binary CV baseline:** a binary edible/poisonous image classifier to validate pipeline and inference integration.
- **Iteration 2 — 46-class species model:** expanded species labels to a mid-sized taxonomy to validate multi-class training and data pipeline.

Fine-grained species recognition is inherently ambiguous from single images: many mushroom species share visual traits and intra-class variability (growth stages, lighting, occlusion). Top-3 and top-5 accuracy measure whether the correct species appears among the top candidates returned by the model — a practical and useful signal for downstream human-in-the-loop workflows and LLM-guided explanations.

### Error analysis and safety rationale

- **Visual similarity:** Several species are visually similar (color variants, subtle cap/gill differences) which reduces single-image top-1 accuracy even when the model retains useful information in higher-ranked predictions.
- **Domain shift:** Field photos from users or external sources often differ from the training distribution (camera, lighting, background), which can degrade performance.

- CV: `Amanita_muscaria` detected with high confidence (e.g., 79.5%).
- Numeric: structured features converted from CSV indicate `edible` with 76.0% probability.

- The app expects `models/mushroom_rf.pkl` and `models/mushroom_cv.pt` to exist.
- OpenAI is optional at runtime; if no key is available, the app falls back to a deterministic safety-focused response.

 ## Project Metadata

 - Project title: Mushroom AI Assistant
 - Student: Andre (repository owner)
 - GitHub repository URL: (replace with repository URL)
 - Deployment URL: (replace with deployment URL if available)
 - Submission date: 2026-05-26

 ### Mandatory Setup Checks

 - [x] At least 2 blocks selected
 - [x] Multiple and different data sources used
 - [ ] Deployment URL provided
 - [ ] Required GitHub users added to repository (`jasminh`, `bkuehnis`)

 ## Selected AI Blocks

 - [x] ML Numeric Data
 - [x] NLP
 - [x] Computer Vision

 Primary blocks used for core solution:
 - Primary block 1: Computer Vision (species recognition)
 - Primary block 2: ML Numeric Data (structured edible/poisonous prediction)


 ---

 ## 1. Project Foundation (Short)

 ### 1.1 Problem Definition
 - Problem statement: Provide an integrated assessment of mushroom safety by combining image-based species recognition, structured-feature-based edible/poisonous prediction, and an LLM explanation/safety layer.
 - Goal: Produce explainable, safety-first advice that avoids recommending cooking when models disagree or confidence is low.
 - Success criteria: Working Gradio app (`app.py`) that accepts an image and structured JSON features, returns CV predictions (top-k), numeric prediction, LLM explanation and a safety decision; top-1/top-3/top-5 metrics reported for CV model.

 ### 1.2 Integration Logic
 - How the selected blocks interact: The Gradio app calls `run_assistant()` in [`app.py`](app.py) which: (1) runs CV inference (`train_cv.predict_image`), (2) runs numeric inference (`predict_numeric_sample` in `app.py` wrapping the numeric pipeline), and (3) calls the LLM (`llm.generate_mushroom_advice`) to combine signals and produce an explanation and safety advice.
 - Data and output flow between blocks: Uploaded image -> CV model -> species/top-k list. Structured JSON -> numeric pipeline -> edible probability. Both outputs -> LLM -> final summary and safety decision.

 ---

 ## 2. Block Documentation

 ### 2A. ML Numeric Data (Selected)

 #### 2A.1 Data Source(s
 | Entry | Source name or link | Type | Size | Role in this block |
 | --- | --- | --- | --- | --- |
 | 1 | `data/Mushroom data.csv` (UCI Mushroom Classification) | CSV (tabular) | small (~8K rows) | Training and evaluation of the numeric edible/poisonous classifier |

 #### 2A.2 Preprocessing and Features
 - Cleaning steps: categorical normalization and alias mapping (see `app.py` `normalize_feature_names`).
 - Preprocessing steps: one-hot / ordinal encoding as appropriate inside `train_numeric.py` preprocessing pipeline.
 - Feature engineering and selection: use provided mushroom attributes (cap_shape, odor, gill_color, habitat, etc.) in `NUMERIC_FEATURE_ORDER` within `app.py` and `train_numeric.py`.

 #### 2A.3 Model Selection
 - Models tested: Logistic Regression, Random Forest (see `train_numeric.py`).
 - Why these models were chosen: reliable baselines for tabular data, interpretable probabilities for the safety layer, and fast training for classroom experimentation.

 #### 2A.4 Model Comparison and Iterations
 | Iteration | Objective | Key changes | Models used | Main metric | Change vs previous |
 | --- | --- | --- | --- | --- | --- |
 | 1 | Baseline numeric classifier | Basic preprocessing, logistic regression | LogisticRegression | F1 / accuracy | Baseline |
 | 2 | Improve robustness | Random Forest tested and tuned | RandomForest | F1 / accuracy | Improved recall/precision tradeoff |

 #### 2A.5 Evaluation and Error Analysis
 - Metrics used: accuracy, F1, class probabilities (used as edible probability for safety decisions).
 - Final results: numeric model provides an edible probability used for safety comparisons (see `models/numeric_metrics.json`).
 - Error patterns and likely causes: structured features may be missing or noisy; structured model does not inspect images and can therefore disagree with CV when features are ambiguous or mis-entered.

 #### 2A.6 Integration with Other Block(s)
 - Inputs received from other block(s): none (numeric model only consumes structured JSON features provided by the user/UI).
 - Outputs provided to other block(s: The numeric edible probability and predicted label are passed to the LLM and to the safety decision logic in `app.py`.

 ---

 ### 2B. NLP (Selected)

 #### 2B.1 Data Source(s)
 | Entry | Source name or link | Type | Size | Role in this block |
 | --- | --- | --- | --- | --- |
 | 1 | OpenAI API (runtime) | External API | N/A | Generates explanations, safety reasoning and cooking suggestions |

 #### 2B.2 Preprocessing and Prompt Design
 - Text preprocessing: minimal — the LLM receives formatted text describing CV and numeric outputs.
 - Prompt design: see `llm.py` for prompt templates and the two prompt variants compared by `compare_prompt_strategies()`; prompts include CV top-k results, numeric edible probability and safety context.

 #### 2B.3 Approach Selection
 - Approach used: Prompt engineering with OpenAI models (small, controlled prompts) and an internal prompt-variant comparator implemented in `llm.py`.
 - Alternatives considered: RAG or retrieval-based augmentation (not required for this project scope).

 #### 2B.4 Comparison and Iterations
 | Iteration | Objective | Key changes | Model or prompt setup | Main metric or qualitative check | Change vs previous |
 | --- | --- | --- | --- | --- | --- |
 | 1 | Minimal safe explanation | Basic prompt with CV+numeric summary | OpenAI prompt A | Qualitative review | Baseline |
 | 2 | Improve clarity & safety | Add structured safety phrasing and context | OpenAI prompt B | Qualitative review & manual checks | Improved clarity |

 #### 2B.5 Evaluation and Error Analysis
 - Evaluation strategy: manual review of examples to ensure safe phrasing and correct inclusion of model outputs; optional A/B via `compare_prompt_strategies()`.
 - Results: LLM explanations are concise and include safety warnings when models disagree.
 - Error patterns: LLM may hallucinate extra facts if prompts are not tightly constrained; mitigated by including only model outputs and explicit instructions in the prompt templates (`llm.py`).

 #### 2B.6 Integration with Other Block(s)
 - Inputs received from other block(s): CV top-k results and numeric edible probability.
 - Outputs provided to other block(s): textual explanation, safety warning, and optional cooking suggestions (blocked when safety logic disables them).

 ---

 ### 2C. Computer Vision (Selected)

 #### 2C.1 Data Source(s
 | Entry | Source name or link | Type | Size | Role in this block |
 | --- | --- | --- | --- | --- |
 | 1 | Edible & Poisonous Mushroom Classification Dataset | Image folders | (project-local) | Initial binary CV baseline (edible/poisonous) |
 | 2 | Kaggle Mushroom species recognition dataset (merged) | Image folders | large (~720k images raw, filtered & limited) | Final species recognition training (169 classes; data limited for experiments) |
 | 3 | Local curated test images (user uploaded) | Image files | small | Runtime inference testing in `app.py` |

 #### 2C.2 Preprocessing and Augmentation
 - Image preprocessing: resize, normalization compatible with torchvision pretrained models (see `train_cv.py` building transforms).
 - Augmentation strategy: typical random crops, flips, color jitter used during training to improve generalization; defined in `train_cv.build_dataloaders()`.

 #### 2C.3 Model Selection
 - Vision model(s) used: `resnet18` (final), optional `efficientnet_b0` for experiments (see `train_cv.py` and `train_species_cv.py`).
 - Why these model(s) were chosen: lightweight, well-known transfer-learning backbones, good tradeoff between performance and resource requirements for classroom runs.

 #### 2C.4 Model Comparison and Iterations
 | Iteration | Objective | Key changes | Model(s) used | Main metric | Change vs previous |
 | --- | --- | --- | --- | --- | --- |
 | 1 | Binary baseline | Train edible/poisonous classifier | ResNet18 | accuracy | Baseline |
 | 2 | Mid-sized species model | 46-class species classification | ResNet18 | top-1/top-3 | Improved multi-class handling |
 | 3 | Final species model | 169-class training, top-k evaluation | ResNet18 | top-1/top-3/top-5 | Final reported metrics (see below) |

 #### 2C.5 Evaluation and Error Analysis
 - Metrics and/or visual checks: top-1, top-3, top-5 accuracies computed in `train_cv.evaluate()` and saved to `models/species_cv_metrics.json`.
 - Final results (reported):
	 - Top-1 accuracy (test): 53.45%
	 - Top-3 accuracy (test): 74.40%
	 - Top-5 accuracy (test): 82.17%
 - Error patterns and limitations: visual confusion between similar species, domain shift for user images, and limited per-class samples for some rare species.

 #### 2C.6 Integration with Other Block(s)
 - Inputs received from other block(s): none; CV operates on user-uploaded image.
 - Outputs provided to other block(s): top-k species predictions and confidences passed to `llm.py` and safety logic in `app.py`.

 ---

 ## 3. Deployment

 - Deployment URL: (replace with deployed Space URL)
 - Main user flow: user opens the Gradio app, uploads an image and optionally fills structured JSON features; app returns CV top-k, numeric prediction, LLM explanation and safety decision.
 - Screenshot or short demo: (attach screenshots in repository or add link to hosted demo)

 ---

 ## 4. Execution Instructions

 - Environment setup:

 ```bash
 python -m venv .venv
 .\\.venv\\Scripts\\Activate.ps1  # PowerShell
 pip install -r requirements.txt
 ```

 - Data setup:
	 - Place the UCI CSV at `data/Mushroom data.csv`.
	 - Provide image datasets under `data/images/` or follow the species import pipeline (`build_species_csv_import.py`) to create `data/species_images/`.

 - Training command(s) (do not retrain unless you intend to):

 ```bash
 # Numeric model
 py train_numeric.py --data-path data/Mushroom\\ data.csv

 # CV species training (example; heavy compute)
 py train_species_cv.py --model-name resnet18 --epochs 15 --max-train-samples 30000 --max-val-samples 6000 --max-test-samples 6000
 ```

 - Inference/run command(s):

 ```bash
 py app.py
 ```

 - Reproducibility notes:
	 - Checkpoints and metrics are saved under `models/` (see `models/species_cv_metrics.json` and `models/mushroom_species_cv_metadata.json`).
	 - Do not commit large datasets — see `.gitignore` for excluded paths.

 ---

 ## 5. Optional Bonus Evidence

 - [x] Third selected block implemented with strong quality (LLM-based explanations and safety layer)
 - [x] More than two data sources used with clear added value
 - [ ] A core section is done exceptionally well
 - [ ] Extended evaluation
 - [ ] Ethics, bias, or fairness analysis
 - [ ] Creative or exceptional use case

 Evidence for selected bonus items: combined LLM safety logic, top-k CV reporting, and numeric/CV disagreement handling implemented in `app.py` and `llm.py`.
