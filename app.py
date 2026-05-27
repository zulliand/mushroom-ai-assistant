"""Gradio application for the integrated Mushroom AI Assistant."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, Tuple

import gradio as gr
import joblib
import pandas as pd

from llm import compare_prompt_strategies, generate_mushroom_advice
from train_cv import predict_image
from species_info import get_species_info
from utils import (
    CV_MODEL_PATH,
    NUMERIC_METRICS_PATH,
    NUMERIC_PIPELINE_PATH,
    setup_logging,
    SPECIES_CV_MODEL_PATH,
)

LOGGER = logging.getLogger("mushroom.app")
EXAMPLES_DIR = Path(__file__).resolve().parent / "data" / "examples"

DEFAULT_FEATURE_TEMPLATE = {
    "cap-diameter": 15.0,
    "cap-shape": "x",
    "cap-surface": "g",
    "cap-color": "o",
    "does-bruise-or-bleed": "f",
    "gill-attachment": "e",
    "gill-spacing": "",
    "gill-color": "w",
    "stem-height": 17.0,
    "stem-width": 17.0,
    "stem-root": "s",
    "stem-surface": "y",
    "stem-color": "w",
    "veil-type": "u",
    "veil-color": "w",
    "has-ring": "t",
    "ring-type": "p",
    "spore-print-color": "",
    "habitat": "d",
    "season": "w",
}

DEFAULT_FEATURE_JSON = json.dumps(DEFAULT_FEATURE_TEMPLATE, indent=2)

NUMERIC_FEATURE_ORDER = list(DEFAULT_FEATURE_TEMPLATE.keys())
NUMERIC_CLASS_LABELS = {0: "edible", 1: "poisonous"}
NUMERIC_VALUE_COLUMNS = {"cap-diameter", "stem-height", "stem-width"}
FEATURE_ALIASES = {
    "cap_diameter": "cap-diameter",
    "cap_shape": "cap-shape",
    "cap_surface": "cap-surface",
    "cap_color": "cap-color",
    "does_bruise_or_bleed": "does-bruise-or-bleed",
    "gill_attachment": "gill-attachment",
    "gill_spacing": "gill-spacing",
    "gill_color": "gill-color",
    "stem_height": "stem-height",
    "stem_width": "stem-width",
    "stem_root": "stem-root",
    "stem_surface": "stem-surface",
    "stem_color": "stem-color",
    "veil_type": "veil-type",
    "veil_color": "veil-color",
    "has_ring": "has-ring",
    "ring_type": "ring-type",
    "spore_print_color": "spore-print-color",
    "habitat": "habitat",
    "season": "season",
}


def get_openai_api_key(fallback: str = "") -> str:
    """Read the OpenAI key from the environment or an optional UI override."""

    return os.getenv("OPENAI_API_KEY", fallback).strip()


def parse_feature_payload(feature_text: str) -> Dict[str, object]:
    """Parse a JSON payload describing mushroom features."""

    if not feature_text.strip():
        return {}
    payload = json.loads(feature_text)
    if not isinstance(payload, dict):
        raise ValueError("Structured mushroom features must be a JSON object.")
    return payload


def normalize_feature_names(features: Dict[str, object]) -> Dict[str, object]:
    """Normalize a few common schema aliases to the trained model's feature names."""

    normalized = dict(features)
    for source_name, target_name in FEATURE_ALIASES.items():
        if source_name in normalized and target_name not in normalized:
            normalized[target_name] = normalized.pop(source_name)
    return normalized


def load_numeric_pipeline() -> Any:
    """Load the trained numeric pipeline artifact."""

    if not NUMERIC_PIPELINE_PATH.exists():
        raise FileNotFoundError(f"Numeric pipeline not found at {NUMERIC_PIPELINE_PATH}. Train the numeric model first.")
    return joblib.load(NUMERIC_PIPELINE_PATH)


def load_numeric_metadata() -> Dict[str, Any]:
    """Load numeric metrics metadata if it exists."""

    if not NUMERIC_METRICS_PATH.exists():
        return {}
    return json.loads(NUMERIC_METRICS_PATH.read_text(encoding="utf-8"))


def build_numeric_frame(feature_json: str) -> pd.DataFrame:
    """Normalize the JSON payload into a single-row dataframe with all required columns."""

    features = normalize_feature_names(parse_feature_payload(feature_json))
    row = {column: features.get(column, DEFAULT_FEATURE_TEMPLATE[column]) for column in NUMERIC_FEATURE_ORDER}
    frame = pd.DataFrame([row], columns=NUMERIC_FEATURE_ORDER)

    # Coerce expected numeric features before model inference.
    for column in NUMERIC_VALUE_COLUMNS:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    return frame


def is_default_numeric_input(feature_json: str) -> bool:
    """Check whether the structured input only contains internal fallback defaults."""

    normalized_features = normalize_feature_names(parse_feature_payload(feature_json))
    if not normalized_features:
        return True

    for column in NUMERIC_FEATURE_ORDER:
        default_value = DEFAULT_FEATURE_TEMPLATE[column]
        current_value = normalized_features.get(column, default_value)
        if column in NUMERIC_VALUE_COLUMNS:
            try:
                if float(current_value) != float(default_value):
                    return False
            except (TypeError, ValueError):
                return False
        elif str(current_value) != str(default_value):
            return False

    return True


def predict_numeric_sample(feature_json: str) -> Dict[str, object]:
    """Run inference through the saved numeric pipeline."""

    pipeline = load_numeric_pipeline()
    metadata = load_numeric_metadata()
    frame = build_numeric_frame(feature_json)
    default_input = is_default_numeric_input(feature_json)
    prediction = int(pipeline.predict(frame)[0])
    probabilities = pipeline.predict_proba(frame)[0] if hasattr(pipeline, "predict_proba") else None

    class_labels = metadata.get("label_classes") if isinstance(metadata.get("label_classes"), list) else None

    predicted_label = NUMERIC_CLASS_LABELS.get(prediction, str(prediction))
    if class_labels and 0 <= prediction < len(class_labels):
        raw_label = str(class_labels[prediction]).strip().lower()
        if raw_label in {"e", "edible", "1", "true"}:
            predicted_label = "edible"
        elif raw_label in {"p", "poisonous", "0", "false"}:
            predicted_label = "poisonous"
        else:
            predicted_label = raw_label

    edible_probability = None
    if probabilities is not None:
        effective_labels = class_labels or ["e", "p"]
        edible_index = 0 if effective_labels and str(effective_labels[0]).lower() in {"e", "edible"} else None
        if edible_index is None and len(probabilities) > 1:
            edible_index = 1
        if edible_index is not None:
            edible_probability = float(probabilities[edible_index])

    return {
        "predicted_label": predicted_label,
        "edible_probability": edible_probability,
        "probabilities": None if probabilities is None else probabilities.tolist(),
        "features": frame.iloc[0].to_dict(),
        "feature_source": "default-feature fallback" if default_input else "user-provided structured features",
        "default_features_used": default_input,
    }


def _shorten_text(text: object, max_length: int = 220) -> str:
    """Trim long text for the compact summary view."""

    if text is None:
        return "—"
    value = str(text).strip()
    if not value:
        return "—"
    if len(value) <= max_length:
        return value
    return value[: max_length - 1].rstrip() + "…"


def run_assistant(
    image_path: str,
    description_text: str,
    feature_json: str,
    api_key: str,
    openai_model: str,
) -> Tuple[str, Dict[str, object], Dict[str, object], Dict[str, object], Dict[str, object], Dict[str, object]]:
    """Run the three-stage assistant and return user-facing outputs."""

    if not image_path:
        raise ValueError("Please upload a mushroom image before running the assistant.")

    _ = description_text

    # Prefer species-level model if available, otherwise fall back to binary CV model
    if SPECIES_CV_MODEL_PATH.exists():
        model_path = SPECIES_CV_MODEL_PATH
        # request top-3 predictions for species model
        cv_result = predict_image(image_path, model_path, top_k=3)
    else:
        model_path = CV_MODEL_PATH
        cv_result = predict_image(image_path, model_path, top_k=1)

    top_k_list = []
    main_pred = None
    main_conf = None
    if isinstance(cv_result, dict) and "top_k" in cv_result:
        top_k_list = cv_result["top_k"]
        if top_k_list:
            main_pred = top_k_list[0]["class"]
            main_conf = float(top_k_list[0].get("confidence", 0.0))
    else:
        main_pred = cv_result.get("predicted_class")
        main_conf = float(cv_result.get("confidence")) if cv_result.get("confidence") is not None else None

    cv_result_for_llm = dict(cv_result) if isinstance(cv_result, dict) else {}
    if main_pred is not None:
        cv_result_for_llm["predicted_class"] = main_pred
        cv_result_for_llm["species"] = main_pred
    if main_conf is not None:
        cv_result_for_llm["confidence"] = main_conf
    if top_k_list:
        cv_result_for_llm["top_k"] = top_k_list

    numeric_result = predict_numeric_sample(feature_json)
    numeric_default_based = bool(numeric_result.get("default_features_used"))
    # Species metadata (common name, edibility, cookable)
    species_meta = get_species_info(main_pred) if main_pred else None
    cv_edibility = None
    if species_meta:
        cv_edibility = species_meta.get("edibility")

    # Determine whether CV expresses an edibility claim (from species metadata or binary CV)
    cv_claim = None
    if species_meta and species_meta.get("edibility"):
        cv_claim = str(species_meta.get("edibility")).lower()
    else:
        cv_label = cv_result_for_llm.get("predicted_class") if isinstance(cv_result_for_llm, dict) else None
        if cv_label:
            cv_label_lower = str(cv_label).lower()
            if cv_label_lower in {"poisonous", "poison", "toxic"}:
                cv_claim = "poisonous"
            elif cv_label_lower in {"edible", "e"}:
                cv_claim = "edible"

    numeric_claim = str(numeric_result.get("predicted_label")).lower() if numeric_result.get("predicted_label") is not None else None
    species_mapping_claim = str(cv_edibility).lower() if cv_edibility else None

    # If the numeric model is only seeing fallback defaults, it stays auxiliary.
    conflict = False
    if species_mapping_claim is not None and numeric_claim is not None and not numeric_default_based:
        conflict = species_mapping_claim != numeric_claim
    elif cv_claim is not None and numeric_claim is not None and not numeric_default_based:
        conflict = cv_claim != numeric_claim

    if isinstance(cv_result_for_llm, dict):
        cv_result_for_llm["common_name"] = species_meta.get("common_name") if species_meta else None
        cv_result_for_llm["cv_edibility"] = cv_edibility
        cv_result_for_llm["conflict_detected"] = conflict

    advice = generate_mushroom_advice(cv_result_for_llm, numeric_result, api_key=api_key, model=openai_model, prompt_variant="B")
    prompt_comparison = compare_prompt_strategies(cv_result_for_llm, numeric_result, api_key=api_key, model=openai_model)

    signals = prompt_comparison["signals"]
    high_trust = signals["allow_cooking"]

    # Build a friendly summary using species metadata when available
    summary_lines = ["## Mushroom AI Assistant Results\n"]

    if main_pred:
        summary_lines.append(f"**CV prediction:** {main_pred} ({(main_conf or 0):.2%})\n")
    else:
        summary_lines.append("**CV prediction:** (no prediction)\n")

    summary_lines.append(f"**Confidence:** {(main_conf if main_conf is not None else 0):.2%}\n" if main_conf is not None else "**Confidence:** —\n")

    # Top-3 display
    if top_k_list:
        top3_text = ", ".join([f"{p['class']} ({p['confidence']:.2%})" for p in top_k_list[:3]])
        summary_lines.append(f"**Top-3 species:** {top3_text}\n")

    if species_meta:
        summary_lines.append(f"**Common name:** {species_meta.get('common_name')}\n")
        summary_lines.append(f"**Species mapping:** {cv_edibility}\n")
        summary_lines.append(f"**Cookable:** {species_meta.get('cookable')}\n")

    numeric_line = f"**Numeric prediction:** {numeric_result['predicted_label']}"
    if numeric_result.get("edible_probability") is not None:
        numeric_line += f" ({numeric_result['edible_probability']:.2%} edible probability)"
    if numeric_default_based:
        numeric_line += " - auxiliary only (default features)"
    summary_lines.append(numeric_line)

    summary = "\n".join(summary_lines)

    if species_mapping_claim == "poisonous":
        final_safety_decision = "Poisonous species detected"
        consumption_advice = "withheld for safety"
    elif species_mapping_claim == "edible":
        final_safety_decision = "Likely edible species detected"
        consumption_advice = "withheld for safety"
    elif conflict and cv_claim == "poisonous" and numeric_claim == "edible":
        final_safety_decision = "Consumption advice withheld for safety"
        consumption_advice = "withheld for safety"
    else:
        CONFIDENCE_THRESHOLD = 0.75
        if high_trust and main_conf is not None and main_conf >= CONFIDENCE_THRESHOLD and not conflict:
            final_safety_decision = "Consumption advice withheld for safety"
            consumption_advice = "withheld for safety"
        else:
            final_safety_decision = "Consumption advice withheld for safety"
            consumption_advice = "withheld for safety"

    summary += (
        "\n\n**Final safety decision:** " + final_safety_decision +
        f"\n**Consumption advice:** {consumption_advice}" +
        f"\n**LLM explanation:** {_shorten_text(advice.get('explanation'))}"
    )

    return summary, cv_result, numeric_result, advice, prompt_comparison, numeric_result.get("features", {})


def build_interface() -> gr.Blocks:
    """Create the Gradio UI."""

    setup_logging()
    css = """
    .gradio-container {
        background: linear-gradient(180deg, #f6f3ec 0%, #f9faf7 48%, #ffffff 100%);
    }
    .app-shell {
        max-width: 1300px;
        margin: 0 auto;
    }
    .hero-card {
        padding: 1.2rem 1.35rem;
        border-radius: 22px;
        background: linear-gradient(135deg, rgba(255,255,255,0.92), rgba(245,240,228,0.92));
        border: 1px solid rgba(85, 74, 56, 0.12);
        box-shadow: 0 12px 30px rgba(54, 45, 28, 0.08);
        margin-bottom: 1rem;
    }
    .summary-card {
        padding: 1rem 1.1rem;
        border-radius: 18px;
        background: rgba(255, 255, 255, 0.86);
        border: 1px solid rgba(85, 74, 56, 0.12);
        box-shadow: 0 10px 24px rgba(54, 45, 28, 0.07);
        font-size: 1.02rem;
        line-height: 1.6;
    }
    .summary-card h3 {
        margin-top: 0;
        margin-bottom: 0.7rem;
    }
    .summary-card strong {
        color: #2f281d;
    }
    .compact-accordion {
        margin-top: 0.7rem;
    }
    """
    example_images = [str(path) for path in sorted(EXAMPLES_DIR.glob("*")) if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}]

    with gr.Blocks(theme=gr.themes.Soft(), title="Mushroom AI Assistant", css=css) as demo:
        with gr.Column(elem_classes=["app-shell"]):
            gr.Markdown(
                "<div class='hero-card'>"
                "<h1>Mushroom AI Assistant</h1>"
                "<p>Upload a mushroom photo, add a short description if you want, and review a concise safety summary with expandable technical details.</p>"
                "</div>"
            )

            with gr.Row(equal_height=True):
                with gr.Column(scale=1, min_width=360):
                    image_input = gr.Image(
                        type="filepath",
                        label="Upload mushroom photo",
                        sources=["upload"],
                    )
                    description_input = gr.Textbox(
                        label="Optional mushroom description",
                        placeholder="Example: Red cap with white spots, white stem, growing in the woods.",
                        lines=3,
                    )
                    if example_images:
                        gr.Examples(
                            examples=example_images[:4],
                            inputs=image_input,
                            label="Example mushrooms",
                            examples_per_page=4,
                        )

                    with gr.Accordion("Advanced settings", open=False):
                        api_key_input = gr.Textbox(label="OpenAI API key override", type="password", value="")
                        model_input = gr.Textbox(label="OpenAI model", value="gpt-4o-mini")

                    run_button = gr.Button("Analyze mushroom", variant="primary")

                with gr.Column(scale=1, min_width=420):
                    summary_output = gr.Markdown(elem_classes=["summary-card"])

                    with gr.Accordion("Computer vision output", open=False, elem_classes=["compact-accordion"]):
                        cv_output = gr.JSON(label="Computer vision output", show_label=False)

                    with gr.Accordion("Numeric model output", open=False, elem_classes=["compact-accordion"]):
                        numeric_output = gr.JSON(label="Numeric model output", show_label=False)

                    with gr.Accordion("LLM output", open=False, elem_classes=["compact-accordion"]):
                        llm_output = gr.JSON(label="LLM output", show_label=False)

                    with gr.Accordion("Prompt comparison", open=False, elem_classes=["compact-accordion"]):
                        comparison_output = gr.JSON(label="Prompt comparison", show_label=False)

                    with gr.Accordion("Structured features JSON", open=False, elem_classes=["compact-accordion"]):
                        features_output = gr.JSON(label="Structured features JSON", show_label=False)

        def submit(image_path: str, description_text: str, api_key_override: str, openai_model: str):
            combined_key = get_openai_api_key(api_key_override)
            return run_assistant(image_path, description_text, DEFAULT_FEATURE_JSON, combined_key, openai_model)

        run_button.click(
            fn=submit,
            inputs=[image_input, description_input, api_key_input, model_input],
            outputs=[summary_output, cv_output, numeric_output, llm_output, comparison_output, features_output],
        )
    return demo


def main() -> None:
    """Launch the Gradio app."""

    demo = build_interface()
    demo.launch(server_name="0.0.0.0", server_port=int(os.getenv("PORT", "7860")))


if __name__ == "__main__":
    main()
