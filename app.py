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

DEFAULT_FEATURE_TEMPLATE = {
    "cap_shape": "x",
    "cap_surface": "s",
    "cap_color": "n",
    "bruises": "t",
    "odor": "n",
    "gill_attachment": "f",
    "gill_spacing": "c",
    "gill_size": "b",
    "gill_color": "k",
    "stalkshape": "e",
    "stalk_root": "b",
    "stalk_surface_above_ring": "s",
    "stalk_surface_below_ring": "s",
    "stalk_color_above_ring": "w",
    "stalk_color_below_ring": "w",
    "veil_type": "p",
    "veil_color": "w",
    "ring_number": "o",
    "ring_type": "p",
    "spore_print_color": "k",
    "population": "s",
    "habitat": "u",
}

NUMERIC_FEATURE_ORDER = list(DEFAULT_FEATURE_TEMPLATE.keys())
NUMERIC_CLASS_LABELS = {0: "edible", 1: "poisonous"}
FEATURE_ALIASES = {"stalk_shape": "stalkshape"}


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
    return pd.DataFrame([row], columns=NUMERIC_FEATURE_ORDER)


def predict_numeric_sample(feature_json: str) -> Dict[str, object]:
    """Run inference through the saved numeric pipeline."""

    pipeline = load_numeric_pipeline()
    metadata = load_numeric_metadata()
    frame = build_numeric_frame(feature_json)
    prediction = int(pipeline.predict(frame)[0])
    probabilities = pipeline.predict_proba(frame)[0] if hasattr(pipeline, "predict_proba") else None

    edible_probability = None
    if probabilities is not None:
        class_labels = metadata.get("label_classes") or ["e", "p"]
        edible_index = 0 if class_labels and str(class_labels[0]).lower() in {"e", "edible"} else None
        if edible_index is None and len(probabilities) > 1:
            edible_index = 1
        if edible_index is not None:
            edible_probability = float(probabilities[edible_index])

    predicted_label = NUMERIC_CLASS_LABELS.get(prediction, str(prediction))
    return {
        "predicted_label": predicted_label,
        "edible_probability": edible_probability,
        "probabilities": None if probabilities is None else probabilities.tolist(),
        "features": frame.iloc[0].to_dict(),
    }


def run_assistant(image_path: str, feature_json: str, api_key: str, openai_model: str) -> Tuple[str, Dict[str, object], Dict[str, object], Dict[str, object], Dict[str, object]]:
    """Run the three-stage assistant and return user-facing outputs."""

    if not image_path:
        raise ValueError("Please upload a mushroom image before running the assistant.")

    # Prefer species-level model if available, otherwise fall back to binary CV model
    if SPECIES_CV_MODEL_PATH.exists():
        model_path = SPECIES_CV_MODEL_PATH
        # request top-3 predictions for species model
        cv_result = predict_image(image_path, model_path, top_k=3)
    else:
        model_path = CV_MODEL_PATH
        cv_result = predict_image(image_path, model_path, top_k=1)

    numeric_result = predict_numeric_sample(feature_json)
    advice = generate_mushroom_advice(cv_result, numeric_result, api_key=api_key, model=openai_model, prompt_variant="B")
    prompt_comparison = compare_prompt_strategies(cv_result, numeric_result, api_key=api_key, model=openai_model)

    signals = prompt_comparison["signals"]
    high_trust = signals["allow_cooking"]

    # Build a friendly summary using species metadata when available
    summary_lines = ["## Mushroom AI Assistant Results\n"]

    # Extract prediction info (support top-k and single outputs)
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

    if main_pred:
        summary_lines.append(f"**CV prediction:** {main_pred} ({(main_conf or 0):.2%})\n")
    else:
        summary_lines.append("**CV prediction:** (no prediction)\n")

    # Top-3 display
    if top_k_list:
        top3_text = ", ".join([f"{p['class']} ({p['confidence']:.2%})" for p in top_k_list[:3]])
        summary_lines.append(f"**Top-3 species:** {top3_text}\n")

    # Species metadata (common name, edibility, cookable)
    species_meta = get_species_info(main_pred) if main_pred else None
    if species_meta:
        summary_lines.append(f"**Common name:** {species_meta.get('common_name')}\n")
        summary_lines.append(f"**Edibility:** {species_meta.get('edibility')}\n")
        summary_lines.append(f"**Cookable:** {species_meta.get('cookable')}\n")

    summary_lines.append(f"**Numeric prediction:** {numeric_result['predicted_label']}")
    if numeric_result.get("edible_probability") is not None:
        summary_lines[-1] += f" ({numeric_result['edible_probability']:.2%} edible probability)"

    summary = "\n".join(summary_lines)

    # Determine whether CV expresses an edibility claim (from species metadata or binary CV)
    cv_claim = None
    if species_meta and species_meta.get("edibility"):
        cv_claim = str(species_meta.get("edibility")).lower()
    else:
        cv_label = cv_result.get("predicted_class") if isinstance(cv_result, dict) else None
        if cv_label:
            cv_label_lower = str(cv_label).lower()
            if cv_label_lower in {"poisonous", "poison", "toxic"}:
                cv_claim = "poisonous"
            elif cv_label_lower in {"edible", "e"}:
                cv_claim = "edible"

    numeric_claim = str(numeric_result.get("predicted_label")).lower() if numeric_result.get("predicted_label") is not None else None

    # If CV and numeric disagree (CV says poisonous and numeric says edible), show a clear conflict block
    conflict = cv_claim is not None and numeric_claim is not None and cv_claim != numeric_claim
    if conflict and cv_claim == "poisonous" and numeric_claim == "edible":
        summary += (
            "\n\n**Conflict detected:**\n"
            "Computer vision predicts poisonous species.\n"
            f"CV prediction: {main_pred} — {(main_conf or 0):.1%}\n"
            "Structured feature model predicts edible.\n"
            f"Structured-model estimate: {numeric_result.get('edible_probability'):.1%} edible (from CSV model, not the image)\n\n"
            "**Result withheld due to disagreement.**\n"
        )
    else:
        summary += (
            f"\n\n**Decision logic:** {'High-trust path enabled' if high_trust else 'High-trust path blocked'}"
            f"\n\n**Prompt strategy chosen:** {advice.get('prompt_variant', 'B')}"
        )

    summary += (
        f"\n\n**LLM explanation:** {advice.get('explanation')}\n\n"
        f"**Safety warning:** {advice.get('safety_warning')}\n\n"
    )

    # Only include cooking suggestions if safety logic allows and confidence threshold passed
    CONFIDENCE_THRESHOLD = 0.75
    if high_trust and main_conf is not None and main_conf >= CONFIDENCE_THRESHOLD:
        summary += f"**Cooking suggestions:** {advice.get('cooking_suggestions')}\n\n"
    else:
        summary += "**Cooking suggestions:** (not shown due to low confidence or safety)\n\n"

    summary += f"**Disclaimer:** {advice.get('disclaimer')}"

    return summary, cv_result, numeric_result, advice, prompt_comparison


def build_interface() -> gr.Blocks:
    """Create the Gradio UI."""

    setup_logging()
    with gr.Blocks(theme=gr.themes.Soft(), title="Mushroom AI Assistant") as demo:
        gr.Markdown(
            "# Mushroom AI Assistant\n"
            "Upload a mushroom image, provide structured mushroom features as JSON, and get an integrated safety assessment with prompt comparison."
        )
        with gr.Row():
            with gr.Column(scale=1):
                image_input = gr.Image(type="filepath", label="Mushroom image")
                feature_input = gr.Textbox(
                    label="Structured mushroom features as JSON",
                    lines=16,
                    value=json.dumps(DEFAULT_FEATURE_TEMPLATE, indent=2),
                    placeholder='{"cap_shape": "x", "odor": "n", "habitat": "u"}',
                )
                api_key_input = gr.Textbox(label="OpenAI API key override", type="password", value="")
                model_input = gr.Textbox(label="OpenAI model", value="gpt-4o-mini")
                run_button = gr.Button("Analyze mushroom", variant="primary")
            with gr.Column(scale=1):
                summary_output = gr.Markdown(label="Integrated assessment")
                cv_output = gr.JSON(label="Computer vision output")
                numeric_output = gr.JSON(label="Numeric model output")
                llm_output = gr.JSON(label="LLM output")
                comparison_output = gr.JSON(label="Prompt comparison")

        def submit(image_path: str, feature_json: str, api_key_override: str, openai_model: str):
            combined_key = get_openai_api_key(api_key_override)
            return run_assistant(image_path, feature_json, combined_key, openai_model)

        run_button.click(
            fn=submit,
            inputs=[image_input, feature_input, api_key_input, model_input],
            outputs=[summary_output, cv_output, numeric_output, llm_output, comparison_output],
        )
    return demo


def main() -> None:
    """Launch the Gradio app."""

    demo = build_interface()
    demo.launch(server_name="0.0.0.0", server_port=int(os.getenv("PORT", "7860")))


if __name__ == "__main__":
    main()
