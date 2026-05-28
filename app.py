"""Gradio application for the integrated Mushroom AI Assistant."""

from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, Tuple

import gradio as gr
import joblib
import pandas as pd

from llm import compare_prompt_strategies, extract_mushroom_features, generate_mushroom_advice
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
NUMERIC_FIELD_LABELS = {
    "cap-diameter": "Cap diameter",
    "cap-shape": "Cap shape",
    "cap-surface": "Cap surface",
    "cap-color": "Cap color",
    "does-bruise-or-bleed": "Bruises or bleeds",
    "gill-attachment": "Gill attachment",
    "gill-spacing": "Gill spacing",
    "gill-color": "Gill color",
    "stem-height": "Stem height",
    "stem-width": "Stem width",
    "stem-root": "Stem root",
    "stem-surface": "Stem surface",
    "stem-color": "Stem color",
    "veil-type": "Veil type",
    "veil-color": "Veil color",
    "has-ring": "Has ring",
    "ring-type": "Ring type",
    "spore-print-color": "Spore print color",
    "habitat": "Habitat",
    "season": "Season",
}
NUMERIC_FIELD_HINTS = {
    "gill-spacing": "Choose the closest match if known.",
    "spore-print-color": "Use this only if you know the print color.",
    "stem-root": "Choose the closest visible stem base type.",
    "season": "Use the observed season, if known.",
}
NUMERIC_CATEGORICAL_OPTIONS = {
    "cap-shape": [("Bell", "b"), ("Conical", "c"), ("Flat", "f"), ("Others / oval", "o"), ("Convex", "p"), ("Sunken", "s"), ("Knobbed / convex", "x")],
    "cap-surface": [("Dry", "d"), ("Fibrous", "e"), ("Grooves", "g"), ("Holes", "h"), ("Silky", "i"), ("Shiny", "k"), ("Smooth", "l"), ("Scaly", "s"), ("Sticky", "t"), ("Wrinkled", "w"), ("Fleshy", "y")],
    "cap-color": [("Brown", "b"), ("Buff", "e"), ("Gray", "g"), ("Black", "k"), ("Green", "l"), ("Yellow", "n"), ("Orange", "o"), ("Pink", "p"), ("Red", "r"), ("Purple", "u"), ("White", "w"), ("Blue", "y")],
    "does-bruise-or-bleed": [("No", "f"), ("Yes", "t")],
    "gill-attachment": [("Attached", "a"), ("Descending", "d"), ("Free", "e"), ("Notched", "f"), ("Pores", "p"), ("Separated", "s"), ("Sinuate", "x")],
    "gill-spacing": [("Unspecified", ""), ("Close", "c"), ("Crowded", "d"), ("Distant", "f")],
    "gill-color": [("Brown", "b"), ("Buff", "e"), ("Red", "f"), ("Gray", "g"), ("Black", "k"), ("Green", "n"), ("Orange", "o"), ("Pink", "p"), ("Purple", "r"), ("Yellow", "u"), ("White", "w"), ("Blue", "y")],
    "stem-root": [("Bulbous", "b"), ("Club", "c"), ("Fusiform", "f"), ("Rooted", "r"), ("Missing", "s")],
    "stem-surface": [("Fibrous", "f"), ("Grooves", "g"), ("Scaly", "h"), ("Silky", "i"), ("Shiny", "k"), ("Smooth", "s"), ("Sticky", "t"), ("Wrinkled", "y")],
    "stem-color": [("Brown", "b"), ("Buff", "e"), ("Red", "f"), ("Gray", "g"), ("Black", "k"), ("Green", "l"), ("Yellow", "n"), ("Orange", "o"), ("Pink", "p"), ("Purple", "r"), ("Purple / violet", "u"), ("White", "w"), ("Blue", "y")],
    "veil-type": [("Universal", "u")],
    "veil-color": [("Buff", "e"), ("Black", "k"), ("Brown", "n"), ("Universal", "u"), ("White", "w"), ("Yellow", "y")],
    "has-ring": [("No", "f"), ("Yes", "t")],
    "ring-type": [("Evanescent", "e"), ("Flaring", "f"), ("Grooved", "g"), ("Large", "l"), ("Moving", "m"), ("Pendant", "p"), ("Sheathing", "r"), ("Zone", "z")],
    "spore-print-color": [("Unspecified", ""), ("Green", "g"), ("Black", "k"), ("Brown", "n"), ("Pink", "p"), ("Purple / violet", "r"), ("Purple", "u"), ("White", "w")],
    "habitat": [("Woods", "d"), ("Grasses", "g"), ("Leaves", "h"), ("Meadows", "l"), ("Waste", "m"), ("Pathways", "p"), ("Urban", "u"), ("Woodland / grassy", "w")],
    "season": [("Spring", "a"), ("Summer", "s"), ("Autumn", "u"), ("Winter", "w")],
}
NUMERIC_NUMBER_DEFAULTS = {
    "cap-diameter": DEFAULT_FEATURE_TEMPLATE["cap-diameter"],
    "stem-height": DEFAULT_FEATURE_TEMPLATE["stem-height"],
    "stem-width": DEFAULT_FEATURE_TEMPLATE["stem-width"],
}
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


def has_meaningful_structured_features(features: Dict[str, object]) -> bool:
    """Check whether extracted structured features contain any usable signal."""

    if not features:
        return False

    for value in features.values():
        if value is None:
            continue
        if isinstance(value, str) and value.strip().lower() in {"", "unknown", "null", "none"}:
            continue
        return True

    return False


def build_manual_structured_features(
    cap_diameter: float,
    cap_shape: str,
    cap_surface: str,
    cap_color: str,
    does_bruise_or_bleed: str,
    gill_attachment: str,
    gill_spacing: str,
    gill_color: str,
    stem_height: float,
    stem_width: float,
    stem_root: str,
    stem_surface: str,
    stem_color: str,
    veil_type: str,
    veil_color: str,
    has_ring: str,
    ring_type: str,
    spore_print_color: str,
    habitat: str,
    season: str,
) -> Dict[str, object]:
    """Build a canonical feature dict from the manual UI controls."""

    return {
        "cap-diameter": cap_diameter,
        "cap-shape": cap_shape,
        "cap-surface": cap_surface,
        "cap-color": cap_color,
        "does-bruise-or-bleed": does_bruise_or_bleed,
        "gill-attachment": gill_attachment,
        "gill-spacing": gill_spacing,
        "gill-color": gill_color,
        "stem-height": stem_height,
        "stem-width": stem_width,
        "stem-root": stem_root,
        "stem-surface": stem_surface,
        "stem-color": stem_color,
        "veil-type": veil_type,
        "veil-color": veil_color,
        "has-ring": has_ring,
        "ring-type": ring_type,
        "spore-print-color": spore_print_color,
        "habitat": habitat,
        "season": season,
    }


def build_numeric_dropdown(field_key: str, value: str, info: str | None = None) -> gr.Dropdown:
    """Build a human-readable dropdown that still returns the raw ML code."""

    dropdown_kwargs: Dict[str, object] = {
        "label": NUMERIC_FIELD_LABELS[field_key],
        "choices": NUMERIC_CATEGORICAL_OPTIONS[field_key],
        "value": value,
    }
    if info:
        dropdown_kwargs["info"] = info
    return gr.Dropdown(**dropdown_kwargs)


def build_numeric_number(field_key: str, value: float) -> gr.Number:
    """Build a readable numeric input for continuous mushroom traits."""

    return gr.Number(label=NUMERIC_FIELD_LABELS[field_key], value=value, precision=2)


def predict_numeric_sample(feature_json: str, feature_source: str = "structured features") -> Dict[str, object]:
    """Run inference through the saved numeric pipeline."""

    pipeline = load_numeric_pipeline()
    metadata = load_numeric_metadata()
    frame = build_numeric_frame(feature_json)
    default_input = is_default_numeric_input(feature_json)
    LOGGER.debug("Numeric model input source=%s default_input=%s raw_features=%s", feature_source, default_input, frame.iloc[0].to_dict())
    prediction = int(pipeline.predict(frame)[0])
    probabilities = pipeline.predict_proba(frame)[0] if hasattr(pipeline, "predict_proba") else None

    transformed_frame = None
    if hasattr(pipeline, "named_steps") and isinstance(getattr(pipeline, "named_steps"), dict):
        preprocessor = pipeline.named_steps.get("preprocessor")
        if preprocessor is not None and hasattr(preprocessor, "transform"):
            try:
                transformed_frame = preprocessor.transform(frame)
                dense_frame = transformed_frame.toarray() if hasattr(transformed_frame, "toarray") else transformed_frame
                LOGGER.debug("Numeric encoded vector=%s", dense_frame.tolist() if hasattr(dense_frame, "tolist") else dense_frame)
            except Exception as exc:  # pragma: no cover - debug aid only
                LOGGER.debug("Numeric encoding debug failed: %s", exc)

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

    LOGGER.debug(
        "Numeric model output prediction=%s edible_probability=%s probabilities=%s",
        predicted_label,
        edible_probability,
        None if probabilities is None else probabilities.tolist(),
    )

    return {
        "predicted_label": predicted_label,
        "edible_probability": edible_probability,
        "probabilities": None if probabilities is None else probabilities.tolist(),
        "features": frame.iloc[0].to_dict(),
        "feature_source": feature_source,
        "default_features_used": default_input,
        "numeric_evaluated": True,
    }


def build_skipped_numeric_result(feature_source: str = "fallback") -> Dict[str, object]:
    """Return a placeholder result when no structured features were provided."""

    LOGGER.debug("Skipping numeric model: no structured features available (source=%s)", feature_source)
    return {
        "predicted_label": None,
        "edible_probability": None,
        "probabilities": None,
        "features": {},
        "feature_source": feature_source,
        "default_features_used": False,
        "numeric_evaluated": False,
        "skipped_reason": "no structured features provided",
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


def _truncate_sentences(text: object, max_sentences: int = 2, max_length: int = 180) -> str:
    """Trim a paragraph to a short, dashboard-friendly explanation."""

    if text is None:
        return "—"
    value = str(text).strip()
    if not value:
        return "—"
    sentences = re.split(r"(?<=[.!?])\s+", value)
    trimmed = " ".join(sentence.strip() for sentence in sentences[:max_sentences] if sentence.strip())
    if len(trimmed) > max_length:
        return trimmed[: max_length - 1].rstrip() + "…"
    return trimmed or _shorten_text(value, max_length=max_length)


def _escape_html(value: object) -> str:
    """Escape text for safe HTML rendering inside the summary card."""

    text = "" if value is None else str(value)
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def _badge(label: str, background: str, foreground: str = "#ffffff") -> str:
    """Render a compact status badge for the summary header."""

    return (
        f"<span style='display:inline-block;padding:0.18rem 0.6rem;margin:0 0.35rem 0.35rem 0;"
        f"border-radius:999px;background:{background};color:{foreground};font-size:0.78rem;"
        f"font-weight:700;letter-spacing:0.02em;text-transform:uppercase;'>"
        f"{_escape_html(label)}</span>"
    )


def _row(label: str, value: object, value_html: bool = False) -> str:
    """Render a compact labeled row for the result card."""

    rendered_value = str(value) if value is not None else "—"
    if not value_html:
        rendered_value = _escape_html(rendered_value)

    return (
        "<div style='display:flex;justify-content:space-between;gap:1rem;padding:0.3rem 0;"
        "border-bottom:1px solid rgba(85, 74, 56, 0.08);'>"
        f"<div style='font-weight:700;color:#3a3124;'>{_escape_html(label)}</div>"
        f"<div style='text-align:right;flex:1;color:#1f1a12;'>{rendered_value}</div>"
        "</div>"
    )


def _section(title: str, rows: list[str], accent: str = "rgba(85,74,56,0.1)", title_color: str = "#2d251a", background: str = "rgba(255,255,255,0.78)") -> str:
    """Render a named section for the summary card."""

    return (
        f"<div style='margin-top:0.75rem;padding:0.8rem 0.9rem;border-radius:16px;"
        f"background:{background};border:1px solid {accent};box-shadow:0 8px 18px rgba(54,45,28,0.05);'>"
        f"<div style='font-size:1rem;font-weight:800;margin-bottom:0.4rem;color:{title_color};'>{_escape_html(title)}</div>"
        + "".join(rows)
        + "</div>"
    )


def build_result_summary(
    main_pred: str | None,
    main_conf: float | None,
    top_k_list: list[dict[str, object]],
    species_meta: Dict[str, object] | None,
    cv_edibility: str | None,
    numeric_result: Dict[str, object],
    advice: Dict[str, object],
    final_safety_decision: str,
    decision_reason: str,
    consumption_advice: str,
) -> str:
    """Build the visible summary card with compact sections and badges."""

    numeric_evaluated = bool(numeric_result.get("numeric_evaluated", False))
    numeric_feature_source = str(numeric_result.get("feature_source", "fallback"))
    numeric_status_text = (
        "Numeric model evaluated using manual / LLM-extracted features."
        if numeric_evaluated
        else "Numeric model not evaluated. Enable structured feature inputs to run the numeric model."
    )
    numeric_prediction_text = (
        str(numeric_result.get("predicted_label")) if numeric_evaluated and numeric_result.get("predicted_label") is not None else "not evaluated"
    )
    numeric_probability_text = (
        f"{float(numeric_result.get('edible_probability')):.2%}" if numeric_evaluated and numeric_result.get("edible_probability") is not None else "—"
    )
    explanation_status = "generated"
    explanation_source = f"Prompt {advice.get('prompt_variant', 'B')}"
    explanation_text = _truncate_sentences(advice.get("explanation"), max_sentences=2, max_length=160)
    top3_items = []
    for prediction in top_k_list[:3]:
        species_name = str(prediction.get("class", "—"))
        species_info = get_species_info(species_name)
        if species_info and species_info.get("common_name"):
            label = f"{species_info.get('common_name')} ({species_name})"
        else:
            label = species_name
        top3_items.append(f"{label} ({float(prediction.get('confidence', 0.0)):.2%})")
    top3_text = ", ".join(top3_items) if top3_items else "—"
    species_mapping_text = cv_edibility or "—"
    if main_pred and species_meta and species_meta.get("common_name"):
        cv_prediction_text = (
            f"<div>{_escape_html(species_meta.get('common_name'))}</div>"
            f"<div style='font-size:0.8rem;color:rgba(31,26,18,0.65);'>{_escape_html(main_pred)}</div>"
        )
    else:
        cv_prediction_text = _escape_html(main_pred or "—")
    cv_confidence_text = f"{main_conf:.2%}" if main_conf is not None else "—"
    final_decision_text = final_safety_decision
    consumption_text = consumption_advice

    badges = (
        _badge("CV", "#7a4b12")
        + _badge("Numeric ML", "#385a7c")
        + _badge("NLP", "#4f6f52")
        + _badge("Safety", "#7a2e2e")
    )

    sections = [
        _section(
            "Computer Vision",
            [
                _row("Species", cv_prediction_text, value_html=True),
                _row("CV confidence", cv_confidence_text),
                _row("Top-3", top3_text),
                _row("Mapping", species_mapping_text),
            ],
        ),
        _section(
            "Numeric ML",
            [
                _row("Status", numeric_status_text),
                *([
                    _row("Prediction", numeric_prediction_text),
                    _row("Probability", numeric_probability_text),
                    _row("Source", numeric_feature_source),
                ] if numeric_evaluated else [
                    _row("How to enable", "Add structured features or a detailed description to run the numeric model."),
                ]),
            ],
        ),
        _section(
            "NLP",
            [
                _row("Status", explanation_status),
                _row("Source", explanation_source),
                _row("Summary", explanation_text),
            ],
        ),
        _section(
            "Final Safety Decision",
            [
                _row("Decision", final_decision_text),
                _row("Reason", decision_reason),
                _row("Advice", consumption_text),
            ],
            accent="rgba(122,46,46,0.22)",
            title_color="#7a2e2e",
            background="rgba(122,46,46,0.06)",
        ),
    ]

    disclaimer = (
        "<div style='margin-top:0.9rem;padding:0.7rem 0.9rem;border-left:4px solid #7a5b2b;"
        "background:rgba(122,91,43,0.08);border-radius:12px;color:#3a3124;'>"
        "Expert verification is required. This app does not provide a consumption recommendation."
        "</div>"
    )

    return (
        "<div class='summary-card'>"
        f"<div style='margin-bottom:0.75rem;'>{badges}</div>"
        + "".join(sections)
        + disclaimer
        + "</div>"
    )


def _panel_update(visible: bool, value: object | None = None) -> object:
    """Build a lightweight visibility update for technical panels."""

    if value is None:
        return gr.update(visible=visible)
    return gr.update(visible=visible, value=value)


def run_assistant(
    image_path: str,
    description_text: str,
    use_manual_structured_features: bool,
    manual_structured_features: Dict[str, object],
    api_key: str,
    openai_model: str,
) -> Tuple[str, Dict[str, object], Dict[str, object], Dict[str, object], Dict[str, object], Dict[str, object]]:
    """Run the three-stage assistant and return user-facing outputs."""

    if not image_path:
        raise ValueError("Please upload a mushroom image before running the assistant.")

    if use_manual_structured_features:
        extracted_structured_features = normalize_feature_names(manual_structured_features)
        structured_features_available = True
        feature_source = "manual"
    else:
        extracted_structured_features = normalize_feature_names(extract_mushroom_features(description_text, api_key=api_key, model=openai_model))
        structured_features_available = has_meaningful_structured_features(extracted_structured_features)
        feature_source = "LLM extracted structured features" if structured_features_available else "fallback"

    feature_json = json.dumps(extracted_structured_features, indent=2) if structured_features_available else DEFAULT_FEATURE_JSON
    LOGGER.debug(
        "Structured feature extraction source=%s available=%s raw_features=%s",
        feature_source,
        structured_features_available,
        extracted_structured_features,
    )

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

    if structured_features_available:
        numeric_result = predict_numeric_sample(feature_json, feature_source=feature_source)
        if use_manual_structured_features:
            numeric_result["default_features_used"] = False
            numeric_result["feature_source"] = "manual"
    else:
        numeric_result = build_skipped_numeric_result(feature_source=feature_source)
    numeric_evaluated = bool(numeric_result.get("numeric_evaluated", False))
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
    low_cv_confidence = main_conf is None or main_conf < 0.80

    conflict = False
    if numeric_evaluated and numeric_claim is not None:
        if species_mapping_claim is not None:
            conflict = species_mapping_claim != numeric_claim
        elif cv_claim is not None:
            conflict = cv_claim != numeric_claim

    LOGGER.debug(
        "Decision inputs cv_pred=%s cv_conf=%.4f species_mapping=%s numeric_evaluated=%s numeric_label=%s numeric_prob=%s conflict=%s source=%s",
        main_pred,
        main_conf or 0.0,
        species_mapping_claim,
        numeric_evaluated,
        numeric_result.get("predicted_label"),
        numeric_result.get("edible_probability"),
        conflict,
        numeric_result.get("feature_source"),
    )

    if isinstance(cv_result_for_llm, dict):
        cv_result_for_llm["common_name"] = species_meta.get("common_name") if species_meta else None
        cv_result_for_llm["cv_edibility"] = cv_edibility
        cv_result_for_llm["conflict_detected"] = conflict
        cv_result_for_llm["numeric_evaluated"] = numeric_evaluated

    advice = generate_mushroom_advice(cv_result_for_llm, numeric_result, api_key=api_key, model=openai_model, prompt_variant="B")
    prompt_comparison = compare_prompt_strategies(cv_result_for_llm, numeric_result, api_key=api_key, model=openai_model)

    # Edible CV mapping is allowed to override only when confidence is high and
    # the evaluated signals do not conflict with real structured features.
    allow_cv_override = (
        species_mapping_claim == "edible"
        and main_conf is not None
        and main_conf >= 0.80
        and not conflict
    )

    if allow_cv_override:
        final_safety_decision = "Likely edible species detected"
        consumption_advice = "No consumption recommendation. Expert verification required."
    elif species_mapping_claim == "poisonous":
        final_safety_decision = "Poisonous species detected"
        consumption_advice = "withheld for safety"
    elif low_cv_confidence:
        final_safety_decision = "Consumption advice withheld for safety"
        consumption_advice = "withheld for safety"
    elif conflict and cv_claim == "poisonous" and numeric_claim == "edible":
        final_safety_decision = "Consumption advice withheld for safety"
        consumption_advice = "withheld for safety"
    else:
        final_safety_decision = "Consumption advice withheld for safety"
        consumption_advice = "withheld for safety"

    if species_mapping_claim == "edible" and main_conf is not None and main_conf >= 0.80 and not conflict:
        decision_reason = "Likely edible because CV confidence is high and species mapping is edible."
    elif species_mapping_claim == "poisonous":
        decision_reason = "Blocked because the CV species mapping is poisonous."
    elif low_cv_confidence:
        decision_reason = "Blocked because CV confidence is low."
    elif conflict:
        decision_reason = "Blocked because CV and numeric signals conflict on evaluated structured features."
    elif not numeric_evaluated:
        decision_reason = "Numeric model was skipped because no structured features were provided."
    else:
        decision_reason = "Blocked conservatively because the signal is not strong enough for any recommendation."

    summary = build_result_summary(
        main_pred=main_pred,
        main_conf=main_conf,
        top_k_list=top_k_list,
        species_meta=species_meta,
        cv_edibility=cv_edibility,
        numeric_result=numeric_result,
        advice=advice,
        final_safety_decision=final_safety_decision,
        decision_reason=decision_reason,
        consumption_advice=consumption_advice,
    )

    numeric_panel_visible = numeric_evaluated
    llm_panel_visible = bool(advice and advice.get("explanation"))
    comparison_panel_visible = bool(prompt_comparison and prompt_comparison.get("comparison"))
    features_panel_visible = bool(extracted_structured_features)

    return (
        summary,
        cv_result,
        gr.update(visible=numeric_panel_visible),
        gr.update(visible=numeric_panel_visible, value=numeric_result if numeric_panel_visible else {}),
        gr.update(visible=llm_panel_visible),
        gr.update(visible=llm_panel_visible, value=advice if llm_panel_visible else {}),
        gr.update(visible=comparison_panel_visible),
        gr.update(visible=comparison_panel_visible, value=prompt_comparison if comparison_panel_visible else {}),
        gr.update(visible=features_panel_visible),
        gr.update(visible=features_panel_visible, value=extracted_structured_features if features_panel_visible else {}),
    )


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
        padding: 0.95rem 1.1rem;
        border-radius: 22px;
        background: linear-gradient(135deg, rgba(255,255,255,0.92), rgba(245,240,228,0.92));
        border: 1px solid rgba(85, 74, 56, 0.12);
        box-shadow: 0 12px 30px rgba(54, 45, 28, 0.08);
        margin-bottom: 0.75rem;
    }
    .summary-card {
        padding: 0.8rem 0.9rem;
        border-radius: 18px;
        background: rgba(255, 255, 255, 0.86);
        border: 1px solid rgba(85, 74, 56, 0.12);
        box-shadow: 0 10px 24px rgba(54, 45, 28, 0.07);
        font-size: 0.95rem;
        line-height: 1.45;
    }
    .summary-card h3 {
        margin-top: 0;
        margin-bottom: 0.45rem;
    }
    .summary-card strong {
        color: #2f281d;
    }
    .summary-card p {
        margin-bottom: 0.35rem;
    }
    .compact-accordion {
        margin-top: 0.45rem;
        font-size: 0.92rem;
    }
    .analyze-button button {
        min-height: 2.3rem !important;
        padding: 0.3rem 0.85rem !important;
        font-size: 0.94rem !important;
        border-radius: 10px !important;
    }
    .mini-muted {
        color: rgba(58, 49, 36, 0.74);
        font-size: 0.88rem;
    }
    .technical-panel-title {
        font-size: 0.95rem;
        font-weight: 700;
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
                    with gr.Accordion("Numeric Model Inputs", open=False):
                        gr.Markdown("These structured mushroom traits are used by the Numeric ML model.")
                        use_manual_structured_features_input = gr.Checkbox(
                            label="Use manual structured features for numeric model",
                            value=False,
                        )
                        with gr.Row():
                            with gr.Column():
                                cap_diameter_input = build_numeric_number("cap-diameter", DEFAULT_FEATURE_TEMPLATE["cap-diameter"])
                                cap_shape_input = build_numeric_dropdown("cap-shape", DEFAULT_FEATURE_TEMPLATE["cap-shape"])
                                cap_surface_input = build_numeric_dropdown("cap-surface", DEFAULT_FEATURE_TEMPLATE["cap-surface"])
                                cap_color_input = build_numeric_dropdown("cap-color", DEFAULT_FEATURE_TEMPLATE["cap-color"])
                                does_bruise_or_bleed_input = build_numeric_dropdown("does-bruise-or-bleed", DEFAULT_FEATURE_TEMPLATE["does-bruise-or-bleed"])
                                gill_attachment_input = build_numeric_dropdown("gill-attachment", DEFAULT_FEATURE_TEMPLATE["gill-attachment"])
                                gill_spacing_input = build_numeric_dropdown("gill-spacing", DEFAULT_FEATURE_TEMPLATE["gill-spacing"], info=NUMERIC_FIELD_HINTS["gill-spacing"])
                                gill_color_input = build_numeric_dropdown("gill-color", DEFAULT_FEATURE_TEMPLATE["gill-color"])
                                stem_height_input = build_numeric_number("stem-height", DEFAULT_FEATURE_TEMPLATE["stem-height"])
                                stem_width_input = build_numeric_number("stem-width", DEFAULT_FEATURE_TEMPLATE["stem-width"])
                            with gr.Column():
                                stem_root_input = build_numeric_dropdown("stem-root", DEFAULT_FEATURE_TEMPLATE["stem-root"], info=NUMERIC_FIELD_HINTS["stem-root"])
                                stem_surface_input = build_numeric_dropdown("stem-surface", DEFAULT_FEATURE_TEMPLATE["stem-surface"])
                                stem_color_input = build_numeric_dropdown("stem-color", DEFAULT_FEATURE_TEMPLATE["stem-color"])
                                veil_type_input = build_numeric_dropdown("veil-type", DEFAULT_FEATURE_TEMPLATE["veil-type"])
                                veil_color_input = build_numeric_dropdown("veil-color", DEFAULT_FEATURE_TEMPLATE["veil-color"])
                                has_ring_input = build_numeric_dropdown("has-ring", DEFAULT_FEATURE_TEMPLATE["has-ring"])
                                ring_type_input = build_numeric_dropdown("ring-type", DEFAULT_FEATURE_TEMPLATE["ring-type"])
                                spore_print_color_input = build_numeric_dropdown("spore-print-color", DEFAULT_FEATURE_TEMPLATE["spore-print-color"], info=NUMERIC_FIELD_HINTS["spore-print-color"])
                                habitat_input = build_numeric_dropdown("habitat", DEFAULT_FEATURE_TEMPLATE["habitat"])
                                season_input = build_numeric_dropdown("season", DEFAULT_FEATURE_TEMPLATE["season"], info=NUMERIC_FIELD_HINTS["season"])
                    run_button = gr.Button("Analyze mushroom", variant="primary", elem_classes=["analyze-button"])
                    gr.Markdown("<div class='mini-muted'>Use the manual inputs if you want the numeric model to run without OpenAI feature extraction.</div>")
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

                with gr.Column(scale=1, min_width=420):
                    summary_output = gr.Markdown(elem_classes=["summary-card"])

                    with gr.Accordion("Computer vision output", open=False, elem_classes=["compact-accordion"]) as cv_accordion:
                        cv_output = gr.JSON(label="Computer vision output", show_label=False)

                    with gr.Accordion("Numeric model output", open=False, elem_classes=["compact-accordion"]) as numeric_accordion:
                        numeric_output = gr.JSON(label="Numeric model output", show_label=False)

                    with gr.Accordion("LLM output", open=False, elem_classes=["compact-accordion"]) as llm_accordion:
                        llm_output = gr.JSON(label="LLM output", show_label=False)

                    with gr.Accordion("Prompt comparison", open=False, elem_classes=["compact-accordion"]) as comparison_accordion:
                        comparison_output = gr.JSON(label="Prompt comparison", show_label=False)

                    with gr.Accordion("Structured features JSON", open=False, elem_classes=["compact-accordion"]) as features_accordion:
                        features_output = gr.JSON(label="Structured features JSON", show_label=False)

        def submit(
            image_path: str,
            description_text: str,
            use_manual_structured_features: bool,
            cap_diameter: float,
            cap_shape: str,
            cap_surface: str,
            cap_color: str,
            does_bruise_or_bleed: str,
            gill_attachment: str,
            gill_spacing: str,
            gill_color: str,
            stem_height: float,
            stem_width: float,
            stem_root: str,
            stem_surface: str,
            stem_color: str,
            veil_type: str,
            veil_color: str,
            has_ring: str,
            ring_type: str,
            spore_print_color: str,
            habitat: str,
            season: str,
            api_key_override: str,
            openai_model: str,
        ):
            combined_key = get_openai_api_key(api_key_override)
            manual_structured_features = build_manual_structured_features(
                cap_diameter=cap_diameter,
                cap_shape=cap_shape,
                cap_surface=cap_surface,
                cap_color=cap_color,
                does_bruise_or_bleed=does_bruise_or_bleed,
                gill_attachment=gill_attachment,
                gill_spacing=gill_spacing,
                gill_color=gill_color,
                stem_height=stem_height,
                stem_width=stem_width,
                stem_root=stem_root,
                stem_surface=stem_surface,
                stem_color=stem_color,
                veil_type=veil_type,
                veil_color=veil_color,
                has_ring=has_ring,
                ring_type=ring_type,
                spore_print_color=spore_print_color,
                habitat=habitat,
                season=season,
            )
            return run_assistant(image_path, description_text, use_manual_structured_features, manual_structured_features, combined_key, openai_model)

        run_button.click(
            fn=submit,
            inputs=[
                image_input,
                description_input,
                use_manual_structured_features_input,
                cap_diameter_input,
                cap_shape_input,
                cap_surface_input,
                cap_color_input,
                does_bruise_or_bleed_input,
                gill_attachment_input,
                gill_spacing_input,
                gill_color_input,
                stem_height_input,
                stem_width_input,
                stem_root_input,
                stem_surface_input,
                stem_color_input,
                veil_type_input,
                veil_color_input,
                has_ring_input,
                ring_type_input,
                spore_print_color_input,
                habitat_input,
                season_input,
                api_key_input,
                model_input,
            ],
            outputs=[
                summary_output,
                cv_output,
                numeric_accordion,
                numeric_output,
                llm_accordion,
                llm_output,
                comparison_accordion,
                comparison_output,
                features_accordion,
                features_output,
            ],
        )
    return demo


def main() -> None:
    """Launch the Gradio app."""

    demo = build_interface()
    demo.launch(server_name="0.0.0.0", server_port=int(os.getenv("PORT", "7860")))


if __name__ == "__main__":
    main()
