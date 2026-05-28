"""Generate mushroom safety guidance from the CV and numeric model outputs."""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, Optional

from openai import OpenAI

LOGGER = logging.getLogger("mushroom.llm")
DEFAULT_MODEL = "gpt-4o-mini"
HIGH_CONFIDENCE_THRESHOLD = 0.80
PROMPT_B_COOKING_THRESHOLD = 0.80
MUSHROOM_FEATURE_KEYS = (
    "cap-diameter",
    "cap-shape",
    "cap-surface",
    "cap-color",
    "does-bruise-or-bleed",
    "gill-attachment",
    "gill-spacing",
    "gill-color",
    "stem-height",
    "stem-width",
    "stem-root",
    "stem-surface",
    "stem-color",
    "veil-type",
    "veil-color",
    "has-ring",
    "ring-type",
    "spore-print-color",
    "habitat",
    "season",
)


@dataclass
class MushroomSignals:
    """Inputs passed from the vision and tabular models into the LLM layer."""

    species: str
    cv_confidence: float
    numeric_class: str
    numeric_edible_probability: Optional[float]
    common_name: Optional[str] = None
    cv_edibility: Optional[str] = None
    conflict_detected: bool = False
    numeric_default_based: bool = False
    numeric_evaluated: bool = True


def should_allow_cooking(signals: MushroomSignals) -> bool:
    """Decide whether the assistant may mention cooking ideas."""

    if not signals.numeric_evaluated:
        return False
    if signals.numeric_default_based:
        return False
    if signals.cv_confidence < PROMPT_B_COOKING_THRESHOLD:
        return False
    if signals.numeric_edible_probability is None:
        return False
    if signals.numeric_edible_probability < PROMPT_B_COOKING_THRESHOLD:
        return False
    return "edible" in signals.numeric_class.lower() or "edible" in signals.species.lower()


def call_llm_json(prompt: str, api_key: Optional[str] = None, model: str = DEFAULT_MODEL) -> str:
    """Call the LLM and return a JSON string response."""

    effective_key = (api_key or os.getenv("OPENAI_API_KEY", "")).strip()
    if not effective_key:
        LOGGER.warning("OpenAI API key not provided, returning empty JSON payload.")
        return "{}"

    client = OpenAI(api_key=effective_key)
    response = client.chat.completions.create(
        model=model,
        temperature=0.0,
        response_format={"type": "json_object"},
        messages=[
            {
                "role": "system",
                "content": (
                    "Return only valid JSON. Do not add markdown, commentary, or extra keys unless explicitly requested."
                ),
            },
            {"role": "user", "content": prompt},
        ],
    )
    return response.choices[0].message.content or "{}"


def parse_json_response(response_text: str, required_keys: tuple[str, ...]) -> Dict[str, Any]:
    """Parse and normalize a strict JSON response into the requested key set."""

    try:
        payload = json.loads(response_text)
    except json.JSONDecodeError:
        LOGGER.warning("Failed to parse JSON response from LLM.")
        return {}

    if not isinstance(payload, dict):
        LOGGER.warning("LLM JSON response was not an object.")
        return {}

    normalized: Dict[str, Any] = {}
    for key in required_keys:
        value = payload.get(key)
        if isinstance(value, str):
            stripped = value.strip()
            normalized[key] = stripped if stripped else None
        else:
            normalized[key] = value
    return normalized


def _has_meaningful_feature_values(features: Dict[str, Any]) -> bool:
    """Check whether at least one extracted value is informative."""

    for value in features.values():
        if value is None:
            continue
        if isinstance(value, str) and value.strip().lower() in {"", "unknown", "null", "none"}:
            continue
        return True
    return False


def extract_mushroom_features(description: str, api_key: Optional[str] = None, model: str = DEFAULT_MODEL) -> Dict[str, Any]:
    """Extract structured mushroom features from a free-text description."""

    if not description.strip():
        return {}

    prompt = json.dumps(
        {
            "instruction": (
                "Extract only mushroom features that are explicitly mentioned in the description. "
                "Do not infer uncertain details. If a feature is not mentioned, set it to null or \"unknown\". "
                "If the description is too vague to extract any useful structured features, return an empty JSON object."
            ),
            "description": description.strip(),
            "required_json_keys": list(MUSHROOM_FEATURE_KEYS),
            "output_rules": [
                "Return strict JSON only.",
                "Use the exact required keys.",
                "Prefer null over guessing.",
                "Use numbers for numeric values when explicitly stated.",
            ],
        },
        indent=2,
    )

    response_text = call_llm_json(prompt, api_key=api_key, model=model)
    parsed = parse_json_response(response_text, MUSHROOM_FEATURE_KEYS)
    if not parsed or not _has_meaningful_feature_values(parsed):
        return {}
    return parsed


def build_context(signals: MushroomSignals) -> Dict[str, Any]:
    """Create a compact, model-agnostic payload for prompt construction."""

    allow_cooking = should_allow_cooking(signals)
    numeric_evaluated = signals.numeric_evaluated
    if not numeric_evaluated:
        numeric_signal_mode = "not evaluated"
    elif signals.numeric_default_based:
        numeric_signal_mode = "auxiliary only (default features)"
    else:
        numeric_signal_mode = "structured user features"
    return {
        "species": signals.species,
        "common_name": signals.common_name,
        "cv_confidence": round(signals.cv_confidence, 4),
        "cv_edibility": signals.cv_edibility,
        "numeric_prediction": signals.numeric_class if numeric_evaluated else None,
        "numeric_prediction_raw": signals.numeric_class,
        "numeric_probability": None if signals.numeric_edible_probability is None else round(signals.numeric_edible_probability, 4),
        "numeric_default_based": signals.numeric_default_based,
        "numeric_evaluated": numeric_evaluated,
        "numeric_signal_mode": numeric_signal_mode,
        "allow_cooking": allow_cooking,
        "conflict_detected": signals.conflict_detected,
        "required_disclaimer": "This application is for educational purposes only and must not be used as the sole basis for eating wild mushrooms.",
    }


def build_prompt_a(signals: MushroomSignals) -> str:
    """Prompt A: a short, generic explanation prompt."""

    context = build_context(signals)
    payload = {
        "instruction": "Explain the mushroom prediction briefly.",
        "context": context,
        "output_format": ["explanation", "safety_warning", "disclaimer"],
    }
    return json.dumps(payload, indent=2)


def build_prompt_b(signals: MushroomSignals) -> str:
    """Prompt B: a stricter safety-first prompt with confidence gating."""

    context = build_context(signals)
    payload = {
        "instruction": "Explain the mushroom prediction with explicit safety language.",
        "context": context,
        "constraints": [
            "Include the CV confidence.",
            "If numeric_evaluated is false, say the structured model was not evaluated because no structured features were provided.",
            "Do not describe numeric_prediction as a real classification when numeric_evaluated is false.",
            "Include the numeric edible probability only when the numeric model was actually evaluated.",
            f"Only include cooking ideas if allow_cooking is true and both confidences are at least {PROMPT_B_COOKING_THRESHOLD}.",
            "Always include the educational disclaimer.",
        ],
        "output_format": ["explanation", "safety_warning", "cooking_suggestions", "disclaimer"],
    }
    return json.dumps(payload, indent=2)


def build_prompt_variants(signals: MushroomSignals) -> Dict[str, str]:
    """Return both prompt strategies for documentation or comparison."""

    return {"A": build_prompt_a(signals), "B": build_prompt_b(signals)}


def heuristic_prompt_score(response: Dict[str, str], allow_cooking: bool) -> Dict[str, float]:
    """Score a response using simple safety and structure heuristics."""

    disclaimer = response.get("disclaimer", "").lower()
    cooking = response.get("cooking_suggestions", "").strip()
    safety_warning = response.get("safety_warning", "").strip()
    explanation = response.get("explanation", "").strip()

    structure_score = float(bool(explanation) + bool(safety_warning) + bool(disclaimer)) / 3.0
    safety_score = 1.0 if "educational" in disclaimer and "wild mushrooms" in safety_warning.lower() else 0.5
    cooking_score = 1.0 if (allow_cooking and cooking) or (not allow_cooking and not cooking) else 0.0
    return {
        "structure": round(structure_score, 3),
        "safety": round(safety_score, 3),
        "cooking_gate": round(cooking_score, 3),
        "overall": round((structure_score + safety_score + cooking_score) / 3.0, 3),
    }


def fallback_response(signals: MushroomSignals, prompt_variant: str, allow_cooking: bool) -> Dict[str, str]:
    """Return a deterministic safety-focused response when the API is unavailable."""

    species_text = signals.species
    if signals.common_name:
        species_text = f"{signals.species} ({signals.common_name})"

    if not signals.numeric_evaluated:
        numeric_text = "The structured model was not evaluated because no structured features were provided."
    elif signals.numeric_default_based:
        numeric_text = "The structured model was not evaluated because only default fallback features were available."
    else:
        numeric_text = f"The structured model predicts {signals.numeric_class}."

    explanation = (
        f"Computer vision suggests {species_text} with confidence {signals.cv_confidence:.2f}. "
        + numeric_text
    )
    if signals.cv_edibility:
        explanation += f" The species mapping marks this as {signals.cv_edibility}."
    if signals.conflict_detected:
        explanation += " The CV and numeric signals disagree, so the safety policy blocks cooking guidance."
    safety_warning = (
        "Wild mushrooms can be dangerous to eat, and visual similarity is not a reliable safety check. "
        "Treat this result as educational only."
    )
    cooking_suggestions = (
        "Because both models are highly confident, you may discuss common culinary uses in a general way."
        if allow_cooking
        else "Cooking suggestions are intentionally withheld because the confidence threshold was not met."
    )
    disclaimer = (
        "This application is for educational purposes only and must not be used as the sole basis for eating wild mushrooms."
    )
    return {
        "prompt_variant": prompt_variant,
        "explanation": explanation,
        "safety_warning": safety_warning,
        "cooking_suggestions": cooking_suggestions,
        "disclaimer": disclaimer,
    }


def generate_mushroom_advice(
    cv_result: Dict[str, Any],
    numeric_result: Dict[str, Any],
    api_key: Optional[str] = None,
    model: str = DEFAULT_MODEL,
    prompt_variant: str = "B",
) -> Dict[str, str]:
    """Generate the final educational explanation and safety guidance."""

    signals = MushroomSignals(
        species=str(cv_result.get("predicted_class", "unknown")),
        cv_confidence=float(cv_result.get("confidence", 0.0) or 0.0),
        numeric_class=str(numeric_result.get("predicted_label", "unknown")),
        numeric_edible_probability=(
            None if numeric_result.get("edible_probability") is None else float(numeric_result["edible_probability"])
        ),
        common_name=(None if cv_result.get("common_name") is None else str(cv_result.get("common_name"))),
        cv_edibility=(None if cv_result.get("cv_edibility") is None else str(cv_result.get("cv_edibility"))),
        conflict_detected=bool(cv_result.get("conflict_detected", False)),
        numeric_default_based=bool(numeric_result.get("default_features_used", False)),
        numeric_evaluated=bool(numeric_result.get("numeric_evaluated", True)),
    )
    allow_cooking = should_allow_cooking(signals)
    prompt = build_prompt_a(signals) if prompt_variant.upper() == "A" else build_prompt_b(signals)

    if not api_key:
        LOGGER.warning("OpenAI API key not provided, using fallback response.")
        return fallback_response(signals, prompt_variant.upper(), allow_cooking)

    try:
        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model=model,
            temperature=0.2,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a cautious mushroom safety assistant. Return valid JSON with keys "
                        "prompt_variant, explanation, safety_warning, cooking_suggestions, and disclaimer. "
                        "Do not recommend eating wild mushrooms. Only include cooking suggestions if allow_cooking is true."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
        )
        content = response.choices[0].message.content or "{}"
        payload = json.loads(content)
        return {
            "prompt_variant": str(payload.get("prompt_variant", prompt_variant.upper())),
            "explanation": str(payload.get("explanation", "No explanation provided.")),
            "safety_warning": str(payload.get("safety_warning", "No safety warning provided.")),
            "cooking_suggestions": str(payload.get("cooking_suggestions", "")),
            "disclaimer": str(payload.get("disclaimer", "")),
        }
    except Exception as exc:  # pragma: no cover - remote API fallback
        LOGGER.exception("OpenAI generation failed, using fallback response: %s", exc)
        return fallback_response(signals, prompt_variant.upper(), allow_cooking)


def compare_prompt_strategies(
    cv_result: Dict[str, Any],
    numeric_result: Dict[str, Any],
    api_key: Optional[str] = None,
    model: str = DEFAULT_MODEL,
) -> Dict[str, Any]:
    """Generate and compare Prompt A and Prompt B outputs."""

    signals = MushroomSignals(
        species=str(cv_result.get("predicted_class", "unknown")),
        cv_confidence=float(cv_result.get("confidence", 0.0) or 0.0),
        numeric_class=str(numeric_result.get("predicted_label", "unknown")),
        numeric_edible_probability=(
            None if numeric_result.get("edible_probability") is None else float(numeric_result["edible_probability"])
        ),
        common_name=(None if cv_result.get("common_name") is None else str(cv_result.get("common_name"))),
        cv_edibility=(None if cv_result.get("cv_edibility") is None else str(cv_result.get("cv_edibility"))),
        conflict_detected=bool(cv_result.get("conflict_detected", False)),
        numeric_default_based=bool(numeric_result.get("default_features_used", False)),
        numeric_evaluated=bool(numeric_result.get("numeric_evaluated", True)),
    )
    allow_cooking = should_allow_cooking(signals)
    response_a = generate_mushroom_advice(cv_result, numeric_result, api_key=api_key, model=model, prompt_variant="A")
    response_b = generate_mushroom_advice(cv_result, numeric_result, api_key=api_key, model=model, prompt_variant="B")
    score_a = heuristic_prompt_score(response_a, allow_cooking)
    score_b = heuristic_prompt_score(response_b, allow_cooking)
    preferred_variant = "B" if score_b["overall"] >= score_a["overall"] else "A"

    return {
        "signals": {
            "species": signals.species,
            "common_name": signals.common_name,
            "cv_confidence": signals.cv_confidence,
            "cv_edibility": signals.cv_edibility,
            "numeric_prediction": signals.numeric_class,
            "numeric_edible_probability": signals.numeric_edible_probability,
            "numeric_default_based": signals.numeric_default_based,
            "numeric_signal_mode": "auxiliary only (default features)" if signals.numeric_default_based else "structured user features",
            "conflict_detected": signals.conflict_detected,
            "allow_cooking": allow_cooking,
        },
        "prompt_a": response_a,
        "prompt_b": response_b,
        "comparison": {
            "A": score_a,
            "B": score_b,
            "preferred_variant": preferred_variant,
        },
    }
