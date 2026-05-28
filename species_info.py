"""Species metadata mapping used by the app for species-level CV output."""

from __future__ import annotations

from typing import Any, Dict, Optional

# Species-level safety labels are intentionally explicit: known edible entries may
# inform the UI, but they never become a consumption recommendation by themselves.
SPECIES_EDIBILITY: Dict[str, str] = {
    "Agaricus_bisporus": "edible",
    "Agaricus_subrufescens": "edible",
    "Amanita_bisporigera": "poisonous",
    "Amanita_muscaria": "poisonous",
    "Amanita_ocreata": "poisonous",
    "Amanita_phalloides": "poisonous",
    "Amanita_smithiana": "poisonous",
    "Amanita_verna": "poisonous",
    "Amanita_virosa": "poisonous",
    "Auricularia_auricula": "edible",
    "Boletus_edulis": "edible",
    "Cantharellus_cibarius": "edible",
    "Cantharellus_cinnabarinus": "edible",
    "Clitocybe_dealbata": "poisonous",
    "Conocybe_filaris": "poisonous",
    "Coprinus_comatus": "edible",
    "Cordyceps_sinensis": "edible",
    "Cortinarius_rubellus": "poisonous",
    "Entoloma_sinuatum": "poisonous",
    "Flammulina_velutipes": "edible",
    "Galerina_marginata": "poisonous",
    "Ganoderma_lucidum": "edible",
    "Grifola_frondosa": "edible",
    "Gyromitra_esculenta": "poisonous",
    "Hericium_erinaceus": "edible",
    "Hydnum_repandum": "edible",
    "Hypholoma_fasciculare": "poisonous",
    "Inocybe_erubescens": "poisonous",
    "Lentinula_edodes": "edible",
    "Lepiota_brunneoincarnata": "poisonous",
    "Macrolepiota_procera": "edible",
    "Morchella_esculenta": "edible",
    "Omphalotus_olearius": "poisonous",
    "Paxillus_involutus": "poisonous",
    "Pholiota_nameko": "edible",
    "Pleurotus_citrinopileatus": "edible",
    "Pleurotus_eryngii": "edible",
    "Pleurotus_ostreatus": "edible",
    "Psilocybe_semilanceata": "poisonous",
    "Rhodophyllus_rhodopolius": "poisonous",
    "Russula_emetica": "poisonous",
    "Russula_virescens": "edible",
    "Scleroderma_citrinum": "poisonous",
    "Suillus_luteus": "edible",
    "Tremella_fuciformis": "edible",
    "Tricholoma_matsutake": "edible",
    "Tuber_melanosporum": "edible",
}

COMMON_NAMES: Dict[str, str] = {
    "Agaricus_bisporus": "Champignon",
    "Amanita_muscaria": "Fliegenpilz",
    "Amanita_phalloides": "Gruener Knollenblaetterpilz",
    "Amanita_verna": "Fruehlings-Knollenblaetterpilz",
    "Amanita_virosa": "Kegeliger Knollenblaetterpilz",
    "Boletus_edulis": "Steinpilz",
    "Cantharellus_cibarius": "Pfifferling",
    "Cantharellus_cinnabarinus": "Roter Pfifferling",
    "Coprinus_comatus": "Schopftintling",
    "Cordyceps_sinensis": "Chinesischer Raupenpilz",
    "Flammulina_velutipes": "Samtfussruebling",
    "Ganoderma_lucidum": "Lackporling",
    "Grifola_frondosa": "Maitake",
    "Hericium_erinaceus": "Igelstachelbart",
    "Lentinula_edodes": "Shiitake",
    "Macrolepiota_procera": "Parasol",
    "Morchella_esculenta": "Speisemorchel",
    "Pleurotus_citrinopileatus": "Goldseitling",
    "Pleurotus_eryngii": "Kraeuterseitling",
    "Pleurotus_ostreatus": "Austernseitling",
    "Suillus_luteus": "Butterpilz",
    "Tremella_fuciformis": "Silberohr",
    "Tricholoma_matsutake": "Matsutake",
    "Tuber_melanosporum": "Schwarzer Trueffel",
}

EDIBLE_SAFETY_NOTE = (
    "Edible in this dataset mapping, but always verify with expert guidance. "
    "Never rely on image classification alone for wild mushroom consumption."
)

POISONOUS_SAFETY_NOTE = (
    "Potentially dangerous or toxic species in this dataset mapping. "
    "Do not consume."
)


def _display_name(species_key: str) -> str:
    return species_key.replace("_", " ")


def _build_species_info() -> Dict[str, Dict[str, Any]]:
    info: Dict[str, Dict[str, Any]] = {}
    for species_key, edibility in SPECIES_EDIBILITY.items():
        # `cookable` only means the dataset mapping marks the species as edible.
        # The app still requires CV confidence and safety checks before showing anything user-facing.
        cookable = edibility == "edible"
        info[species_key] = {
            "scientific_name": species_key,
            "display_name": _display_name(species_key),
            "common_name": COMMON_NAMES.get(species_key),
            "edibility": edibility,
            "cookable": cookable,
            "safety_note": EDIBLE_SAFETY_NOTE if cookable else POISONOUS_SAFETY_NOTE,
        }
    return info


SPECIES_INFO: Dict[str, Dict[str, Any]] = _build_species_info()


def get_species_info(species_key: str) -> Optional[Dict[str, Any]]:
    """Return species metadata for an exact species key (e.g. Agaricus_bisporus)."""

    return SPECIES_INFO.get(species_key)
