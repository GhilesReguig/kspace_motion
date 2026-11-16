"""Webapp to simulate k-space motion artefacts on the Colin27 T1w template.

    python webapp/app.py [--host 127.0.0.1] [--port 8000]

Step 1: the motion time course (random motion and/or periodic ghosting motion) is simulated from the
parameters and shown as curves.
Step 2: the user starts the simulation, which applies exactly this time course on Colin27, then the
k-space artefacts that were switched on (noise, spikes, Gibbs ...).
"""
import os

# Several OpenMP runtimes live in the same process (pip torch, conda finufft, ...): with more
# than one thread the NUFFT of a full 1 mm volume hangs. Must be set before importing torch.
os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse
import base64
import functools
import sys
import threading
import time
import warnings
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
import torchio as tio
from flask import Flask, jsonify, request
from scipy.spatial.distance import pdist

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))
from kspace_motion import MotionGPU, RandomMotionGPU  # noqa: E402
from kspace_motion import kspace_artifacts as ka  # noqa: E402

warnings.filterwarnings("ignore", message="Argument .* does not satisfy")

COLIN27_T1 = ROOT / "data" / "colin27_t1.nii.gz"
RESOLUTIONS = [1.0, 1.5, 2.0, 3.0]

# Single source of truth for the form (served to the front-end by /api/config) and the validation.
#   kind "number": one value, "range": [min, max] ("optional": can be disabled -> None),
#   "select": one of `options`, "boolean": a group switch (see `toggle`)
#   scope "trajectory" (default) changes the curves, "simulation" only the volume computation; a list
#   gives both. `advanced` fields and groups are hidden in the simple mode of the interface.
#   A select with `presets` ({value: {key: value}}) fills other fields of the form when chosen and
#   falls back to "custom" as soon as one of them is edited. A group with a `toggle` (key of a boolean
#   field) is a card that can be switched on; `summary` is its text on the saved simulations.
MOTION_PRESETS = {
    "none": {},
    "mild": {"noiseProba": 1.0, "noiseBasePars": [3, 8], "maxDisp": [1, 2], "maxRot": [1, 2],
             "swallowProba": 0.0, "swallowFrequency": [1, 4], "swallowMagnitude": [1, 2],
             "suddenProba": 0.3, "suddenFrequency": [1, 2], "suddenMagnitude": [1, 2],
             "maxGlobalDisp": [1, 2], "maxGlobalRot": [1, 2], "preserve_center_frequency_pct": 0.0},
    "moderate": {"noiseProba": 1.0, "noiseBasePars": [5, 15], "maxDisp": [2, 5], "maxRot": [2, 5],
                 "swallowProba": 0.5, "swallowFrequency": [1, 4], "swallowMagnitude": [2, 6],
                 "suddenProba": 0.7, "suddenFrequency": [1, 4], "suddenMagnitude": [2, 6],
                 "maxGlobalDisp": [3, 6], "maxGlobalRot": [3, 6], "preserve_center_frequency_pct": 0.0},
    "strong": {"noiseProba": 1.0, "noiseBasePars": [8, 15], "maxDisp": [5, 10], "maxRot": [5, 10],
               "swallowProba": 1.0, "swallowFrequency": [2, 6], "swallowMagnitude": [4, 10],
               "suddenProba": 1.0, "suddenFrequency": [2, 6], "suddenMagnitude": [4, 10],
               "maxGlobalDisp": [8, 12], "maxGlobalRot": [8, 12], "preserve_center_frequency_pct": 0.0},
}
GHOST_PRESETS = {
    "respiration": {"ghost_amplitude": 1.5, "ghost_cycles": 6, "ghost_axis": 1},
    "pulsation": {"ghost_amplitude": 0.6, "ghost_cycles": 18, "ghost_axis": 2},
}
SPIKE_PRESETS = {
    "low": {"spikes_radius": [5, 20]},
    "mid": {"spikes_radius": [20, 50]},
    "high": {"spikes_radius": [50, 90]},
}
AXIS_ROLES = [["phase", "Axe de phase (le plus lent)"], ["phase2", "Second axe de phase"],
              ["readout", "Axe de lecture"]]
BOTH = ["trajectory", "simulation"]

CONFIG = [
    {"group": "Reproductibilité", "fields": [
        {"key": "seed", "label": "Graine aléatoire", "kind": "number", "int": True, "default": 0,
         "min": 0, "max": 999999, "step": 1, "scope": BOTH,
         "help": "Même graine et mêmes paramètres : même simulation."},
        {"key": "nT", "label": "Points de la trajectoire", "kind": "number", "int": True, "default": 200,
         "min": 10, "max": 1000, "step": 10, "advanced": True,
         "help": "Réinterpolée ensuite sur le nombre de lignes de k-space de l'axe de phase."},
    ]},
    {"group": "Mouvement du patient", "summary": "mouvement {motion_level}", "fields": [
        {"key": "motion_level", "label": "Intensité", "kind": "select", "default": "moderate",
         "options": [["custom", "Personnalisé"], ["none", "Aucun"], ["mild", "Léger"], ["moderate", "Modéré"],
                     ["strong", "Fort"]],
         "presets": MOTION_PRESETS,
         "help": "Remplit les réglages détaillés du mode avancé. « Aucun » : pas de mouvement aléatoire."},
    ]},
    {"group": "Dérive lente (bruit de Perlin)", "advanced": True, "fields": [
        {"key": "noiseProba", "label": "Probabilité", "kind": "number", "default": 1.0,
         "min": 0, "max": 1, "step": 0.05},
        {"key": "noiseBasePars", "label": "Nombre d'octaves", "kind": "range", "int": False,
         "default": [5, 15], "min": 1, "max": 20, "step": 1,
         "help": "Plus il est élevé, plus la trajectoire contient de variations rapides."},
        {"key": "maxDisp", "label": "Amplitude translation", "kind": "range", "default": [2, 5],
         "min": 0, "max": 30, "step": 0.5, "unit": "mm"},
        {"key": "maxRot", "label": "Amplitude rotation", "kind": "range", "default": [2, 5],
         "min": 0, "max": 30, "step": 0.5, "unit": "°"},
    ]},
    {"group": "Déglutitions", "advanced": True, "fields": [
        {"key": "swallowProba", "label": "Probabilité", "kind": "number", "default": 0.5,
         "min": 0, "max": 1, "step": 0.05},
        {"key": "swallowFrequency", "label": "Nombre d'événements", "kind": "range", "int": True,
         "exclusive_max": True, "default": [1, 4], "min": 0, "max": 30, "step": 1,
         "help": "Tirage entre min (inclus) et max (exclu)."},
        {"key": "swallowMagnitude", "label": "Amplitude", "kind": "range", "default": [2, 6],
         "min": 0, "max": 30, "step": 0.5, "unit": "mm / °",
         "help": "Translation en z et tangage."},
    ]},
    {"group": "Mouvements brusques", "advanced": True, "fields": [
        {"key": "suddenProba", "label": "Probabilité", "kind": "number", "default": 0.7,
         "min": 0, "max": 1, "step": 0.05},
        {"key": "suddenFrequency", "label": "Nombre d'événements", "kind": "range", "int": True,
         "exclusive_max": True, "default": [1, 4], "min": 0, "max": 30, "step": 1,
         "help": "Tirage entre min (inclus) et max (exclu)."},
        {"key": "suddenMagnitude", "label": "Amplitude", "kind": "range", "default": [2, 6],
         "min": 0, "max": 30, "step": 0.5, "unit": "mm / °"},
    ]},
    {"group": "Amplitude globale", "advanced": True, "fields": [
        {"key": "maxGlobalDisp", "label": "Écart max translation", "kind": "range", "optional": True,
         "default": [3, 6], "min": 0, "max": 50, "step": 0.5, "unit": "mm",
         "help": "La trajectoire est remise à l'échelle pour que la plus grande distance entre "
                 "deux positions vaille une valeur tirée dans cet intervalle."},
        {"key": "maxGlobalRot", "label": "Écart max rotation", "kind": "range", "optional": True,
         "default": [3, 6], "min": 0, "max": 50, "step": 0.5, "unit": "°"},
        {"key": "preserve_center_frequency_pct", "label": "Centre du k-space préservé",
         "kind": "number", "default": 0.0, "min": 0, "max": 1, "step": 0.05,
         "help": "Fraction centrale de l'acquisition sans mouvement (limite le flou global)."},
    ]},
    {"group": "Ghosting (mouvement périodique)", "toggle": "ghost_enabled",
     "summary": "ghosting {ghost_axis} {ghost_amplitude} × {ghost_cycles}", "fields": [
        {"key": "ghost_enabled", "label": "Ghosting", "kind": "boolean", "default": False},
        {"key": "ghost_source", "label": "Source", "kind": "select", "default": "respiration",
         "options": [["custom", "Personnalisé"], ["respiration", "Respiration"], ["pulsation", "Pulsation"]],
         "presets": GHOST_PRESETS},
        {"key": "ghost_amplitude", "label": "Amplitude", "kind": "number", "default": 1.5,
         "min": 0.1, "max": 10, "step": 0.1, "unit": "mm / °"},
        {"key": "ghost_cycles", "label": "Cycles pendant l'acquisition", "kind": "number", "int": True,
         "default": 6, "min": 1, "max": 30, "step": 1, "hint": "ghost_spacing",
         "help": "Un mouvement sinusoïdal ajouté à la trajectoire : l'objet est répliqué le long de "
                 "l'axe de phase, avec un écart égal au champ de vue divisé par le nombre de cycles."},
        {"key": "ghost_axis", "label": "Axe du mouvement", "kind": "select", "default": 1,
         "options": [[0, "Translation x (gauche-droite)"], [1, "Translation y (postéro-antérieur)"],
                     [2, "Translation z (inféro-supérieur)"], [3, "Rotation autour de x"],
                     [4, "Rotation autour de y"], [5, "Rotation autour de z"]],
         "help": "Degré de liberté sur lequel la sinusoïde est simulée (translation en mm, rotation en °)."},
    ]},
    {"group": "Bruit (Ricien)", "toggle": "noise_enabled", "scope": "simulation", "summary": "SNR {noise_snr} dB",
     "fields": [
        {"key": "noise_enabled", "label": "Bruit", "kind": "boolean", "default": False},
        {"key": "noise_snr", "label": "Rapport signal/bruit", "kind": "number", "default": 25,
         "min": 0, "max": 60, "step": 1, "unit": "dB",
         "help": "Bruit gaussien complexe ajouté au k-space ; le module de l'image est alors Ricien. "
                 "Niveau de signal : moyenne des tissus."},
    ]},
    {"group": "Pics dans le k-space", "toggle": "spikes_enabled", "scope": "simulation",
     "summary": "{spikes_count} pic(s) {spikes_amplitude} %", "fields": [
        {"key": "spikes_enabled", "label": "Pics", "kind": "boolean", "default": False},
        {"key": "spikes_count", "label": "Nombre de pics", "kind": "number", "int": True, "default": 3,
         "min": 1, "max": 20, "step": 1},
        {"key": "spikes_amplitude", "label": "Intensité des franges", "kind": "number", "default": 10,
         "min": 0.1, "max": 100, "step": 0.5, "unit": "%",
         "help": "Amplitude de chaque motif de franges, en % du signal des tissus (« spike noise »)."},
        {"key": "spikes_frequency", "label": "Fréquence des franges", "kind": "select", "default": "mid",
         "options": [["custom", "Personnalisée"], ["low", "Basse (bandes larges)"], ["mid", "Moyenne"],
                     ["high", "Haute (franges fines)"]], "presets": SPIKE_PRESETS},
        {"key": "spikes_radius", "label": "Distance au centre du k-space", "kind": "range", "advanced": True,
         "default": [20, 50], "min": 1, "max": 150, "step": 1, "unit": "%"},
    ]},
    {"group": "Gibbs (troncature)", "toggle": "gibbs_enabled", "scope": "simulation",
     "summary": "Gibbs {gibbs_fraction} %", "fields": [
        {"key": "gibbs_enabled", "label": "Gibbs", "kind": "boolean", "default": False},
        {"key": "gibbs_fraction", "label": "K-space conservé", "kind": "number", "default": 60,
         "min": 10, "max": 100, "step": 1, "unit": "%",
         "help": "Les hautes fréquences au-delà sont supprimées : oscillations au bord des structures."},
        {"key": "gibbs_axes", "label": "Axes tronqués", "kind": "select", "default": "all", "advanced": True,
         "options": [["all", "Les trois axes"], *AXIS_ROLES]},
    ]},
    {"group": "Flou (filtre passe-bas)", "toggle": "blur_enabled", "scope": "simulation",
     "summary": "flou {blur_cutoff} %", "fields": [
        {"key": "blur_enabled", "label": "Flou", "kind": "boolean", "default": False},
        {"key": "blur_cutoff", "label": "Largeur du filtre", "kind": "number", "default": 40,
         "min": 5, "max": 100, "step": 1, "unit": "%",
         "help": "Plus elle est petite, plus l'image est floue."},
        {"key": "blur_filter", "label": "Type de filtre", "kind": "select", "default": "gaussian",
         "advanced": True, "options": [["gaussian", "Gaussien"], ["hamming", "Hamming"]]},
    ]},
    {"group": "Décroissance T2* (flou en lecture)", "toggle": "t2star_enabled", "scope": "simulation",
     "summary": "T2* {t2star_ratio}", "fields": [
        {"key": "t2star_enabled", "label": "T2*", "kind": "boolean", "default": False},
        {"key": "t2star_ratio", "label": "T2* / durée de lecture", "kind": "number", "default": 0.8,
         "min": 0.1, "max": 10, "step": 0.1,
         "help": "Le signal décroît pendant la lecture : flou le long de l'axe de lecture, d'autant "
                 "plus fort que le rapport est petit."},
    ]},
    {"group": "Lignes manquantes ou corrompues", "toggle": "dropout_enabled", "scope": "simulation",
     "summary": "{dropout_pct} % de lignes perdues", "fields": [
        {"key": "dropout_enabled", "label": "Lignes manquantes", "kind": "boolean", "default": False},
        {"key": "dropout_pct", "label": "Part des lignes touchées", "kind": "number", "default": 5,
         "min": 0.5, "max": 50, "step": 0.5, "unit": "%"},
        {"key": "dropout_unit", "label": "Unité", "kind": "select", "default": "planes", "advanced": True,
         "options": [["planes", "Plans entiers (stries)"], ["lines", "Lignes isolées"]]},
        {"key": "dropout_mode", "label": "Répartition", "kind": "select", "default": "random",
         "advanced": True, "options": [["random", "Aléatoire"], ["grouped", "Un bloc consécutif"]]},
        {"key": "dropout_gain", "label": "Facteur appliqué", "kind": "number", "default": 0.0,
         "min": 0, "max": 2, "step": 0.1, "advanced": True,
         "help": "0 : données perdues ; 0,5 : atténuées ; 2 : amplifiées."},
        {"key": "dropout_keep_center", "label": "Centre épargné", "kind": "number", "default": 0.0,
         "min": 0, "max": 90, "step": 5, "unit": "%", "advanced": True},
    ]},
    {"group": "Fourier partiel", "toggle": "pf_enabled", "scope": "simulation",
     "summary": "Fourier partiel {pf_fraction} %", "fields": [
        {"key": "pf_enabled", "label": "Fourier partiel", "kind": "boolean", "default": False},
        {"key": "pf_fraction", "label": "K-space acquis", "kind": "number", "default": 65,
         "min": 50, "max": 95, "step": 1, "unit": "%",
         "help": "Le reste est rempli de zéros : perte de résolution et oscillations."},
        {"key": "pf_axis", "label": "Axe", "kind": "select", "default": "phase", "advanced": True,
         "options": AXIS_ROLES},
    ]},
    {"group": "Repliement (aliasing)", "toggle": "alias_enabled", "scope": "simulation",
     "summary": "repliement ×{alias_factor}", "fields": [
        {"key": "alias_enabled", "label": "Repliement", "kind": "boolean", "default": False},
        {"key": "alias_factor", "label": "Facteur d'accélération", "kind": "number", "int": True,
         "default": 2, "min": 2, "max": 4, "step": 1,
         "help": "Une ligne sur N est conservée, sans reconstruction parallèle : l'image est répétée "
                 "N fois à 1/N de l'intensité."},
        {"key": "alias_axis", "label": "Axe", "kind": "select", "default": "phase", "advanced": True,
         "options": AXIS_ROLES},
    ]},
    {"group": "Zipper (fuite RF)", "toggle": "zipper_enabled", "scope": "simulation",
     "summary": "zipper {zipper_amplitude} %", "fields": [
        {"key": "zipper_enabled", "label": "Zipper", "kind": "boolean", "default": False},
        {"key": "zipper_amplitude", "label": "Intensité", "kind": "number", "default": 30,
         "min": 1, "max": 200, "step": 1, "unit": "%",
         "help": "Ligne pointillée le long de l'axe de lecture, au centre de l'image."},
        {"key": "zipper_position", "label": "Fréquence de la fuite", "kind": "number", "default": 35,
         "min": 5, "max": 100, "step": 5, "unit": "%", "advanced": True,
         "help": "Règle l'espacement des pointillés."},
    ]},
    {"group": "Offset du récepteur (point central)", "toggle": "dc_enabled", "scope": "simulation",
     "summary": "offset {dc_amplitude} %", "fields": [
        {"key": "dc_enabled", "label": "Offset", "kind": "boolean", "default": False},
        {"key": "dc_amplitude", "label": "Intensité", "kind": "number", "default": 100,
         "min": 1, "max": 1000, "step": 10, "unit": "%",
         "help": "Une constante ajoutée au k-space donne un voxel brillant au centre de l'image."},
    ]},
    {"group": "Saturation du récepteur", "toggle": "clip_enabled", "scope": "simulation",
     "summary": "saturation {clip_pct} %", "fields": [
        {"key": "clip_enabled", "label": "Saturation", "kind": "boolean", "default": False},
        {"key": "clip_pct", "label": "Échantillons saturés", "kind": "number", "default": 0.5,
         "min": 0.01, "max": 20, "step": 0.1, "unit": "%",
         "help": "Les plus grandes valeurs du k-space (le centre) sont écrêtées."},
    ]},
    {"group": "Acquisition simulée", "fields": [
        {"key": "phase_axis", "label": "Axe de phase (le plus lent)", "kind": "select", "scope": "simulation",
         "advanced": True, "default": 1, "options": [[0, "x (gauche-droite)"], [1, "y (postéro-antérieur)"],
                                                     [2, "z (inféro-supérieur)"]]},
        {"key": "resolution", "label": "Résolution", "kind": "select", "scope": "simulation",
         "default": 2.0, "options": [[r, f"{r:g} mm"] for r in RESOLUTIONS],
         "help": "1 mm : volume 181×217×181, calcul de quelques secondes."},
        {"key": "oversampling_pct", "label": "Suréchantillonnage (padding)", "kind": "number", "scope": "simulation",
         "advanced": True, "default": 30, "min": 0, "max": 100, "step": 5, "unit": "%",
         "help": "Zéros ajoutés autour du volume avant la simulation (puis retirés). Le mouvement agit dans "
                 "le Fourier, où ce qui sort d'un côté du champ de vue rentre par l'autre : le padding "
                 "supprime ces bandes de bord. Augmente le temps de calcul."},
        {"key": "coregistration", "label": "Corégistration avec l'original", "kind": "select",
         "scope": "simulation", "default": "none",
         "options": [["none", "Aucune"], ["elastix", "Rigide (Elastix)"], ["torchreg", "Affine (torchreg)"]],
         "help": "Après la simulation, l'image bougée est recalée sur l'image originale. "
                 "Elastix : transformation rigide, rapide. torchreg : affine, plus lent à haute résolution."},
    ]},
]
# a group without scope of its own inherits the scope of its fields: the default is "trajectory"
for _group in CONFIG:
    for _field in _group["fields"]:
        if "scope" in _group and "scope" not in _field:
            _field["scope"] = _group["scope"]
COREGISTRATION_NAMES = {"elastix": "Elastix (rigide)", "torchreg": "torchreg (affine)"}
FIELDS = {f["key"]: f for g in CONFIG for f in g["fields"]}

app = Flask(__name__, static_folder=str(ROOT / "static"), static_url_path="/static")
_lock = threading.Lock()   # torch's global RNG and the NUFFT are used by one request at a time


# ----------------------------------------------------------------------------------------------
# validation
# ----------------------------------------------------------------------------------------------
def _number(field, value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value):
        raise ValueError(f"« {field['label']} » doit être un nombre.")
    if not field["min"] <= value <= field["max"]:
        raise ValueError(f"« {field['label']} » doit être entre {field['min']} et {field['max']}.")
    return int(round(value)) if field.get("int") else float(value)


def scopes_of(field):
    scope = field.get("scope", "trajectory")
    return scope if isinstance(scope, list) else [scope]


def validate(payload, scope="trajectory"):
    """Check the form values against CONFIG and return a clean dict (raises ValueError)."""
    if not isinstance(payload, dict):
        raise ValueError("Requête invalide.")
    clean = {}
    for key, field in FIELDS.items():
        if scope not in scopes_of(field):
            continue
        value = payload.get(key, field["default"])
        if field["kind"] == "boolean":
            if not isinstance(value, bool):
                raise ValueError(f"« {field['label']} » doit être vrai ou faux.")
            clean[key] = value
        elif field["kind"] == "number":
            clean[key] = _number(field, value)
        elif field["kind"] == "select":
            allowed = [o[0] for o in field["options"]]
            if value not in allowed:
                raise ValueError(f"« {field['label']} » : valeur non autorisée.")
            clean[key] = value
        else:  # range
            if value is None:
                if not field.get("optional"):
                    raise ValueError(f"« {field['label']} » est obligatoire.")
                clean[key] = None
                continue
            if not isinstance(value, (list, tuple)) or len(value) != 2:
                raise ValueError(f"« {field['label']} » doit être un intervalle [min, max].")
            lo, hi = (_number(field, v) for v in value)
            if lo > hi or (field.get("exclusive_max") and lo >= hi):
                relation = "<" if field.get("exclusive_max") else "<="
                raise ValueError(f"« {field['label']} » : il faut min {relation} max.")
            clean[key] = [lo, hi]
    return clean


# ----------------------------------------------------------------------------------------------
# trajectory
# ----------------------------------------------------------------------------------------------
GHOST_POINTS_PER_CYCLE = 8


def add_ghosting(trajectory, amplitude, cycles, axis):
    """Add a sinusoid of `cycles` periods over the acquisition on one degree of freedom `axis`
    (0-2: translations x, y, z in mm; 3-5: rotations about x, y, z in degrees). The
    trajectory is resampled when it has too few points to describe the sinusoid."""
    n = max(trajectory.shape[1], GHOST_POINTS_PER_CYCLE * cycles)
    if n > trajectory.shape[1]:
        old, new = np.linspace(0, 1, trajectory.shape[1]), np.linspace(0, 1, n)
        trajectory = np.array([np.interp(new, old, row) for row in trajectory])
    trajectory = trajectory.copy()
    trajectory[axis] += amplitude * np.sin(2 * np.pi * cycles * np.linspace(0, 1, n))
    return trajectory


def simulate_trajectory(p):
    """Euler motion parameters (6, n): 3 translations (mm) then 3 rotations (degrees): the random
    motion (none if the intensity is "none") plus the periodic motion of the ghosting if it is on."""
    if p["motion_level"] == "none":
        trajectory = np.zeros((6, p["nT"]))
    else:
        trajectory = simulate_random_trajectory(p)
    if p["ghost_enabled"]:
        trajectory = add_ghosting(trajectory, p["ghost_amplitude"], p["ghost_cycles"], p["ghost_axis"])
    return trajectory


def simulate_random_trajectory(p):
    transform = RandomMotionGPU(
        nT=p["nT"],
        maxDisp=tuple(p["maxDisp"]),
        maxRot=tuple(p["maxRot"]),
        noiseBasePars=(*p["noiseBasePars"], p["noiseProba"]),
        swallowFrequency=(*p["swallowFrequency"], p["swallowProba"]),
        swallowMagnitude=tuple(p["swallowMagnitude"]),
        suddenFrequency=(*p["suddenFrequency"], p["suddenProba"]),
        suddenMagnitude=tuple(p["suddenMagnitude"]),
        maxGlobalDisp=tuple(p["maxGlobalDisp"]) if p["maxGlobalDisp"] else None,
        maxGlobalRot=tuple(p["maxGlobalRot"]) if p["maxGlobalRot"] else None,
        preserve_center_frequency_pct=p["preserve_center_frequency_pct"],
    )
    torch.manual_seed(p["seed"])
    transform._simulate_random_trajectory()
    return transform.euler_motion_params.numpy()


def colin27_fov_mm():
    """Field of view of the template along x, y, z (mm), from the header only."""
    header = nib.load(COLIN27_T1).header
    return [float(n * z) for n, z in zip(header.get_data_shape()[:3], header.get_zooms()[:3])]


@app.get("/api/config")
def api_config():
    return jsonify(config=CONFIG, fov_mm=colin27_fov_mm())


@app.post("/api/trajectory")
def api_trajectory():
    try:
        params = validate(request.get_json(silent=True))
    except ValueError as error:
        return jsonify(error=str(error)), 400
    with _lock:
        trajectory = simulate_trajectory(params)
    return jsonify(
        params=np.round(trajectory, 4).tolist(),
        stats={
            "max_translation": float(pdist(trajectory[:3].T).max()),
            "max_rotation": float(pdist(trajectory[3:].T).max()),
        },
    )


# ----------------------------------------------------------------------------------------------
# simulation on Colin27
# ----------------------------------------------------------------------------------------------
@functools.lru_cache(maxsize=len(RESOLUTIONS))
def colin27(resolution):
    """Colin27 T1w intensity-normalised to [0, 1], resampled to an isotropic resolution (mm)."""
    subject = tio.Subject(t1=tio.ScalarImage(COLIN27_T1))
    if resolution != 1.0:
        subject = tio.Resample(resolution)(subject)
    subject = tio.RescaleIntensity(out_min_max=(0, 1), percentiles=(0.5, 99.5))(subject)
    subject["t1"].load()
    return subject


def _encode(volume, vmax):
    quantised = np.clip(volume / vmax * 255, 0, 255).round().astype(np.uint8)
    return base64.b64encode(np.ascontiguousarray(quantised).tobytes()).decode("ascii")


def padding_for(shape, oversampling_pct):
    """Voxels added on each side of every axis (same rule as the CPU `oversampling_pct`)."""
    return tuple(int(np.ceil(size * oversampling_pct / 100 / 2)) for size in shape)


def crop(volume, pads):
    return volume[tuple(slice(p, size - p) for p, size in zip(pads, volume.shape))]


def build_artifacts(p):
    """The k-space artefacts switched on in the form, as `kspace_artifacts` objects. Each random one
    has its own seed derived from the seed of the form."""
    seed, artifacts = p["seed"], []
    roles = (lambda axes: list(ka.ROLES) if axes == "all" else [axes])
    if p["gibbs_enabled"]:
        artifacts.append(ka.Truncation(p["gibbs_fraction"] / 100, axes=roles(p["gibbs_axes"])))
    if p["blur_enabled"]:
        artifacts.append(ka.LowPassFilter(p["blur_cutoff"] / 100, kind=p["blur_filter"]))
    if p["t2star_enabled"]:
        artifacts.append(ka.T2StarDecay(p["t2star_ratio"]))
    if p["pf_enabled"]:
        artifacts.append(ka.PartialFourier(p["pf_fraction"] / 100, axis=p["pf_axis"]))
    if p["alias_enabled"]:
        artifacts.append(ka.Undersampling(p["alias_factor"], axis=p["alias_axis"]))
    if p["dropout_enabled"]:
        artifacts.append(ka.LineDropout(p["dropout_pct"] / 100, unit=p["dropout_unit"], mode=p["dropout_mode"],
                                        gain=p["dropout_gain"], keep_center=p["dropout_keep_center"] / 100,
                                        seed=seed + 2))
    if p["spikes_enabled"]:
        artifacts.append(ka.Spikes(p["spikes_count"], p["spikes_amplitude"] / 100,
                                   radius=tuple(r / 100 for r in p["spikes_radius"]), seed=seed + 1))
    if p["zipper_enabled"]:
        artifacts.append(ka.Zipper(p["zipper_amplitude"] / 100, position=p["zipper_position"] / 100))
    if p["dc_enabled"]:
        artifacts.append(ka.DCOffset(p["dc_amplitude"] / 100))
    if p["clip_enabled"]:
        artifacts.append(ka.Clipping(p["clip_pct"]))
    if p["noise_enabled"]:
        artifacts.append(ka.RicianNoise(p["noise_snr"], seed=seed + 3))
    return artifacts


def run_motion(resolution, phase_axis, trajectory, oversampling_pct=0.0, artifacts=None):
    """Simulate the motion and the k-space artefacts on zero-padded Colin27. Returns the *padded*
    (original, moved) volumes, the padding of each axis and the computing time: cropping is left to
    the caller so that a registration can still use the padded field of view. `trajectory` is in mm
    (translations) and degrees (rotations), `resolution` is the isotropic voxel size in mm."""
    order = tuple([a for a in (0, 1, 2) if a != phase_axis] + [phase_axis])
    subject = colin27(resolution)
    original = subject["t1"].data[0].numpy()
    pads = padding_for(original.shape, oversampling_pct)
    padded = np.pad(original, [(p, p) for p in pads])
    padded_subject = tio.Subject(t1=tio.ScalarImage(tensor=torch.from_numpy(padded)[None],
                                                    affine=subject["t1"].affine))
    # the trajectory is in mm and degrees, MotionGPU translates in voxels
    trajectory = np.array(trajectory, dtype=np.float64)
    trajectory[:3] /= resolution
    started = time.perf_counter()
    corrupted = MotionGPU(euler_motion_params=trajectory, kspace_order=order,
                          kspace_artifacts=artifacts)(padded_subject)
    elapsed = time.perf_counter() - started
    moved = corrupted["t1"].data[0].numpy().astype(np.float32)
    return padded, moved, pads, elapsed


def run_coregistration(method, subject, original, moved):
    """Register the moved volume to the original one, on the original grid. Returns (volume, seconds).
    `original` and `moved` are the (possibly padded) arrays, `subject` gives the affine."""
    # imported here: they pull in itk / torchreg, which are only needed when the option is used
    if method == "elastix":
        from kspace_motion.coregister_itk import Coregister
    else:
        from kspace_motion.coregister_torchreg import Coregister
    affine = subject["t1"].affine
    pair = tio.Subject(
        ref=tio.ScalarImage(tensor=torch.from_numpy(np.ascontiguousarray(original))[None], affine=affine),
        moving=tio.ScalarImage(tensor=torch.from_numpy(np.ascontiguousarray(moved))[None], affine=affine),
    )
    started = time.perf_counter()
    registered = Coregister(target="ref")(pair)["moving"].data[0].numpy().astype(np.float32)
    return registered, time.perf_counter() - started


def _metrics(original, other):
    value_range = float(original.max() - original.min())
    return {
        "correlation": float(np.corrcoef(original.ravel(), other.ravel())[0, 1]),
        "nrmse": float(np.sqrt(np.mean((original - other) ** 2)) / value_range),
    }


@app.post("/api/simulate")
def api_simulate():
    body = request.get_json(silent=True)
    try:
        options = validate(body, scope="simulation")
        trajectory = np.asarray(body.get("trajectory"), dtype=np.float64)
        if trajectory.ndim != 2 or trajectory.shape[0] != 6 or trajectory.shape[1] < 2:
            raise ValueError("La trajectoire doit avoir la forme (6, N).")
        if not np.isfinite(trajectory).all():
            raise ValueError("La trajectoire contient des valeurs non finies.")
    except (ValueError, TypeError) as error:
        return jsonify(error=str(error)), 400

    method = options["coregistration"]
    registered, coregistration, coregistration_error = None, None, None
    with _lock:
        original, moved, pads, elapsed = run_motion(options["resolution"], options["phase_axis"], trajectory,
                                                    options["oversampling_pct"], build_artifacts(options))
        if method != "none":
            try:
                registered, seconds = run_coregistration(method, colin27(options["resolution"]), original, moved)
                coregistration = {"method": COREGISTRATION_NAMES[method], "seconds": seconds}
            except Exception as error:   # keep the simulation, report why the registration failed
                app.logger.exception("coregistration failed")
                coregistration_error = f"La corégistration a échoué : {error}"
    original, moved = crop(original, pads), crop(moved, pads)
    if registered is not None:
        registered = crop(registered, pads)
        coregistration.update(_metrics(original, registered))

    vmax = max(float(original.max()), float(moved.max()),
               float(registered.max()) if registered is not None else 0.0, 1e-6)
    return jsonify(
        shape=list(original.shape),
        spacing=options["resolution"],
        original=_encode(original, vmax),
        corrupted=_encode(moved, vmax),
        coregistered=_encode(registered, vmax) if registered is not None else None,
        metrics={**_metrics(original, moved), "seconds": elapsed},
        coregistration=coregistration,
        coregistration_error=coregistration_error,
    )


@app.get("/")
def index():
    return app.send_static_file("index.html")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    print(f"k-space motion webapp: http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port, threaded=True)
