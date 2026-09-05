"""Shared visual and naming style for manuscript outputs."""

from __future__ import annotations

MODE_ORDER: tuple[str, ...] = ("unweighted", "source", "target", "both", "crump", "overlap")

MODE_LABELS = {
    "unweighted": "Unweighted",
    "source": "Source-weighted",
    "target": "Target-weighted",
    "both": "Doubly weighted",
    "crump": "Crump-trimmed",
    "overlap": "Overlap-weighted",
}

MODE_FIGURE_LABELS = {
    "unweighted": "Unweighted D-SOS",
    "source": "Source-weighted",
    "target": "Target-weighted",
    "both": "Doubly weighted",
    "crump": "Crump-trimmed",
    "overlap": "Overlap-weighted",
}

MODE_COLORS = {
    "unweighted": "#222222",
    "source": "#1b9e77",
    "target": "#d95f02",
    "both": "#7570b3",
    "crump": "#e7298a",
    "overlap": "#e6ab02",
}

MODE_MARKERS = {
    "unweighted": "o",
    "source": "s",
    "target": "^",
    "both": "D",
    "crump": "v",
    "overlap": "p",
}

SCENARIO_ORDER: tuple[str, ...] = ("source_only", "target_only", "both_sides")

SCENARIO_LABELS = {
    "source_only": "Source-only\ncontamination",
    "target_only": "Target-only\ncontamination",
    "both_sides": "Both-side\ncontamination",
}

EXPERIMENT_STYLE = {
    "calibration": {"color": "#1b9e77", "label": "Calibration"},
    "power": {"color": "#d95f02", "label": "Power"},
}

MODE_STYLE = {
    mode: {
        "color": MODE_COLORS[mode],
        "marker": MODE_MARKERS[mode],
        "label": MODE_FIGURE_LABELS[mode],
    }
    for mode in MODE_ORDER
}
