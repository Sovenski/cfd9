"""Build pine/speculatores_v17_5_signalcard.pine FROM the v17 gold file (T13).

Spec §4.1 (FIXED v3): the detection engine stays BYTE-IDENTICAL to
``pine/speculatores_v17_presets_gold.pine`` (the 2026-06-10 parity-PASS
state). The generated diff touches ONLY:

* the indicator title (version "17.5", plus ``max_labels_count`` for the
  frozen historical cards);
* the APPENDED ``// === CALIBRATION BLOCK (generated) ===`` constants;
* the appended card display block (live conditioning for the ACTIVE signal
  per side only, F5; frozen labels + final-grade badge for history);
* input tooltips carrying the §1.4 plain-language T1/T2/T3 text, the R3
  conditional-on-match label, the H0 explanation, the F5 scope note and the
  F7 in-sample disclaimer (verbatim);
* data-window parity export plots for the card numerics incl. stop level.

The builder consumes a calibration JSON (``{"high": side_payload, "low":
side_payload}`` where ``side_payload`` is the ``calibrate_run`` §5 payload,
or a full run JSON via ``--run-json``). A synthetic sample calibration is
generated so the file builds TODAY; the real one comes from the next run.
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))

from src.scoring_v5 import SPAN_GRID  # noqa: E402
from src.v17_card.calibration import IN_SAMPLE_DISCLAIMER  # noqa: E402

logger = logging.getLogger(__name__)

V17_PINE = _REPO / "pine" / "speculatores_v17_presets_gold.pine"
OUT_PINE = _REPO / "pine" / "speculatores_v17_5_signalcard.pine"
SAMPLE_JSON = _REPO / "temp" / "sample_calibration_v17_5.json"

CALIB_MARKER = "// === CALIBRATION BLOCK (generated) ==="
CALIB_END = "// === END CALIBRATION BLOCK (generated) ==="

V17_5_TITLE = (
    'indicator("Speculatores V17.5 Signal Card - Per-Side Regimes '
    '(GPU CMA-ES)", overlay=true, max_bars_back=5000, max_labels_count=500)'
)

# --- tooltip texts (test-anchored; §1.4 / R3 / F5 / F7 / H0) ---------------
TIER_TEXT = (
    "T1 = a major turning point: the highest/lowest price of roughly the "
    "surrounding year or more. T2 = a meaningful swing turn: extremum of "
    "the surrounding quarter, not strong enough to be T1. T3 = a minor "
    "swing turn (weeks-scale). Display vocabulary only."
)
SCOPE_NOTE = (
    "Live conditioning runs ONLY for the most recent (active) signal per "
    "side (Pine object limits, F5); historical signals get a FROZEN card "
    "with the values computed at fire time, plus a final-grade badge once "
    "their span resolves."
)
LEGEND_TEXT = (
    "Card legend: P(T2+)/P(T1) are shown as a cluster-bootstrap band "
    "(5th-95th pct envelope, F6), never a point value; the live band "
    "applies the conditioning and L-clamp to S_lo and S_hi separately "
    "(a pragmatic envelope). move = expected move if this is a real turn "
    "(conditional on match, R3). hold = span-clock E[hold] in bars. "
    "DISCLAIMER: " + IN_SAMPLE_DISCLAIMER + "."
)
H0_TEXT = (
    "H0 hard stop = the candidate pivot low: min(low[t-1], low[t]) at "
    "fire, widened once to include low[t+1] when bar t+1 closes (HIGH "
    "side mirrors with highs). Within the pivot window a breach "
    "definitionally refutes the matched-pivot hypothesis - alpha ~ 0, not "
    "a statistical stop. Breach at t+1 is close-based vs the fire-time "
    "stop; from t+2 onward the stop is FINAL and any intrabar tick through "
    "it invalidates. Stop valid ~E[hold] bars; exit by clock beyond that."
)


def _pf(v: Optional[float]) -> str:
    """Pine float literal: shortest round-trip repr; na for None/non-finite."""
    if v is None:
        return "na"
    f = float(v)
    if not np.isfinite(f):
        return "na"
    return repr(f)


def _arr(values: Optional[Sequence[float]]) -> str:
    vals = list(values or [])
    if not vals:
        return "array.new_float(0)"
    return "array.from(" + ", ".join(_pf(v) for v in vals) + ")"


def _side_payload(calibration: dict, side: str) -> tuple[dict, str]:
    if side not in calibration:
        raise ValueError(f"calibration JSON missing side {side!r} "
                         "(v17.5 needs both sides; rerun with sides=high,low)")
    payload = calibration[side]
    cal = payload.get("calibration") or {}
    if cal.get("degenerate") or not cal.get("S_R"):
        raise ValueError(f"degenerate calibration for side {side!r} "
                         "(0 signals or no survival table) — cannot build "
                         "the v17.5 card")
    h = payload.get("calibration_block_hash") or \
        payload.get("trace", {}).get("calibration_block_hash", "")
    return cal, str(h)


def _calibration_block(sides: dict[str, tuple[dict, str]]) -> str:
    lines = [CALIB_MARKER]
    for side, (cal, h) in sides.items():
        fd = cal.get("fit_diagnostics") or {}
        lines += [
            f"// {side}: hash {h}",
            f"// {side}: band={cal.get('band_method')} n_boot="
            f"{cal.get('n_boot')} seed={cal.get('seed')} n_signals="
            f"{cal.get('n_signals')} n_streams={cal.get('n_streams')} "
            f"c_r2={cal.get('c_side', {}).get('r_squared')}",
        ]
    lines.append("var array<float> CARD_GRID = "
                 + _arr([float(g) for g in SPAN_GRID]))
    for side, (cal, _h) in sides.items():
        s = side.upper()
        cs = cal.get("c_side") or {}
        lines += [
            f"var array<float> CARD_S_R_{s} = {_arr(cal['S_R'])}",
            f"var array<float> CARD_S_LO_{s} = {_arr(cal['S_lo'])}",
            f"var array<float> CARD_S_HI_{s} = {_arr(cal['S_hi'])}",
            f"var array<float> CARD_CONV_BP_{s} = "
            f"{_arr(cal.get('conviction_breakpoints'))}",
            f"float CARD_C_SIDE_{s} = {_pf(cs.get('c'))}",
            f"bool CARD_C_FALLBACK_{s} = "
            f"{'true' if cs.get('use_fallback') else 'false'}",
            f"float CARD_C_MEDIAN_{s} = {_pf(cs.get('fallback_median'))}",
            f"int CARD_CLOCK_{s} = {int(cal.get('clock_bars', 1))}",
        ]
    lines.append(CALIB_END)
    return "\n".join(lines)


_HELPERS = """\
// R4 — off-grid step-function-floor lookup: evaluates the table at the
// largest grid value <= x; S(x) = 1.0 below the 20-bar noise floor;
// arguments above the cap floor to the cap. Engine and Pine share this
// EXACT rule (spec §3.2 / §4.2 parity assertion).
card_lookup(array<float> tbl, float x) =>
    float s = 1.0
    if x >= array.get(CARD_GRID, 0)
        int idx = 0
        for j = 0 to array.size(CARD_GRID) - 1
            if array.get(CARD_GRID, j) <= x
                idx := j
        s := array.get(tbl, idx)
    s

// §3.2 THE LIVE UPDATE RULE (F1): P(N*_eff >= x | L, survived k) = 0 if
// x > L, else S(max(x, k)) / S(k) — the left span enters ONLY via the
// L-clamp (no double-counting); k counted from the candidate pivot bar.
card_cond(array<float> tbl, float x, float k, float l) =>
    float p = 0.0
    if x <= l
        float sk = card_lookup(tbl, k)
        if sk > 0.0
            p := math.min(1.0, card_lookup(tbl, math.max(x, k)) / sk)
    p

// §3.4 span clock: E[hold] = sum_j P(N*_eff >= x_j | L, k) * dx_j, cap 500.
card_ehold(array<float> tbl, float k, float l) =>
    float total = 0.0
    float prev = 0.0
    for j = 0 to array.size(CARD_GRID) - 1
        float xj = array.get(CARD_GRID, j)
        total += card_cond(tbl, xj, k, l) * (xj - prev)
        prev := xj
    math.min(total, 500.0)

// F1 — proven left-span L: causal backward scan with early exit, capped at
// the 500 grid cap and the data's left edge (mirror of compute_left_span).
card_left_span(bool lhs_is_high, int back) =>
    float v = lhs_is_high ? high[back] : low[back]
    int l = 0
    for d = 1 to 500
        float p = lhs_is_high ? high[back + d] : low[back + d]
        if na(p) or (lhs_is_high ? p > v : p < v)
            break
        l := d
    l

// §3.1 sigma_HAR — per-bar sqrt(har_forecast), annualization-free (the
// engine's read-only HAR mirror; same GK weights as calc_har_vol).
float card_gk = math.max(0.5 * math.pow(safe_log_ratio(high, low), 2) - (2.0 * math.log(2.0) - 1.0) * math.pow(safe_log_ratio(close, open), 2), 1e-10)
float card_gk_w = nz(ta.sma(card_gk, 5), card_gk)
float card_gk_m = nz(ta.sma(card_gk, 22), card_gk)
float card_sigma = math.sqrt(math.max(0.36 * card_gk + 0.28 * card_gk_w + 0.28 * card_gk_m, 1e-10))"""


def _state_block(side: str) -> str:
    """Per-side live ``var`` state + §3.5 truth-table updates + numerics."""
    s = side
    hi = side == "high"
    px = "high" if hi else "low"
    mm = "math.max" if hi else "math.min"
    tie = ">=" if hi else "<="          # row-1 tie -> the EARLIER bar
    brc = ">" if hi else "<"            # breach direction
    flag = "true" if hi else "false"
    return f"""\
// --- live card state — {s.upper()} side (F5: the most recent active signal
// --- only; §3.5 truth table; the fire branch doubles as the row-6
// --- same-side re-fire reset: old card freezes, state resets per row 1)
var int card_fire_{s} = na
var int card_i_{s} = na
var float card_stop_{s} = na
var float card_fire_stop_{s} = na
var float card_sigfire_{s} = na
var int card_L_{s} = 0
var bool card_intact_{s} = false
var bool card_final_{s} = false
var float card_conv_{s} = na
var label card_lbl_{s} = na

bool card_fire_evt_{s} = bar_index > 0 and pivot_{s} and not pivot_{s}[1]
if card_fire_evt_{s}
    // row 1 — fire-time stop = candidate pivot {px}; i = arg-extreme (tie -> earlier)
    card_stop_{s} := {mm}({px}[1], {px})
    card_fire_stop_{s} := card_stop_{s}
    card_i_{s} := {px}[1] {tie} {px} ? bar_index - 1 : bar_index
    card_fire_{s} := bar_index
    card_L_{s} := card_left_span({flag}, bar_index - card_i_{s})
    card_sigfire_{s} := card_sigma
    card_intact_{s} := true
    card_final_{s} := false
    float card_eh0_{s} = card_ehold(CARD_S_R_{s.upper()}, 0.0, card_L_{s})
    float card_em0_{s} = CARD_C_FALLBACK_{s.upper()} ? CARD_C_MEDIAN_{s.upper()} : CARD_C_SIDE_{s.upper()} * card_sigma * math.sqrt(math.max(card_eh0_{s}, 0.0))
    float card_prod_{s} = card_cond(CARD_S_R_{s.upper()}, 50.0, 0.0, card_L_{s}) * card_em0_{s}
    if array.size(CARD_CONV_BP_{s.upper()}) > 0
        int card_dec_{s} = 0
        for j = 0 to array.size(CARD_CONV_BP_{s.upper()}) - 1
            if card_prod_{s} >= array.get(CARD_CONV_BP_{s.upper()}, j)
                card_dec_{s} := j
        card_conv_{s} := card_dec_{s} * 10.0
else if card_intact_{s}
    if bar_index == card_fire_{s} + 1
        // rows 2/3 — close-based breach vs the PRE-widening fire-time stop
        if close {brc} card_fire_stop_{s}
            card_intact_{s} := false
        else
            if {px} {brc} {px}[bar_index - card_i_{s}]
                card_i_{s} := bar_index
                card_L_{s} := card_left_span({flag}, 0)
            card_stop_{s} := {mm}(card_stop_{s}, {px})
        card_final_{s} := true
    else if {px} {brc} card_stop_{s}
        // row 4 — intrabar breach of the FINAL stop
        card_intact_{s} := false

// live card numerics — na unless an intact active card (F5)
float card_k_{s} = card_intact_{s} ? bar_index - card_i_{s} : na
float card_p_t2_lo_{s} = card_intact_{s} ? card_cond(CARD_S_LO_{s.upper()}, 50.0, card_k_{s}, card_L_{s}) : na
float card_p_t2_hi_{s} = card_intact_{s} ? card_cond(CARD_S_HI_{s.upper()}, 50.0, card_k_{s}, card_L_{s}) : na
float card_p_t1_lo_{s} = card_intact_{s} ? card_cond(CARD_S_LO_{s.upper()}, 200.0, card_k_{s}, card_L_{s}) : na
float card_p_t1_hi_{s} = card_intact_{s} ? card_cond(CARD_S_HI_{s.upper()}, 200.0, card_k_{s}, card_L_{s}) : na
float card_ehold_{s} = card_intact_{s} ? card_ehold(CARD_S_R_{s.upper()}, card_k_{s}, card_L_{s}) : na
float card_move_{s} = card_intact_{s} ? (CARD_C_FALLBACK_{s.upper()} ? CARD_C_MEDIAN_{s.upper()} : CARD_C_SIDE_{s.upper()} * card_sigfire_{s} * math.sqrt(math.max(card_ehold_{s}, 0.0))) : na"""


def _label_block(side: str) -> str:
    s = side
    hi = side == "high"
    px = "high" if hi else "low"
    style = "label.style_label_down" if hi else "label.style_label_up"
    col = "color.red" if hi else "color.green"
    return f"""\
if show_card and show_card_labels and card_fire_evt_{s}
    // F5 — FROZEN card on the historical signal: fire-time values baked in
    string card_txt_{s} = "{s.upper()} card\\nP(T2+)= " + str.tostring(card_p_t2_lo_{s} * 100, "#") + "-" + str.tostring(card_p_t2_hi_{s} * 100, "#") + "% · P(T1)= " + str.tostring(card_p_t1_lo_{s} * 100, "#") + "-" + str.tostring(card_p_t1_hi_{s} * 100, "#") + "%\\nmove ±" + str.tostring(card_move_{s} * 100, "#.##") + "% | hold ~" + str.tostring(card_ehold_{s}, "#") + " bars | conv " + str.tostring(card_conv_{s}, "#") + "/100\\nstop " + str.tostring(card_stop_{s}, format.mintick)
    card_lbl_{s} := label.new(bar_index, {px}, card_txt_{s}, style={style}, color=color.new({col}, 60), textcolor=color.white, size=size.small)
if show_card and show_card_labels and not na(card_lbl_{s}) and not card_intact_{s} and card_intact_{s}[1]
    // final-grade badge: the right span resolves exactly at the stop breach
    int card_r_{s} = bar_index - card_i_{s}
    int card_eff_{s} = math.min(card_L_{s}, card_r_{s})
    string card_tier_{s} = card_eff_{s} >= 200 ? "T1" : card_eff_{s} >= 50 ? "T2" : card_eff_{s} >= 20 ? "T3" : "miss"
    label.set_text(card_lbl_{s}, label.get_text(card_lbl_{s}) + "\\nresolved: " + card_tier_{s} + " (span " + str.tostring(card_eff_{s}) + ")")"""


def _band_cell(s: str, p: str) -> str:
    return (f'card_intact_{s} ? str.tostring(card_p_{p}_lo_{s} * 100, "#") + '
            f'"-" + str.tostring(card_p_{p}_hi_{s} * 100, "#") + "%" : "-"')


def _table_block() -> str:
    rows = [
        '    table.cell(card_tbl, 0, 0, "V17.5 card", text_color=color.yellow, text_size=size.tiny)',
        '    table.cell(card_tbl, 1, 0, "HIGH (active)", text_color=color.red, text_size=size.tiny)',
        '    table.cell(card_tbl, 2, 0, "LOW (active)", text_color=color.green, text_size=size.tiny)',
        '    table.cell(card_tbl, 0, 1, "P(T2+) band", text_color=color.white, text_size=size.tiny)',
        f'    table.cell(card_tbl, 1, 1, {_band_cell("high", "t2")}, text_color=color.white, text_size=size.tiny)',
        f'    table.cell(card_tbl, 2, 1, {_band_cell("low", "t2")}, text_color=color.white, text_size=size.tiny)',
        '    table.cell(card_tbl, 0, 2, "P(T1) band", text_color=color.white, text_size=size.tiny)',
        f'    table.cell(card_tbl, 1, 2, {_band_cell("high", "t1")}, text_color=color.white, text_size=size.tiny)',
        f'    table.cell(card_tbl, 2, 2, {_band_cell("low", "t1")}, text_color=color.white, text_size=size.tiny)',
        '    table.cell(card_tbl, 0, 3, "move ± / hold", text_color=color.white, text_size=size.tiny)',
        '    table.cell(card_tbl, 1, 3, card_intact_high ? str.tostring(card_move_high * 100, "#.##") + "% / " + str.tostring(card_ehold_high, "#") + "b" : "-", text_color=color.white, text_size=size.tiny)',
        '    table.cell(card_tbl, 2, 3, card_intact_low ? str.tostring(card_move_low * 100, "#.##") + "% / " + str.tostring(card_ehold_low, "#") + "b" : "-", text_color=color.white, text_size=size.tiny)',
        '    table.cell(card_tbl, 0, 4, "stop / k / conv", text_color=color.white, text_size=size.tiny)',
        '    table.cell(card_tbl, 1, 4, card_intact_high ? str.tostring(card_stop_high, format.mintick) + " / " + str.tostring(card_k_high, "#") + " / " + str.tostring(card_conv_high, "#") : "-", text_color=color.white, text_size=size.tiny)',
        '    table.cell(card_tbl, 2, 4, card_intact_low ? str.tostring(card_stop_low, format.mintick) + " / " + str.tostring(card_k_low, "#") + " / " + str.tostring(card_conv_low, "#") : "-", text_color=color.white, text_size=size.tiny)',
    ]
    return ("if show_card and barstate.islast\n"
            "    var table card_tbl = table.new(position.top_right, 3, 5, "
            "bgcolor=color.new(color.black, 75), border_width=1)\n"
            + "\n".join(rows))


def _plots_block() -> str:
    lines = [
        "// visible H0 stop lines (plot(intact ? stop : na) — §3.5 decided "
        "post-invalidation semantic: na from the invalidation bar onward)",
        'plot(show_card_stop and card_intact_high ? card_stop_high : na, '
        '"H0 stop HIGH", color=color.new(color.red, 0), '
        'style=plot.style_linebr)',
        'plot(show_card_stop and card_intact_low ? card_stop_low : na, '
        '"H0 stop LOW", color=color.new(color.green, 0), '
        'style=plot.style_linebr)',
        "// V17.5 parity exports (§4.2): card numerics + stop, both sides",
    ]
    for s in ("high", "low"):
        for q in ("p_t2_lo", "p_t2_hi", "p_t1_lo", "p_t1_hi", "move",
                  "ehold"):
            lines.append(f'plot(card_{q}_{s}, "card_{q}_{s}", '
                         f'display=display.data_window)')
        lines.append(f'plot(card_intact_{s} ? card_stop_{s} : na, '
                     f'"card_stop_{s}", display=display.data_window)')
    return "\n".join(lines)


def _v17_5_block(sides: dict[str, tuple[dict, str]]) -> str:
    banner = (
        "// " + "=" * 76 + "\n"
        "// === V17.5 SIGNAL CARD (generated by temp/build_pine_v17_5.py) "
        "===========\n"
        "// Detection engine ABOVE is byte-identical to "
        "speculatores_v17_presets_gold\n"
        "// (parity PASS 2026-06-10). Everything below is "
        "display/calibration only —\n"
        "// it READS detection outputs and never feeds back into them "
        "(spec §0).\n"
        "// " + "=" * 76)
    inputs = (
        'string GRP_CARD = "Signal Card (V17.5)"\n'
        f'bool show_card = input.bool(true, "Show Signal Card", '
        f'group=GRP_CARD, tooltip="{LEGEND_TEXT}")\n'
        f'bool show_card_labels = input.bool(true, "Frozen Cards On '
        f'Historical Signals", group=GRP_CARD, '
        f'tooltip="{TIER_TEXT} {SCOPE_NOTE}")\n'
        f'bool show_card_stop = input.bool(true, "Plot Active Stop (H0)", '
        f'group=GRP_CARD, tooltip="{H0_TEXT}")')
    parts = [banner, inputs, _calibration_block(sides), _HELPERS,
             _state_block("high"), _state_block("low"),
             _label_block("high"), _label_block("low"),
             _table_block(), _plots_block()]
    return "\n\n".join(parts) + "\n"


def _assert_diff_surface(v17_text: str, built: str) -> None:
    """The HARD §4.1 claim: title-line edit + append-only, nothing else."""
    orig = v17_text.splitlines()
    gen = built.splitlines()
    if len(gen) <= len(orig):
        raise AssertionError("v17.5 must strictly APPEND content")
    diffs = [i for i, (a, b) in enumerate(zip(gen[:len(orig)], orig))
             if a != b]
    if len(diffs) != 1 or "indicator(" not in orig[diffs[0]]:
        raise AssertionError(
            f"detection lines modified beyond the title: lines {diffs}")
    if "V17.5" not in gen[diffs[0]]:
        raise AssertionError("title must carry the V17.5 version string")


def build_pine_v17_5(v17_text: str, calibration: dict) -> str:
    """Pure builder: v17 gold text + calibration JSON -> v17.5 text.

    Deterministic (same inputs -> byte-identical output); raises
    ``ValueError`` on a missing/degenerate side and ``AssertionError`` if
    the generated diff surface violates the §4.1 claim.
    """
    sides = {s: _side_payload(calibration, s) for s in ("high", "low")}
    out_lines = []
    replaced = 0
    for ln in v17_text.splitlines():
        if ln.startswith("indicator(") and replaced == 0:
            out_lines.append(V17_5_TITLE)
            replaced += 1
        else:
            out_lines.append(ln)
    if replaced != 1:
        raise ValueError("v17 indicator() title line not found")
    built = "\n".join(out_lines) + "\n" + _v17_5_block(sides)
    _assert_diff_surface(v17_text, built)
    return built


# ---------------------------------------------------------------------------
# parse-back (test + parity tooling)
# ---------------------------------------------------------------------------

_ARRAY_RE = re.compile(
    r"^var array<float> CARD_(\w+) = (?:array\.from\((.*)\)|array\.new_float\(0\))$")
_SCALAR_RE = re.compile(r"^(?:float|int|bool) CARD_(\w+) = (.+)$")


def _scalar(raw: str):
    if raw == "na":
        return None
    if raw in ("true", "false"):
        return raw == "true"
    return int(raw) if re.fullmatch(r"-?\d+", raw) else float(raw)


def parse_calibration_block(pine_text: str) -> dict:
    """Extract the generated calibration constants back out of the Pine."""
    raw: dict[str, object] = {}
    in_block = False
    for ln in pine_text.splitlines():
        if ln.strip() == CALIB_MARKER:
            in_block = True
            continue
        if ln.strip() == CALIB_END:
            break
        if not in_block:
            continue
        m = _ARRAY_RE.match(ln)
        if m:
            vals = m.group(2)
            raw[m.group(1)] = ([float(v) for v in vals.split(", ")]
                               if vals is not None else [])
            continue
        m = _SCALAR_RE.match(ln)
        if m:
            raw[m.group(1)] = _scalar(m.group(2).strip())
    out: dict = {"grid": [int(v) for v in raw["GRID"]]}
    for side in ("high", "low"):
        s = side.upper()
        out[side] = {
            "S_R": raw[f"S_R_{s}"], "S_lo": raw[f"S_LO_{s}"],
            "S_hi": raw[f"S_HI_{s}"],
            "conviction_breakpoints": raw[f"CONV_BP_{s}"],
            "c": raw[f"C_SIDE_{s}"], "use_fallback": raw[f"C_FALLBACK_{s}"],
            "fallback_median": raw[f"C_MEDIAN_{s}"],
            "clock_bars": raw[f"CLOCK_{s}"],
        }
    return out


# ---------------------------------------------------------------------------
# sample calibration — synthetic, deterministic, builds TODAY
# ---------------------------------------------------------------------------


def _synthetic_stream(seed: int, n: int = 720, period: int = 240) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    t = np.arange(n, dtype=float)
    close = (100.0 + 12.0 * np.sin(2.0 * np.pi * t / period) + 0.01 * t
             + rng.normal(0.0, 0.05, n))
    spread = np.abs(rng.normal(0.25, 0.05, n))
    return pd.DataFrame({
        "open": close + rng.normal(0.0, 0.05, n),
        "high": close + spread, "low": close - spread,
        "close": close, "volume": np.ones(n),
    })


def _fires_at_extremes(arr: np.ndarray, kind: str,
                       period: int = 240) -> np.ndarray:
    """Fires at each cycle extreme (direct hits) plus leg fires (short and
    long right-spans) so the sample KM tables carry real, varied events."""
    sig = np.zeros(arr.shape[0], dtype=bool)
    for c0 in range(0, arr.shape[0], period):
        w = arr[c0:c0 + period]
        if w.shape[0] < 40:
            continue
        ext = c0 + int(np.argmax(w) if kind == "max" else np.argmin(w))
        for off in (0, -37, 23):
            idx = ext + off
            if 1 <= idx < arr.shape[0]:
                sig[idx] = True
    return sig


def make_sample_calibration(seed: int = 0, n_boot: int = 50) -> dict:
    """Synthetic two-stream calibration so the v17.5 file builds TODAY.

    Real runs replace this via ``--run-json`` (the §5 run payload).
    """
    from src.v17_card.calibration import calibrate_run

    frames: dict[str, pd.DataFrame] = {}
    lows: dict[str, np.ndarray] = {}
    highs: dict[str, np.ndarray] = {}
    for k in range(2):
        sid = f"SYN{k}_1D"
        df = _synthetic_stream(seed + k + 1)
        frames[sid] = df
        lows[sid] = _fires_at_extremes(df["low"].to_numpy(), "min")
        highs[sid] = _fires_at_extremes(df["high"].to_numpy(), "max")
    return {
        "high": calibrate_run(frames, highs, "high", seed=seed, n_boot=n_boot),
        "low": calibrate_run(frames, lows, "low", seed=seed, n_boot=n_boot),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: Optional[Sequence[str]] = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--calibration", type=Path, default=None,
                    help="calibration JSON ({'high': payload, 'low': payload})")
    ap.add_argument("--run-json", type=Path, default=None,
                    help="full run_v17_gpu results JSON (sides extracted)")
    ap.add_argument("--out", type=Path, default=OUT_PINE)
    ap.add_argument("--make-sample", action="store_true",
                    help="regenerate the synthetic sample calibration JSON")
    args = ap.parse_args(argv)

    if args.run_json:
        run = json.loads(Path(args.run_json).read_text(encoding="utf-8"))
        calibration = {s: d for s, d in run.get("sides", {}).items()
                       if s in ("high", "low")}
    elif args.calibration:
        calibration = json.loads(
            Path(args.calibration).read_text(encoding="utf-8"))
    else:
        if args.make_sample or not SAMPLE_JSON.exists():
            logger.info("generating synthetic sample calibration -> %s",
                        SAMPLE_JSON)
            SAMPLE_JSON.write_text(
                json.dumps(make_sample_calibration(), indent=2),
                encoding="utf-8")
        calibration = json.loads(SAMPLE_JSON.read_text(encoding="utf-8"))

    v17_text = V17_PINE.read_text(encoding="utf-8")
    built = build_pine_v17_5(v17_text, calibration)
    args.out.write_text(built, encoding="utf-8")
    n_orig = len(v17_text.splitlines())
    n_new = len(built.splitlines())
    logger.info("wrote %s (%d lines: %d detection-identical + title edit "
                "+ %d appended)", args.out, n_new, n_orig, n_new - n_orig)


if __name__ == "__main__":
    main()
