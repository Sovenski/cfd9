"""V17.5 parity audit (spec §4.2) — extends the 2026-06-10 v17 TV audit.

Gates, in order of bindingness:

1. ``signal_high``/``signal_low`` — EXACT bool equality AND count diff == 0
   (the unchanged binding gate from the v17 audit).
2. Card numeric columns (P(T2+)/P(T1) band edges lo+hi, expected move,
   expected hold) — ``atol=1e-6`` with an EXACT NaN mask match
   (engine-NaN == Pine-na on bars without an intact active card).
3. Stop level — EXACT bar-for-bar, incl. the §3.5 row-2/row-3 t+1 ordering
   and the row-6 same-side re-fire switch bar; engine-NaN == Pine-na on
   post-invalidation bars (the decided plotting semantic).
4. Survival lookups — EXACT at non-grid arguments (the R4 step-function
   floor rule; the Pine helper transcription must equal the engine rule).

Usage (after the user pastes the generated v17.5 into TradingView and
exports the chart CSV):

    python temp/parity_v17_5.py <tv_export.csv> [calibration.json]

The calibration JSON defaults to ``temp/sample_calibration_v17_5.json``;
hand the run JSON's sides for a real audit. The TV export must carry the
``card_*`` data-window plots emitted by the v17.5 builder.
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Sequence, Union

import numpy as np
import pandas as pd

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))

from src.scoring_v5 import SPAN_GRID  # noqa: E402
from src.v17_card.calibration import _har_sigma  # noqa: E402
from src.v17_card.conditioning import (  # noqa: E402
    conditional_survival, expected_hold, survival_lookup,
)
from src.v17_card.expected_move import CSideFit, expected_move  # noqa: E402
from src.v17_card.stop_rule import init_stop_state, update_stop_state  # noqa: E402

logger = logging.getLogger(__name__)

ATOL = 1e-6
SIDES = ("high", "low")
CARD_NUMERICS = ("p_t2_lo", "p_t2_hi", "p_t1_lo", "p_t1_hi", "move", "ehold")

#: R4 probe arguments — deliberately ON and OFF the grid, below the noise
#: floor and above the cap.
LOOKUP_PROBES = (5.0, 19.999, 20.0, 25.0, 47.0, 70.5, 199.0, 200.0, 450.0,
                 500.0, 700.0)


# ---------------------------------------------------------------------------
# audits (unit-tested in tests/test_pine_v17_5_builder.py)
# ---------------------------------------------------------------------------


def audit_signals(tv: np.ndarray, py: np.ndarray,
                  side: str) -> tuple[bool, str]:
    """Binding gate: exact bool equality AND count diff == 0."""
    tv_b = np.asarray(tv, dtype=bool)
    py_b = np.asarray(py, dtype=bool)
    if tv_b.shape != py_b.shape:
        return False, f"signal_{side}: shape mismatch {tv_b.shape}!={py_b.shape}"
    flips = int(np.count_nonzero(tv_b != py_b))
    cdiff = int(tv_b.sum()) - int(py_b.sum())
    ok = flips == 0 and cdiff == 0
    return ok, f"signal_{side}: flips={flips} count_diff={cdiff:+d}"


def audit_numeric(tv: np.ndarray, py: np.ndarray, name: str,
                  atol: float = ATOL) -> tuple[bool, str]:
    """Card numeric column: NaN mask EXACT, finite values within atol."""
    tv_f = np.asarray(tv, dtype=float)
    py_f = np.asarray(py, dtype=float)
    if tv_f.shape != py_f.shape:
        return False, f"{name}: shape mismatch {tv_f.shape}!={py_f.shape}"
    m_tv, m_py = np.isnan(tv_f), np.isnan(py_f)
    if not np.array_equal(m_tv, m_py):
        n_bad = int(np.count_nonzero(m_tv != m_py))
        return False, f"{name}: NaN/na mask mismatch on {n_bad} bars"
    diff = np.abs(tv_f[~m_tv] - py_f[~m_py])
    mx = float(diff.max()) if diff.size else 0.0
    return bool(mx <= atol), f"{name}: max|diff|={mx:.3g} (atol={atol:g})"


def audit_stop(tv: np.ndarray, py: np.ndarray,
               side: str) -> tuple[bool, str]:
    """Stop level EXACT bar-for-bar; engine-NaN == Pine-na (§3.5 semantic)."""
    tv_f = np.asarray(tv, dtype=float)
    py_f = np.asarray(py, dtype=float)
    if tv_f.shape != py_f.shape:
        return False, f"card_stop_{side}: shape mismatch"
    m_tv, m_py = np.isnan(tv_f), np.isnan(py_f)
    if not np.array_equal(m_tv, m_py):
        n_bad = int(np.count_nonzero(m_tv != m_py))
        return False, f"card_stop_{side}: na mask mismatch on {n_bad} bars"
    n_neq = int(np.count_nonzero(tv_f[~m_tv] != py_f[~m_py]))
    return n_neq == 0, f"card_stop_{side}: exact-mismatch bars={n_neq}"


def pine_lookup(tbl: np.ndarray, x: float,
                grid: Sequence[int] = tuple(SPAN_GRID)) -> float:
    """Literal Python transcription of the emitted Pine ``card_lookup``."""
    s = 1.0
    if x >= grid[0]:
        idx = 0
        for j in range(len(grid)):
            if grid[j] <= x:
                idx = j
        s = float(tbl[idx])
    return s


def audit_survival_lookup(calibration: dict) -> tuple[bool, str]:
    """R4: Pine helper == engine ``survival_lookup`` at every probe."""
    for side in SIDES:
        cal = calibration[side]["calibration"]
        for nm in ("S_R", "S_lo", "S_hi"):
            tbl = np.asarray(cal[nm], dtype=float)
            for x in LOOKUP_PROBES:
                if pine_lookup(tbl, x) != survival_lookup(tbl, x):
                    return False, (f"survival lookup diverges: side={side} "
                                   f"table={nm} x={x}")
    return True, "survival lookups exact (R4 floor rule, on+off grid)"


# ---------------------------------------------------------------------------
# engine card series — the §4.2 reference columns (mirrors the Pine block)
# ---------------------------------------------------------------------------


def engine_card_series(
    df: pd.DataFrame,
    signals: Union[pd.Series, np.ndarray],
    side: str,
    calib: dict,
) -> pd.DataFrame:
    """Per-bar engine card numerics for one side (NaN where Pine plots na).

    Mirrors the generated Pine state machine exactly: stop rows 1-6 via the
    T7 ``StopState`` (incl. the row-6 same-side re-fire reset), live
    conditioning per §3.2 with the F1 L-clamp and the R2 k-origin, expected
    move on the FIRE-time sigma, all values na unless an intact active card.
    """
    is_high = side == "high"
    prices = df["high" if is_high else "low"].to_numpy(dtype=float)
    closes = df["close"].to_numpy(dtype=float)
    sigma = _har_sigma(df)

    s_r = np.asarray(calib["S_R"], dtype=float)
    s_lo = np.asarray(calib["S_lo"], dtype=float)
    s_hi = np.asarray(calib["S_hi"], dtype=float)
    cs = calib["c_side"]
    c_fit = CSideFit(
        c_side=float(cs["c"]) if cs.get("c") is not None else float("nan"),
        r_squared=(float(cs["r_squared"])
                   if cs.get("r_squared") is not None else float("nan")),
        n_fit=int(cs.get("n_fit", 0)),
        use_fallback=bool(cs.get("use_fallback", True)),
        fallback_median=(float(cs["fallback_median"])
                         if cs.get("fallback_median") is not None
                         else float("nan")),
    )

    mask = (signals.fillna(False).astype(bool).to_numpy()
            if isinstance(signals, pd.Series)
            else np.asarray(signals, dtype=bool))
    fires = {int(b) for b in np.flatnonzero(mask) if b >= 1}

    n = prices.shape[0]
    cols = {f"card_{q}_{side}": np.full(n, np.nan)
            for q in (*CARD_NUMERICS, "stop")}
    state = None
    sigma_fire = float("nan")
    for u in range(n):
        if u in fires:  # row 1 / row-6 reset (fire takes precedence)
            state = init_stop_state(prices, u, is_high)
            sigma_fire = float(sigma[u])
        elif state is not None and u > state.last_bar:
            state = update_stop_state(state, prices, closes, u)
        if state is None or not state.intact:
            continue
        k = float(state.bars_survived)
        left = float(state.left_span)
        e_hold = expected_hold(s_r, k=k, left_span=left)
        cols[f"card_p_t2_lo_{side}"][u] = conditional_survival(s_lo, 50.0, k, left)
        cols[f"card_p_t2_hi_{side}"][u] = conditional_survival(s_hi, 50.0, k, left)
        cols[f"card_p_t1_lo_{side}"][u] = conditional_survival(s_lo, 200.0, k, left)
        cols[f"card_p_t1_hi_{side}"][u] = conditional_survival(s_hi, 200.0, k, left)
        cols[f"card_ehold_{side}"][u] = e_hold
        cols[f"card_move_{side}"][u] = expected_move(c_fit, sigma_fire, e_hold)
        cols[f"card_stop_{side}"][u] = state.stop
    return pd.DataFrame(cols, index=df.index)


# ---------------------------------------------------------------------------
# CLI — the user's manual §4.2 step
# ---------------------------------------------------------------------------


def _load_tv_export(path: Path) -> pd.DataFrame:
    tv = pd.read_csv(path)
    tv.columns = [str(c).strip() for c in tv.columns]
    ren = {c: c.lower() for c in tv.columns
           if c.lower() in ("time", "open", "high", "low", "close", "volume")}
    tv = tv.rename(columns=ren)
    for side, col in (("high", "Pivot High"), ("low", "Pivot Low")):
        if col in tv.columns:
            tv[f"signal_{side}"] = tv[col].fillna(0).astype(float) > 0
    return tv


def main(argv: Sequence[str]) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    if len(argv) < 1:
        print(__doc__)
        return 2
    export = Path(argv[0])
    calib_path = Path(argv[1]) if len(argv) > 1 else \
        _REPO / "temp" / "sample_calibration_v17_5.json"
    calibration = json.loads(calib_path.read_text(encoding="utf-8"))

    from src.detector import SpeculatorDetector, build_detector_artifacts
    from src.indicators import Params

    # The v17-GPU Run 1 preset — identical to the 2026-06-10 audit. Replace
    # with the winner Params when auditing a new run.
    params = Params(
        pct_extreme_high=0.554436424188316,
        min_agreement_high=0.47057701572775845,
        dur_extreme_pct_high=0.6388478276878595,
        scale_div_thresh_high=0.23202978018671275,
        pivot_drift_thresh_high=0.02970966456271708,
        pivot_drift_gate_mult_high=4.846518321894109,
        momentum_velocity_thresh_high=0.0109414285980165,
        pct_extreme_low=0.9898309362374363,
        min_agreement_low=0.10161243349676918,
        dur_extreme_pct_low=0.8481916445277483,
        scale_div_thresh_low=0.24889258429931752,
        pivot_drift_thresh_low=0.013284897982313939,
        momentum_velocity_thresh_low=0.007715178292267143,
        vola_high_pct_low=0.7099270056620911,
    )

    tv = _load_tv_export(export)
    df = tv[["open", "high", "low", "close", "volume"]].copy()
    res = SpeculatorDetector(df, params, build_detector_artifacts(df)).run()

    results: list[tuple[bool, str]] = []
    for side in SIDES:
        py_sig = res[f"signal_{side}"].to_numpy()
        if f"signal_{side}" in tv.columns:
            results.append(audit_signals(
                tv[f"signal_{side}"].to_numpy(), py_sig, side))
        else:
            results.append((False, f"signal_{side}: column missing in export"))
            continue
        cards = engine_card_series(df, py_sig, side,
                                   calibration[side]["calibration"])
        for q in CARD_NUMERICS:
            col = f"card_{q}_{side}"
            if col in tv.columns:
                results.append(audit_numeric(
                    tv[col].to_numpy(dtype=float),
                    cards[col].to_numpy(), col))
            else:
                results.append((False, f"{col}: column missing in export"))
        stop_col = f"card_stop_{side}"
        if stop_col in tv.columns:
            results.append(audit_stop(
                tv[stop_col].to_numpy(dtype=float),
                cards[stop_col].to_numpy(), side))
        else:
            results.append((False, f"{stop_col}: column missing in export"))
    results.append(audit_survival_lookup(calibration))

    fails = 0
    for ok, msg in results:
        print(f"{'PASS' if ok else 'FAIL':4s}  {msg}")
        fails += 0 if ok else 1
    print(f"\nPARITY v17.5: {'PASS' if fails == 0 else f'FAIL ({fails})'}")
    return 0 if fails == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
