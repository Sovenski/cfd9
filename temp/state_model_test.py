"""Pre-registered test of the 2x2 STATE MODEL of pivots.

States per bar (all trailing/causal):
  price axis UP      iff close[t] > close[t-20]
  vol axis RISING    iff rv20[t] > rv20[t-20],  rv20 = 20-bar rolling std of log-returns
  S1 = up/falling-vol   S2 = up/rising-vol   S3 = down/rising-vol   S4 = down/falling-vol

Pre-registered measurements (all reported, no tuning):
  1. State map: occupancy, dwell mean/median, dwell CCDF (hazard shape).
  2. Pivot density map: P(real TOP/BOTTOM within +-2 | state), pooled + base rates;
     transition-bar probabilities (S1->S2/S3 for tops, S3->S1/S4 for bottoms).
  3. KEY TEST: trailing-50-bar extreme trigger, unconditional vs state-conditional
     vs rate-matched (trailing-100) control; cluster bootstrap by stream (2000).
     PASS: conditional >= 1.5x unconditional AND beats control with p < 0.05.
  4. Forward 20/40-bar trade-direction log-returns (report only).
  5. GBM null (mu=0, per-stream vol-matched) through the identical pipeline.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import logging

logging.disable(logging.INFO)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.pooled_validation import load_stream_frame
from src.universe import resolve_streams

SWING_W, DROP_ATR, TOL = 20, 2.0, 2
MIN_BAR = 300
LOOKBACK = 20          # state model lookback (both axes)
TRIG_W, CTRL_W = 50, 100
N_BOOT = 2000
RNG = np.random.default_rng(42)


# ---------------------------------------------------------------- ground truth
def wilder_atr(df: pd.DataFrame, n: int = 14) -> np.ndarray:
    h, l, c = (df[k].to_numpy(float) for k in ("high", "low", "close"))
    tr = np.maximum(h[1:] - l[1:], np.maximum(np.abs(h[1:] - c[:-1]), np.abs(l[1:] - c[:-1])))
    a = pd.Series(np.concatenate([[h[0] - l[0]], tr])).ewm(alpha=1 / n, adjust=False).mean().to_numpy()
    a[a == 0] = np.nan
    return a


def real_turns(df: pd.DataFrame, kind: str, atr: np.ndarray) -> np.ndarray:
    h, l = df["high"].to_numpy(float), df["low"].to_numpy(float)
    n = len(df)
    out = np.zeros(n, bool)
    for t in range(SWING_W, n - SWING_W):
        if kind == "top" and h[t] == h[t - SWING_W:t + SWING_W + 1].max() and \
           (h[t] - l[t + 1:t + SWING_W + 1].min()) >= DROP_ATR * atr[t]:
            out[t] = True
        elif kind == "bot" and l[t] == l[t - SWING_W:t + SWING_W + 1].min() and \
             (h[t + 1:t + SWING_W + 1].max() - l[t]) >= DROP_ATR * atr[t]:
            out[t] = True
    return out


def dilate(x: np.ndarray, tol: int = TOL) -> np.ndarray:
    """near[t] = any pivot in [t-tol, t+tol]."""
    out = x.copy()
    for k in range(1, tol + 1):
        out[k:] |= x[:-k]
        out[:-k] |= x[k:]
    return out


# ---------------------------------------------------------------- state model
def compute_states(close: np.ndarray) -> np.ndarray:
    """Return state array in {0=undefined,1..4}, fully causal."""
    n = len(close)
    logret = np.full(n, np.nan)
    logret[1:] = np.diff(np.log(close))
    rv20 = pd.Series(logret).rolling(LOOKBACK).std().to_numpy()
    st = np.zeros(n, int)
    for t in range(2 * LOOKBACK + 1, n):
        if np.isnan(rv20[t]) or np.isnan(rv20[t - LOOKBACK]):
            continue
        up = close[t] > close[t - LOOKBACK]
        rising = rv20[t] > rv20[t - LOOKBACK]
        st[t] = (1 if not rising else 2) if up else (3 if rising else 4)
    return st


def trailing_extreme(close: np.ndarray, w: int, kind: str) -> np.ndarray:
    s = pd.Series(close)
    if kind == "max":
        return (close >= s.rolling(w).max().to_numpy()) & ~np.isnan(s.rolling(w).max().to_numpy())
    return (close <= s.rolling(w).min().to_numpy()) & ~np.isnan(s.rolling(w).min().to_numpy())


def dwells(st: np.ndarray, mask: np.ndarray) -> dict[int, list[int]]:
    out: dict[int, list[int]] = {1: [], 2: [], 3: [], 4: []}
    idx = np.where(mask)[0]
    if len(idx) == 0:
        return out
    run_s, run_len = st[idx[0]], 1
    for j in range(1, len(idx)):
        t = idx[j]
        contiguous = idx[j] == idx[j - 1] + 1
        if contiguous and st[t] == run_s:
            run_len += 1
        else:
            if run_s in out:
                out[run_s].append(run_len)
            run_s, run_len = st[t], 1
    if run_s in out:
        out[run_s].append(run_len)
    return out


# ---------------------------------------------------------------- GBM null
def make_gbm(df: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Vol-matched GBM (mu=0) with synthetic high/low matching real wick scale."""
    rng = np.random.default_rng(seed)
    c = df["close"].to_numpy(float)
    h, l = df["high"].to_numpy(float), df["low"].to_numpy(float)
    o = df["open"].to_numpy(float) if "open" in df.columns else np.concatenate([[c[0]], c[:-1]])
    lr = np.diff(np.log(c))
    sigma = np.nanstd(lr)
    s_h = np.nanstd(np.log(np.maximum(h, 1e-12) / np.maximum(np.maximum(o, c), 1e-12)))
    s_l = np.nanstd(np.log(np.maximum(np.minimum(o, c), 1e-12) / np.maximum(l, 1e-12)))
    n = len(c)
    z = rng.standard_normal(n - 1)
    gc = np.empty(n)
    gc[0] = c[0] if np.isfinite(c[0]) and c[0] > 0 else 100.0
    gc[1:] = gc[0] * np.exp(np.cumsum(sigma * z))
    go = np.concatenate([[gc[0]], gc[:-1]])
    uh = np.abs(rng.standard_normal(n)) * s_h
    ul = np.abs(rng.standard_normal(n)) * s_l
    gh = np.maximum(go, gc) * np.exp(uh)
    gl = np.minimum(go, gc) * np.exp(-ul)
    return pd.DataFrame({"open": go, "high": gh, "low": gl, "close": gc})


# ---------------------------------------------------------------- per-stream
def analyze(df: pd.DataFrame, sid: str) -> dict:
    n = len(df)
    atr = wilder_atr(df)
    close = df["close"].to_numpy(float)
    tops = real_turns(df, "top", atr)
    bots = real_turns(df, "bot", atr)
    near_top, near_bot = dilate(tops), dilate(bots)
    st = compute_states(close)
    mask = np.zeros(n, bool)
    mask[MIN_BAR:n - SWING_W] = True
    mask &= st > 0

    res: dict = {"sid": sid, "n_valid": int(mask.sum()),
                 "n_tops": int((tops & mask).sum()), "n_bots": int((bots & mask).sum())}

    # 1. occupancy + dwells
    res["occ"] = {s: int(((st == s) & mask).sum()) for s in (1, 2, 3, 4)}
    res["dwells"] = dwells(st, mask)

    # 2. pivot density per state + transitions
    res["dens"] = {}
    for s in (1, 2, 3, 4):
        m = (st == s) & mask
        res["dens"][s] = (int((near_top & m).sum()), int((near_bot & m).sum()), int(m.sum()))
    res["base"] = (int((near_top & mask).sum()), int((near_bot & mask).sum()), int(mask.sum()))
    prev = np.concatenate([[0], st[:-1]])
    tr_top = mask & (((prev == 1) & (st == 2)) | ((prev == 1) & (st == 3)))
    tr_bot = mask & (((prev == 3) & (st == 1)) | ((prev == 3) & (st == 4)))
    res["trans"] = {"top": (int((near_top & tr_top).sum()), int(tr_top.sum())),
                    "bot": (int((near_bot & tr_bot).sum()), int(tr_bot.sum()))}

    # 3. triggers
    trig_hi = trailing_extreme(close, TRIG_W, "max") & mask
    trig_lo = trailing_extreme(close, TRIG_W, "min") & mask
    ctrl_hi = trailing_extreme(close, CTRL_W, "max") & mask
    ctrl_lo = trailing_extreme(close, CTRL_W, "min") & mask

    def fh(fire: np.ndarray, near: np.ndarray) -> tuple[int, int]:
        return int((fire & near).sum()), int(fire.sum())

    res["trig"] = {
        "top_uncond": fh(trig_hi, near_top),
        "top_S1": fh(trig_hi & (st == 1), near_top),
        "top_S2": fh(trig_hi & (st == 2), near_top),
        "top_S12": fh(trig_hi & ((st == 1) | (st == 2)), near_top),
        "top_ctrl": fh(ctrl_hi, near_top),
        "bot_uncond": fh(trig_lo, near_bot),
        "bot_S3": fh(trig_lo & (st == 3), near_bot),
        "bot_S4": fh(trig_lo & (st == 4), near_bot),
        "bot_S34": fh(trig_lo & ((st == 3) | (st == 4)), near_bot),
        "bot_ctrl": fh(ctrl_lo, near_bot),
    }

    # 4. forward payoff (sum, count) of trade-direction log-returns
    logc = np.log(close)
    res["fwd"] = {}
    for hzn in (20, 40):
        f = np.full(n, np.nan)
        f[:n - hzn] = logc[hzn:] - logc[:n - hzn]
        ok = mask & np.isfinite(f)
        def ssum(fire: np.ndarray, sign: float) -> tuple[float, int]:
            m = fire & ok
            return float(sign * f[m].sum()), int(m.sum())
        res["fwd"][hzn] = {
            "top_S1": ssum(trig_hi & (st == 1), -1.0),
            "top_S2": ssum(trig_hi & (st == 2), -1.0),
            "bot_S3": ssum(trig_lo & (st == 3), +1.0),
            "bot_S4": ssum(trig_lo & (st == 4), +1.0),
            "base_short": ssum(ok, -1.0),
            "base_long": ssum(ok, +1.0),
        }
    return res


# ---------------------------------------------------------------- pooling
def pool_prec(rows: list[dict], key: str) -> tuple[float, int, int]:
    h = sum(r["trig"][key][0] for r in rows)
    f = sum(r["trig"][key][1] for r in rows)
    return (h / f if f else np.nan), h, f


def boot_diffs(rows: list[dict], k_cond: str, k_ref: str) -> tuple[float, float]:
    """Cluster bootstrap by stream: returns (p_one_sided diff<=0, mean diff)."""
    H = np.array([[r["trig"][k][0] for k in (k_cond, k_ref)] for r in rows], float)
    F = np.array([[r["trig"][k][1] for k in (k_cond, k_ref)] for r in rows], float)
    ns = len(rows)
    idx = RNG.integers(0, ns, size=(N_BOOT, ns))
    hs, fs = H[idx].sum(axis=1), F[idx].sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        prec = hs / fs
    d = prec[:, 0] - prec[:, 1]
    d = d[np.isfinite(d)]
    return float((d <= 0).mean()), float(np.nanmean(d))


def density_table(rows: list[dict]) -> pd.DataFrame:
    rec = []
    for s in (1, 2, 3, 4):
        ht = sum(r["dens"][s][0] for r in rows)
        hb = sum(r["dens"][s][1] for r in rows)
        m = sum(r["dens"][s][2] for r in rows)
        rec.append(dict(state=f"S{s}", bars=m,
                        p_top=ht / m if m else np.nan, p_bot=hb / m if m else np.nan))
    bt = sum(r["base"][0] for r in rows)
    bb = sum(r["base"][1] for r in rows)
    bm = sum(r["base"][2] for r in rows)
    rec.append(dict(state="BASE", bars=bm, p_top=bt / bm, p_bot=bb / bm))
    return pd.DataFrame(rec).set_index("state")


def run_pipeline(frames: list[tuple[str, pd.DataFrame]]) -> list[dict]:
    return [analyze(df, sid) for sid, df in frames]


# ---------------------------------------------------------------- main
def main() -> None:
    streams = resolve_streams(["INDICES", "COMMODITIES", "FX", "WORLD_ETF"], ["1D"], "data/raw_v16")
    frames = [(st.stream_id, load_stream_frame(st.path)) for st in streams]
    print(f"streams: {len(frames)}  total bars: {sum(len(f) for _, f in frames)}")

    rows = run_pipeline(frames)
    gbm_frames = [(sid, make_gbm(df, seed=1000 + i)) for i, (sid, df) in enumerate(frames)]
    rows_g = run_pipeline(gbm_frames)

    # ---- 1. state map
    print("\n================ 1. STATE MAP (real) ================")
    occ_tot = {s: sum(r["occ"][s] for r in rows) for s in (1, 2, 3, 4)}
    nv = sum(occ_tot.values())
    dw_all = {s: np.concatenate([np.array(r["dwells"][s], float) for r in rows if r["dwells"][s]])
              for s in (1, 2, 3, 4)}
    names = {1: "S1 up/fall-vol", 2: "S2 up/rise-vol", 3: "S3 dn/rise-vol", 4: "S4 dn/fall-vol"}
    for s in (1, 2, 3, 4):
        d = dw_all[s]
        print(f"  {names[s]:<16s} occ {occ_tot[s]/nv:6.1%}  dwell mean {d.mean():5.2f} "
              f"median {np.median(d):4.0f}  max {d.max():4.0f}  n_runs {len(d)}")
    # hazard shape: R^2 of log-CCDF vs log d (power-law) vs vs d (exponential)
    print("  hazard shape (CCDF fit R^2, tail d<=p99):")
    for s in (1, 2, 3, 4):
        d = np.sort(dw_all[s])
        u, cnt = np.unique(d, return_counts=True)
        ccdf = 1.0 - np.cumsum(cnt) / cnt.sum()
        keep = (ccdf > 0) & (u <= np.quantile(d, 0.99))
        x, y = u[keep], np.log(ccdf[keep])
        r2_pl = np.corrcoef(np.log(x), y)[0, 1] ** 2
        r2_ex = np.corrcoef(x, y)[0, 1] ** 2
        print(f"    S{s}: R2(power-law)={r2_pl:.3f}  R2(exponential)={r2_ex:.3f}"
              f"  -> {'power-law-ish (decreasing hazard)' if r2_pl > r2_ex else 'exponential-ish (flat hazard)'}")

    # ---- 2. pivot density map
    print("\n================ 2. PIVOT DENSITY MAP ================")
    dt_real = density_table(rows)
    dt_gbm = density_table(rows_g)
    print("REAL:  P(pivot within +-2 | state)")
    print(dt_real.to_string(float_format=lambda v: f"{v:.4f}"))
    base_t, base_b = dt_real.loc["BASE", "p_top"], dt_real.loc["BASE", "p_bot"]
    print("lift vs base:")
    for s in (1, 2, 3, 4):
        print(f"  S{s}: top x{dt_real.loc[f'S{s}','p_top']/base_t:5.2f}   "
              f"bot x{dt_real.loc[f'S{s}','p_bot']/base_b:5.2f}")
    tt_h = sum(r["trans"]["top"][0] for r in rows); tt_n = sum(r["trans"]["top"][1] for r in rows)
    tb_h = sum(r["trans"]["bot"][0] for r in rows); tb_n = sum(r["trans"]["bot"][1] for r in rows)
    print(f"transition bars: P(top | S1->S2/S3 +-2) = {tt_h/tt_n:.4f} (n={tt_n}, lift x{tt_h/tt_n/base_t:.2f})")
    print(f"                 P(bot | S3->S1/S4 +-2) = {tb_h/tb_n:.4f} (n={tb_n}, lift x{tb_h/tb_n/base_b:.2f})")
    print("\nGBM NULL (mu=0, vol-matched):")
    print(dt_gbm.to_string(float_format=lambda v: f"{v:.4f}"))

    # ---- 3. key test
    print("\n================ 3. KEY TEST: extremeness-in-the-right-state ================")
    variants = {"top": ["top_uncond", "top_S1", "top_S2", "top_S12", "top_ctrl"],
                "bot": ["bot_uncond", "bot_S3", "bot_S4", "bot_S34", "bot_ctrl"]}
    prec: dict[str, tuple[float, int, int]] = {}
    for side, ks in variants.items():
        print(f"--- {side.upper()}S (trigger: close = trailing-{TRIG_W} {'max' if side=='top' else 'min'};"
              f" control: trailing-{CTRL_W}) ---")
        for k in ks:
            p, h, f = pool_prec(rows, k)
            prec[k] = (p, h, f)
            print(f"  {k:<11s} precision {p:7.4f}  hits {h:5d} / fires {f:6d}")
    verdicts = []
    for side, conds in (("top", ["top_S1", "top_S2", "top_S12"]),
                        ("bot", ["bot_S3", "bot_S4", "bot_S34"])):
        unc, ctrl = f"{side}_uncond", f"{side}_ctrl"
        print(f"--- {side.upper()} bootstrap (cluster by stream, {N_BOOT}) ---")
        side_pass = False
        for k in conds:
            p_a, d_a = boot_diffs(rows, k, unc)
            p_c, d_c = boot_diffs(rows, k, ctrl)
            ratio = prec[k][0] / prec[unc][0] if prec[unc][0] else np.nan
            ok = (ratio >= 1.5) and (p_c < 0.05)
            side_pass |= ok
            print(f"  {k:<8s} vs uncond: diff {d_a:+.4f} p={p_a:.4f} | vs ctrl: diff {d_c:+.4f} "
                  f"p={p_c:.4f} | ratio x{ratio:.2f} | registered-pass: {ok}")
        verdicts.append((side, side_pass))
        print(f"  >>> {side.upper()} side registered verdict: {'PASS' if side_pass else 'FAIL'}")
    print(f"\n>>> OVERALL registered verdict: "
          f"{'PASS' if any(v for _, v in verdicts) else 'FAIL'} "
          f"(tops={'PASS' if verdicts[0][1] else 'FAIL'}, bottoms={'PASS' if verdicts[1][1] else 'FAIL'})")

    # ---- 4. forward payoff
    print("\n================ 4. FORWARD PAYOFF (trade direction, report-only) ================")
    for hzn in (20, 40):
        print(f"  horizon {hzn} bars (mean log-return in trade direction):")
        for k in ("top_S1", "top_S2", "bot_S3", "bot_S4", "base_short", "base_long"):
            s = sum(r["fwd"][hzn][k][0] for r in rows)
            c = sum(r["fwd"][hzn][k][1] for r in rows)
            print(f"    {k:<11s} {s/c if c else np.nan:+.5f}  (n={c})")

    # ---- figure
    fig = plt.figure(figsize=(15, 11))
    gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.28)

    ax = fig.add_subplot(gs[0, 0])
    mat = np.array([[dt_real.loc[f"S{s}", "p_top"], dt_real.loc[f"S{s}", "p_bot"],
                     dt_gbm.loc[f"S{s}", "p_top"], dt_gbm.loc[f"S{s}", "p_bot"]] for s in (1, 2, 3, 4)])
    im = ax.imshow(mat, cmap="viridis", aspect="auto")
    ax.set_xticks(range(4), ["top REAL", "bot REAL", "top GBM", "bot GBM"])
    ax.set_yticks(range(4), [names[s] for s in (1, 2, 3, 4)])
    for i in range(4):
        for j in range(4):
            ax.text(j, i, f"{mat[i, j]:.3f}", ha="center", va="center",
                    color="white" if mat[i, j] < mat.max() * 0.6 else "black", fontsize=9)
    ax.set_title(f"Pivot density P(pivot +-{TOL} | state) — real vs GBM null\n"
                 f"base: top {base_t:.3f} / bot {base_b:.3f} (real)")
    fig.colorbar(im, ax=ax, shrink=0.8)

    ax = fig.add_subplot(gs[0, 1])
    for s, col in zip((1, 2, 3, 4), ("tab:green", "tab:olive", "tab:red", "tab:blue")):
        d = np.sort(dw_all[s])
        u, cnt = np.unique(d, return_counts=True)
        ccdf = 1.0 - np.cumsum(cnt) / cnt.sum()
        keep = ccdf > 0
        ax.loglog(u[keep], ccdf[keep], label=names[s], color=col)
    ax.set_xlabel("dwell length d (bars)")
    ax.set_ylabel("P(D > d)")
    ax.set_title("Dwell-duration CCDF per state (log-log)\nstraight line = power-law = decreasing hazard")
    ax.legend(fontsize=8)
    ax.grid(True, which="both", alpha=0.3)

    for col_i, (side, ks, labels) in enumerate((
            ("top", ["top_uncond", "top_S1", "top_S2", "top_ctrl"],
             ["uncond (50)", "cond S1", "cond S2", "ctrl (100)"]),
            ("bot", ["bot_uncond", "bot_S3", "bot_S4", "bot_ctrl"],
             ["uncond (50)", "cond S3", "cond S4", "ctrl (100)"]))):
        ax = fig.add_subplot(gs[1, col_i])
        vals = [prec[k][0] for k in ks]
        fires = [prec[k][2] for k in ks]
        bars = ax.bar(labels, vals, color=["gray", "tab:green", "tab:olive", "black"] if side == "top"
                      else ["gray", "tab:red", "tab:blue", "black"])
        for b, v, f in zip(bars, vals, fires):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.002, f"{v:.3f}\nn={f}",
                    ha="center", fontsize=8)
        ax.axhline(base_t if side == "top" else base_b, ls="--", color="red", lw=1,
                   label="all-bars base rate")
        ax.set_ylabel(f"precision: real {'top' if side=='top' else 'bottom'} within +-{TOL}")
        ax.set_title(f"Test 3 — {'TOPS' if side=='top' else 'BOTTOMS'}: extreme trigger precision")
        ax.legend(fontsize=8)
    fig.suptitle("State model of pivots — pre-registered test (16 streams, 1D, bars>=300)", fontsize=13)
    out = Path(__file__).with_suffix(".png")
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print(f"\nfigure -> {out}")


if __name__ == "__main__":
    main()
