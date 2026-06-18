# Feasibility Research: World-Model + LM "Unified Layer" for Market Pivot Prediction

**Date:** 2026-06-11
**Scope:** Can a VLA-style (vision-language-action) world-model/LM combo — or its pragmatic nearest neighbor, a pretrained time-series foundation model (TSFM) — predict daily swing highs/lows (pivots) on our 16-stream daily pool, inference-only on Colab, evaluated by our existing harness?
**Constraint:** Production path is TradingView Pine parity. Any NN model is forever an *offline research / signal-validation tool or external signal feed*, never the indicator.

---

## Executive Summary

1. **The robotics analogy is real but already instantiated for markets — by TSFMs, not VLAs.** The "unified layer" in OpenVLA/GR00T/RT-2 is a shared token space where observations and actions are co-embedded so one autoregressive transformer predicts actions as just-more-tokens. The market-native equivalent exists: **Kronos** (MIT, 4M–102M params) tokenizes OHLCV K-lines into hierarchical discrete tokens and autoregressively predicts future K-lines — it is literally "observation tokenization + autoregressive action-space rollout" for markets, minus the language modality. Nobody publicly ships a market VLA with demonstrated out-of-sample trading edge.
2. **Zero-shot TSFMs on financial returns are empirically weak.** The most rigorous 2025 study (arXiv 2511.18578) finds Chronos-large zero-shot on daily excess returns gets **R² = −1.37%, directional accuracy ≈ 51%**; TimesFM-500M gets **R² = −2.80%, dir. acc. < 50%** — both lose to CatBoost/LightGBM. Domain pretraining (Kronos, pfnet's finance-tuned TimesFM) helps but no published result demonstrates an edge at the precision/firing-rate operating point our break-even thresholds demand.
3. **LLM-as-trader literature is largely a leakage artifact.** "Profit Mirage" (arXiv 2510.07920) shows almost every published LLM trading agent (FinMem, FinAgent lineage) fails to beat a *random* baseline once evaluation moves past the model's knowledge cutoff.
4. **What IS cheaply feasible:** an inference-only Colab experiment (Chronos-Bolt / Chronos-2 / Kronos, all Apache-2.0 or MIT, all ≤ 3 GB VRAM, trivially fitting L4) producing N-day quantile forecasts or sampled OHLCV paths → turn-probability signal → scored by our existing pooled harness against the dip baseline and break-even precisions (LOW > ~30% @ 1–2% firing; HIGH > ~20–25%). Cost: a few dollars of credits, ~1–6 GPU-hours, ~1–2 days of work.
5. **Verdict:** training a unified world-model layer is fantasy at our scale; running a pretrained TSFM as a *candidate signal source / benchmark null* through our harness is realistic, cheap, and worth one spike — with low expected edge but high informational value (it calibrates how much signal a 100B-observation pretrained prior extracts from our pool vs. our hand-built detector).

---

## 1. The Robotics Analogy, Taken Seriously

### 1.1 What exists publicly

| System | Org / Year | Size | Open weights | Architecture / "unified layer" |
|---|---|---|---|---|
| **OpenVLA** | Stanford et al., 2024 | 7B | Yes (MIT) | Llama-2 backbone + DINOv2/SigLIP vision; actions discretized to 256 bins/dim and mapped into the LLM token vocabulary — actions ARE tokens ([openvla.github.io](https://openvla.github.io/), [HF](https://huggingface.co/openvla/openvla-7b)) |
| **Octo** | UC Berkeley, 2024 | 27M / 93M | Yes (MIT) | Transformer trunk + diffusion action head; lightweight generalist policy ([octo-models.github.io](https://octo-models.github.io/)) |
| **NVIDIA GR00T N1/N1.5** | NVIDIA, 2025 | ~2B | Yes | Dual-system: System-2 VLM reasons, System-1 diffusion-transformer emits continuous actions; trained on synthetic + real humanoid data ([developer.nvidia.com](https://developer.nvidia.com/isaac/gr00t)) |
| **RT-1 / RT-2 / RT-X** | Google DeepMind, 2022–24 | up to 55B (RT-2) | No (datasets yes: Open X-Embodiment, 1M+ episodes, 22 embodiments) | RT-2 co-fine-tunes a VLM so action tokens share vocabulary with text — the canonical "unified layer" ([robotics-transformer-x.github.io](https://robotics-transformer-x.github.io/)) |
| **V-JEPA 2 / V-JEPA 2-AC** | Meta, June 2025 | ~1.2B (ViT-g) | Yes | Self-supervised *latent* world model on 1M+ hrs video; -AC variant post-trained on 62h robot data does zero-shot manipulation by planning in latent space — world model ≠ token unification ([ai.meta.com/vjepa](https://ai.meta.com/vjepa/), [github.com/facebookresearch/vjepa2](https://github.com/facebookresearch/vjepa2)) |
| **Genie 2/3** | Google DeepMind, 2024–25 | — | **No** | Generative interactive world models (playable video). No public weights; irrelevant for inference-only use |
| **UniVLA / unified-tokenization line** | 2025 | varies | partial | All modalities mapped to one token superset; single transformer does joint vision-language-action reasoning ([survey](https://www.emergentmind.com/topics/unified-vision-language-action-vla-tokenization), [OpenReview](https://openreview.net/forum?id=PklMD8PwUy)) |

### 1.2 What the "unified layer" actually does

Three jobs: (a) **tokenize heterogeneous observations** into a shared discrete/latent space; (b) let a **single autoregressive transformer** model the joint sequence so cross-modal structure is learned at the foundation level; (c) emit **actions as tokens** in the same vocabulary, so "acting" is just continued next-token prediction. The world-model variants (V-JEPA 2, Genie) instead learn a *predictive latent dynamics model* and either plan in it (-AC) or generate from it.

### 1.3 The market translation — and who's already done it

- **Observations** = OHLCV bars → discrete tokens. This is exactly **Kronos** (arXiv [2508.02739](https://arxiv.org/abs/2508.02739), [github.com/shiyu-coder/Kronos](https://github.com/shiyu-coder/Kronos), MIT): a hierarchical coarse+fine quantizer turns each multivariate K-line into structured tokens; a decoder-only transformer pretrained on **12B+ K-line records from 45 exchanges** autoregressively predicts future K-lines. Sampled rollouts = a generative world model of price paths.
- **Actions** = pivot calls. In VLA terms a "pivot LOW here" action token would be co-trained with the observation stream. Nobody has published this. The functional equivalent is derivable *post hoc*: sample M future paths, compute P(bar t is the minimum of window [t−w, t+k]) — i.e., the action head is a cheap deterministic function of the world model's predictive distribution. **This is why the unified layer buys little for markets**: there is no high-dimensional action space (robot joint torques) that benefits from learned compression; the "action" is one bit + a threshold.
- **Language modality**: candlestick-chart VLMs exist — VISTA (arXiv [2505.18570](https://arxiv.org/html/2505.18570v3), training-free VLM inference on stock charts) and the benchmark "Do VLMs Truly Read Candlesticks?" (arXiv [2604.12659](https://arxiv.org/html/2604.12659v1)) — and the benchmark's answer is essentially *no*: VLMs underperform at multi-scale visual price forecasting. News-text fusion is where LLMs add information, but that path is leak-ridden (§3).
- **Market world models proper**: Microsoft's **MarS** (arXiv [2409.07486](https://arxiv.org/abs/2409.07486), code public) is an order-*level* generative foundation model ("Large Market Model") for market simulation — impressive as a simulator, but it targets order-flow microstructure, needs order-book data we don't have for 155 years of daily indices, and publishes no daily-pivot trading edge. TRADES (arXiv [2502.07071](https://arxiv.org/html/2502.07071v2)) does diffusion-based LOB simulation, same caveats.

**Bottom line:** the robotics stack's transferable idea — tokenize the observation stream and pretrain an autoregressive predictor at scale — already exists for markets as Kronos. The parts that make VLA *more* than a forecaster (language grounding, high-DOF action heads, embodiment transfer) have no validated market counterpart.

---

## 2. Pragmatic Nearest Neighbors: Pretrained TSFMs for Zero-Shot Inference

All sizes fit Colab trivially: largest candidate is 710M params ≈ 2.9 GB fp32 (1.5 GB bf16) vs. L4 22 GB / A100 40 GB. VRAM is a non-issue; throughput and license are the real axes.

| Model | Params | Arch / output | Context | License | Financial evidence | Pivot-framing fit |
|---|---|---|---|---|---|---|
| **Chronos-Bolt** ([HF](https://huggingface.co/amazon/chronos-bolt-base)) | 9M / 21M / 48M / 205M | T5 enc-dec, patch input, **direct multi-step quantile forecasts** | 2048 | Apache-2.0 | Generic benchmarks only; original Chronos-large: R²=−1.37%, dir.acc.≈51% on daily excess returns ([2511.18578](https://arxiv.org/pdf/2511.18578)) | Good: 9 quantiles/horizon step → turn-probability directly; 250× faster than Chronos, runs even on CPU |
| **Chronos-2** ([HF](https://huggingface.co/amazon/chronos-2), arXiv [2510.15821](https://arxiv.org/abs/2510.15821), Oct 2025) | 120M flagship (family 9M–710M) | Encoder-only, group attention; **multivariate + covariates**, quantiles | 8192 | Apache-2.0 | None published on returns | Best modern default: can condition a stream's forecast on the other 15 pool streams as covariates |
| **TimesFM 2.5** ([HF](https://huggingface.co/google/timesfm-2.5-200m-pytorch), [GitHub](https://github.com/google-research/timesfm)) | 200M | Decoder-only, patched; continuous quantile head | **16,384** | Apache-2.0 | TimesFM-500M zero-shot: R²=−2.80%, dir.acc.<50% ([2511.18578](https://arxiv.org/pdf/2511.18578)) | Long context = whole daily history of most streams in one window; leads GIFT-Eval zero-shot ([MarkTechPost](https://www.marktechpost.com/2025/09/16/google-ai-ships-timesfm-2-5-smaller-longer-context-foundation-model-that-now-leads-gift-eval-zero-shot-forecasting/)) |
| **Kronos** ([GitHub](https://github.com/shiyu-coder/Kronos), [HF NeoQuasar](https://huggingface.co/NeoQuasar/Kronos-base)) | mini 4.1M / small 24.7M / base 102.3M | OHLCV tokenizer + decoder-only LM; **sampled K-line paths** | 2048 (mini) / 512 (small, base) | **MIT** | Pretrained on 12B K-lines, 45 exchanges; paper reports SOTA on price-series benchmarks; no audited live edge | **Best conceptual fit**: native OHLCV in/out; pivot prob = fraction of sampled paths where t is window extremum. Caveat: our streams are very likely in its training data (memorization risk, §3) |
| **Moirai / Moirai-MoE / 2.0** ([uni2ts](https://github.com/SalesforceAIResearch/uni2ts)) | 14M / 91M / 311M | Masked encoder, any-variate, mixture distributions | 1000s (patched) | **CC-BY-NC-4.0** | Generic only | Capable but **non-commercial license** — research-only dead end for anything that might feed trading |
| **Lag-Llama** ([GitHub](https://github.com/time-series-foundation-models/lag-llama)) | ~2.4M | Decoder-only w/ lag features, Student-t output | ~1024 (lags) | Apache-2.0 | Generic; outdated, beaten by TTM by ~40% ([TTM paper](https://arxiv.org/pdf/2401.03955)) | Skip — superseded |
| **Nixtla TimeGPT** ([nixtla.io](https://www.nixtla.io/)) | closed | API-only, paid | — | Proprietary API | Generic | Skip — no weights, per-call cost, can't audit lookahead; violates inference-only-on-Colab premise |
| **IBM Granite-TTM** ([HF](https://huggingface.co/ibm-granite/granite-timeseries-ttm-r2)) | 1–5M | MLP-Mixer, no attention; point + few-shot heads | 512/1024/1536 | Apache-2.0 | Generic; beats TimesFM by 19% on standard benches at 1/40 size ([NeurIPS'24](https://proceedings.neurips.cc/paper_files/paper/2024/file/874a4d89f2d04b4bcf9a2c19545cf040-Paper-Conference.pdf)) | CPU-fast sanity baseline; weaker probabilistic output |
| **TiRex** (NX-AI, [HF](https://huggingface.co/NX-AI/TiRex)) | 35M | xLSTM, quantiles; top GIFT-Eval scores | long | **NXAI community license** (restrictive) | Generic | Strong but license-awkward; use only as research comparison |
| **Toto** (Datadog, [GitHub](https://github.com/DataDog/toto)) | 151M | Multivariate, Student-T mixture; 2T training points | long | Apache-2.0 | Observability-domain focus | Optional extra |
| **pfnet TimesFM-fin** ([HF](https://huggingface.co/pfnet/timesfm-1.0-200m-fin), [blog](https://tech.preferred.jp/en/blog/timesfm/), arXiv [2412.09880](https://arxiv.org/html/2412.09880v1)) | 200M | TimesFM 1.0 continual-pretrained on 100M financial points (S&P500/TOPIX500/FX/crypto) | 512 | Apache-2.0-derived | Mock-trading improvements over base TimesFM reported by authors | The "finance fine-tune exists and is free" data point; old TimesFM 1.0 base though |

**Key negative result to internalize:** "Re(Visiting) Time Series Foundation Models in Finance" (arXiv [2511.18578](https://arxiv.org/pdf/2511.18578)) — the most systematic study to date — concludes off-the-shelf zero-shot AND fine-tuned TSFMs underperform gradient-boosted trees on daily excess returns; only *from-scratch pretraining on financial data* delivers gains. That favors Kronos/pfnet-fin over generic TSFMs for our spike, and caps expectations for all of them.

However, an important nuance for *our* task: those studies score next-day *return* forecasting (nearly pure noise). Pivot detection is a different operating point — a rare-event, multi-day-horizon, distribution-shape question ("is a local extremum forming?"), closer to what quantile/path forecasts can express. No published work tests TSFMs at exactly this framing. That is the one genuinely open question the spike answers.

---

## 3. LLM-as-Trader Evidence Base

- **Headline claims:** FinMem (arXiv [2311.13743](https://arxiv.org/abs/2311.13743), GPT-4-Turbo agent with layered memory) and FinAgent (arXiv [2402.18485](https://arxiv.org/pdf/2402.18485), multimodal, reads K-line charts) report large Sharpe improvements. Survey: arXiv [2408.06361](https://arxiv.org/html/2408.06361v2).
- **The debunking:** "Profit Mirage: Revisiting Information Leakage in LLM-based Financial Agents" (arXiv [2510.07920](https://arxiv.org/pdf/2510.07920)) re-evaluates the field: **almost every published LLM agent fails to beat a random baseline once evaluation moves past the LLM's knowledge cutoff**; GPT-4-class models answer financial QA about their backtest period at >85% accuracy — they memorized the test window. "A Test of Lookahead Bias in LLM Forecasts" (arXiv [2512.23847](https://arxiv.org/pdf/2512.23847)) formalizes detection: if accuracy rises with input familiarity, the signal is memory, not reasoning. MemGuard-Alpha (arXiv [2603.26797](https://arxiv.org/pdf/2603.26797)) builds filtering for exactly this contamination.
- **Implications for us:**
  1. Any text-LLM-in-the-loop design is presumptively contaminated; we won't pursue it.
  2. Even *numeric* TSFMs carry a subtler version: **Kronos almost certainly saw SPX/major-index daily bars in pretraining** (45 exchanges, data through ~2024-25). A backtest on pre-cutoff data measures memorization + generalization mixed. Mitigation: split evaluation at the model's data cutoff and weight the post-cutoff segment (our v17 holdout-era machinery does this naturally if the holdout era is recent), and compare pre- vs post-cutoff precision à la Profit Mirage.
  3. Generic TSFMs (Chronos, TimesFM) trained mostly on non-financial corpora have *less* memorization risk on our exact streams, which is an argument for running one of them alongside Kronos as a contamination control.

---

## 4. Concrete Minimal Experiment for THIS Project (inference-only, Colab)

**Goal:** treat a pretrained TSFM as one more candidate signal source and score it with the machinery we already have (`src/pooled_validation.py`: `build_holdout_slices` / `evaluate_holdout` / `holdout_era_pass`, `cluster_weights`; `src/pooled_scoring.py`; null baselines and cluster bootstrap as in the v17 acceptance flow). No training, no new validation code.

**Models (two arms + control):**
- Arm A — `amazon/chronos-bolt-base` (205M, Apache-2.0) or `amazon/chronos-2` (120M, Apache-2.0): quantile forecasts q∈{0.1,…,0.9} for horizons h=1…20 days.
- Arm B — `NeoQuasar/Kronos-small` or `-base` (MIT): M=30–50 sampled OHLCV paths, 20 bars ahead, context 512.
- Control — naive dip baseline + existing null baselines from the harness.

**Signal construction (LOW side; HIGH mirrored):**
- Per stream s, per day t (causal: context = bars ≤ t only, same normalization discipline as the detector):
  - Chronos arm: `p_turn(t) = P(min over h∈[1..k] of forecast median path > close_t · (1−ε))` approximated from the quantile grid — i.e., probability the market does NOT trade materially lower over the next k days, gated by a local-drawdown precondition so it fires in dip contexts. Simplest robust variant: `p_up_k = P(close_{t+k} > close_t)` read off the quantile at zero return, conditioned on close_t being an n-day low.
  - Kronos arm: `p_turn(t) = (1/M) Σ 1[ low_t ≤ min(low of path m over t+1..t+k) ]` — fraction of sampled futures in which today's low survives as the local minimum. This is the direct "world-model rollout → action probability" reduction of the VLA idea.
- Threshold τ chosen on the in-sample folds to hit the 1–2% firing-rate band; evaluate hit-vs-spam precision on holdout.

**Acceptance bar (existing project thresholds):** LOW must exceed ~30% precision at 1–2% firing rate to beat break-even; HIGH ~20–25%. Both arms also must beat the naive dip baseline and the harness nulls, with cluster-bootstrap CIs, and we report the pre/post-model-cutoff split (§3).

**Compute / cost estimate (L4, 22 GB):**
- Pool: 16 daily streams, longest ~39k bars (SPX 1871→), most far shorter; call it ~250k–400k forecast origins total.
- Chronos-Bolt-base: batched (256–512 windows/batch), >1k forecasts/s on L4 → **well under 1 GPU-hour**, <2 GB VRAM.
- Chronos-2 with 15 covariate streams: ~2–4× slower → ~1–2 h.
- Kronos-base, 30 sampled 20-step autoregressive paths per origin: the expensive arm — ~3–6 h on L4, ~1–2 h on A100. VRAM < 4 GB either way.
- Total: **< 10 Colab compute units (a few dollars)**; engineering ~1–2 days, dominated by the signal-CSV → harness adapter, not by inference.

**Deliverable:** one notebook + a signal CSV per arm (`date, stream, p_turn_low, p_turn_high`), fed through the pooled harness; a one-page result table: precision @ firing-rate vs dip baseline vs nulls, pre/post-cutoff split.

---

## 5. Honest Feasibility Verdict

**Realistic (do it):** zero-shot TSFM as a *signal source / benchmark*, inference-only, scored by our harness. Cheap, fast, fully reuses validation machinery, and answers a question no paper has: does a 100-billion-observation prior find pivot structure our hand-built detector misses? Expected outcome based on the literature: **probably fails the 30%-precision bar** (zero-shot TSFMs lose to GBDTs on daily returns; dir. acc. ~50–51%). But pivots ≠ returns, Kronos is OHLCV-native and untested at this framing, and even a negative result is a valuable calibrated null — it quantifies how much of our edge is detector-specific vs. "any strong prior would find it".

**Fantasy (don't):** training a unified world-model + LM layer for markets at our scale. Kronos needed 12B K-lines across 45 exchanges to learn a useful prior and still shows no audited trading edge; MarS needed Microsoft-scale order-flow data; the VLA "unified layer" exists to compress high-DOF action spaces and ground language — markets have a 1-bit action and no trustworthy language channel (leakage, §3). The unified-layer concept adds **nothing today** over "probabilistic forecaster + thresholding rule": the action head is a deterministic functional of the forecast distribution.

**Risks:**
- *Lookahead/memorization:* Kronos's corpus likely contains our streams; mandatory pre/post-cutoff split. Text-LLM approaches rejected outright on Profit-Mirage grounds.
- *Licensing:* stick to Apache-2.0/MIT (Chronos family, TimesFM, Kronos, TTM, Toto). Avoid Moirai (CC-BY-NC-4.0) and TiRex (NXAI community license) for anything that could ever touch a commercial signal feed. TimeGPT rejected (closed API, unauditable).
- *Project fit:* Pine parity means even a winning TSFM signal can only be (a) an offline benchmark for the detector, (b) a label-quality/feature-validation tool, or (c) an external alert feed beside the Pine indicator. It can never *be* the indicator. The spike should be framed and budgeted accordingly — as research calibration, not a production candidate.

**Recommended next step:** a half-day Colab spike with `amazon/chronos-bolt-base` (smallest-risk, fastest arm): quantile forecasts on the 16-stream pool → LOW turn-probability per §4 → pooled harness vs dip baseline and nulls. Only if it clears the nulls, escalate to the Kronos sampled-path arm (with the cutoff-split contamination check) and optionally Chronos-2 with cross-stream covariates.

---

## Sources

- VLA/robotics: [OpenVLA](https://openvla.github.io/) · [Octo](https://octo-models.github.io/) · [GR00T](https://developer.nvidia.com/isaac/gr00t) · [RT-X / Open X-Embodiment](https://robotics-transformer-x.github.io/) · [V-JEPA 2](https://ai.meta.com/vjepa/) · [Unified VLA tokenization overview](https://www.emergentmind.com/topics/unified-vision-language-action-vla-tokenization) · [UniVLA (OpenReview)](https://openreview.net/forum?id=PklMD8PwUy) · [VLA Wikipedia](https://en.wikipedia.org/wiki/Vision-language-action_model)
- Market world models / VLM-finance: [Kronos paper](https://arxiv.org/abs/2508.02739) · [Kronos GitHub](https://github.com/shiyu-coder/Kronos) · [Kronos-base HF](https://huggingface.co/NeoQuasar/Kronos-base) · [MarS](https://arxiv.org/abs/2409.07486) · [TRADES](https://arxiv.org/html/2502.07071v2) · [VISTA](https://arxiv.org/html/2505.18570v3) · [Do VLMs Read Candlesticks?](https://arxiv.org/html/2604.12659v1) · [Kinlay on Kronos](https://jonathankinlay.com/2026/02/time-series-foundation-models-for-financial-markets-kronos-and-the-rise-of-pre-trained-market-models/)
- TSFMs: [chronos-bolt-base](https://huggingface.co/amazon/chronos-bolt-base) · [Chronos-Bolt AWS blog](https://aws.amazon.com/blogs/machine-learning/fast-and-accurate-zero-shot-forecasting-with-chronos-bolt-and-autogluon/) · [Chronos-2 HF](https://huggingface.co/amazon/chronos-2) · [Chronos-2 paper](https://arxiv.org/abs/2510.15821) · [Chronos-2 Amazon Science](https://www.amazon.science/blog/introducing-chronos-2-from-univariate-to-universal-forecasting) · [TimesFM GitHub](https://github.com/google-research/timesfm) · [timesfm-2.5-200m HF](https://huggingface.co/google/timesfm-2.5-200m-pytorch) · [TimesFM 2.5 (MarkTechPost)](https://www.marktechpost.com/2025/09/16/google-ai-ships-timesfm-2-5-smaller-longer-context-foundation-model-that-now-leads-gift-eval-zero-shot-forecasting/) · [uni2ts/Moirai](https://github.com/SalesforceAIResearch/uni2ts) · [Moirai 2.0 blog](https://www.salesforce.com/blog/moirai-2-0/) · [Lag-Llama](https://github.com/time-series-foundation-models/lag-llama) · [Granite-TTM r2](https://huggingface.co/ibm-granite/granite-timeseries-ttm-r2) · [TTM NeurIPS](https://proceedings.neurips.cc/paper_files/paper/2024/file/874a4d89f2d04b4bcf9a2c19545cf040-Paper-Conference.pdf) · [TiRex](https://huggingface.co/NX-AI/TiRex) · [Toto](https://github.com/DataDog/toto) · [pfnet timesfm-fin](https://huggingface.co/pfnet/timesfm-1.0-200m-fin) · [PFN blog](https://tech.preferred.jp/en/blog/timesfm/) · [Financial fine-tuning TimesFM](https://arxiv.org/html/2412.09880v1)
- Finance evals & leakage: [Re(Visiting) TSFMs in Finance](https://arxiv.org/pdf/2511.18578) · [TSFMs in Finance survey (ACM)](https://dl.acm.org/doi/full/10.1145/3785706.3785728) · [FinMem](https://arxiv.org/abs/2311.13743) · [FinAgent](https://arxiv.org/pdf/2402.18485) · [LLM trading agents survey](https://arxiv.org/html/2408.06361v2) · [Profit Mirage](https://arxiv.org/pdf/2510.07920) · [Lookahead-bias test](https://arxiv.org/pdf/2512.23847) · [MemGuard-Alpha](https://arxiv.org/pdf/2603.26797) · [LLM market simulations (Lopez-Lira)](https://arxiv.org/abs/2504.10789)
