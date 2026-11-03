# Merge decision

**Recommend a limited optional merge of the v6 precision engine and evaluation tools after review. Do not replace the v5 default or publish this branch.** Keep the frozen v5 RC1 as the ready-to-submit artifact. No actual merge into the baseline branch has been performed.

Why it passes an optional engineering gate:

- Original25 plus6 new tests pass. Actual recorded personal motion drives freshly stepped neural and adaptive simulations. No numerical warnings, joint-limit excursions or control-bound violations were measured in the evaluation.
- On the identical866 accepted observations, hybrid reduces raw tip proxy error12.32% and pinch error24.83% relative to neural mapping, with less mean mapping computation than always solving. The personal MLP is both ordinary output and solver warm start; the initialization ablation supports its contribution.
- Command acceleration is bounded to30rad/s² rather than the old filter's measured147rad/s² peak. This is a command-level property; hard physical servo derivative guarantees are not claimed.
- The new analytic bounded optimizer is a useful optional maximum-precision preset. It remains more accurate than the adaptive strategy.

Why it must not become the default/submission replacement yet:

- Mean physical fingertip error is slightly worse than old-filter neural (50.096 versus49.932mm); recovery is about0.10s slower by the sample-based proxy.
- The camera-worker path increased unattended display cadence25.16→29.59fps but software read-completion-to-render delay worsened51.07→60.12ms. Both checks detected zero hands; the user deferred attended testing. No live precision or true motion-latency improvement is established.
- The extra quality gate rejects55/921 finite detections; these are proxies and may reject valid difficult poses. Full physical metrics retain them. Existing test recordings are reused and categories are geometric, not new-person/general-purpose evidence.
- Prediction was rejected; optional G1/Shadow adaptation was not attempted. Fixed-base G1, scripted left hand and0/5 grasp results remain unchanged.

Practical decision: retain `neural_live.py` / `live_mirror.py` as verified defaults. Offer `python -m v6.live --controller hybrid` and `--controller optimizer` as explicitly experimental options. Promote the camera worker only after an attended, matched-condition trial demonstrates a useful tradeoff. Do not silently merge an app-wide replacement or claim every metric improved.

Frozen candidate, private artifacts, undecided original-code license and owner publication approval remain unchanged. The new video and any derived data need their own release review; existing13-file approvals, still pending, do not cover new content.


Final integration update (November 3): attended neural and hybrid both tracked; operator reported no obvious improvement. Local optional selector defaults to unchanged v5. 35 tests and portable held-out checks pass; final report and approval list are in the new candidate. No default switch, Git merge, push or publication is authorized by this result.
