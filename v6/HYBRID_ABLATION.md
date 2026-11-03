# Hybrid ablation — executed evidence

## Solver budget selection, validation only

80 validation poses, same8ms deadline and personal neural initialization:

| Maximum evaluations | Tip proxy mm | Pinch proxy mm | Solver p50 / p95 ms | Timeouts |
|---|---:|---:|---:|---:|
| 4 | 23.365 | 10.439 | 3.917 / 6.469 | 0/80 |
| 6 | 22.709 | 10.045 | 6.034 / 7.729 | 4/80 |
| 8 | 24.608 | 11.348 | 7.494 / 8.550 | 25/80 |

Late outputs are discarded, so more iterations can worsen effective accuracy. Four evaluations were selected for deadline reliability before test evaluation. Later runs can have different timeout counts due to scheduling/load. These are short validation samples, not statistical guarantees. Initial full validation with8 evaluations is preserved separately and not presented as the final configuration.

## Does the learned model contribute?

Another80-pose validation pass,4 evaluations/8ms:

| Initialization | Tip proxy mm | Pinch proxy mm | Solver p95 ms | Timeouts |
|---|---:|---:|---:|---:|
| Calibrated ROM | 29.975 | 13.555 | 6.952 | 0/80 |
| Personal MLP | 23.364 | 10.459 | 6.352 | 1/80 |

The MLP genuinely supplies a more useful starting configuration in this bounded solve and ordinary output when precision is not requested. This is not a new-person personalization ablation, and it does not separate personal training from every possible generic pretrained model. No generic network was trained or compared.

## Adaptive selection versus always solving, test recordings

Same866 accepted observations out of921 original valid inputs:

| Method | Raw tip mm | Raw pinch mm | Mean mapping ms | Pooled p50 / p95 ms |
|---|---:|---:|---:|---:|
| Personal neural | 31.111 | 15.621 | 0.460 | 0.453 / 0.625 |
| Always bounded optimizer | 22.064 | 7.477 | 3.474 | 3.476 / 4.888 |
| Adaptive | 27.278 | 11.742 | 1.806 | 0.739 / 3.941 |

Mapping timing includes quality handling and geometry as used, from precomputed79 features. CPU costs for the full world-landmarks-to-safe-command path are separately measured in `full_controller_profile.json`; its mean values are1.614ms for preserved neural implementation,5.614ms for bounded optimizer with v6 safety and4.120ms for hybrid. Different boundaries must not be interchanged.

Hybrid uses368 precision and498 neural outputs in clean evaluation, with zero deadline fallbacks in368 attempts. Always solving has one deadline fallback in866 attempts. Hybrid is not the most accurate solution; it occupies a speed/accuracy tradeoff. Raw geometry coverage is shared and disclosed. Its55 rejected finite detections still contribute to physical error/fallback metrics.

## Filtering versus dynamic accuracy

Validation neural-only filter ablation:

| Filter | Physical tip mm | Mean command acceleration vector norm rad/s² |
|---|---:|---:|
| v5 filter | 63.525 | 66.098 |
| v6 tau.045s | 65.510 | 53.755 |
| v6 tau.025s | 65.280 | 60.686 |
| v6 tau.015s | 65.283 | 65.593 |

The extra acceleration constraint reduces abrupt commands but adds lag; faster gains did not recover the old filter's physical error and sacrificed smoothness, so.045s was retained. Do not describe this as universal responsiveness improvement.

In final sample-weighted test replay, physical tip error old-filter neural49.932mm versus hybrid50.096mm. Maximum discrete command acceleration147.059 versus30.000rad/s²; mean acceleration vector norm78.809 versus63.190. Joint-target RMSE0.03896 versus0.034rad approximately (exact values in CSV). Thus the precision/command-smoothness gain does not translate into a lower overall physical human-target error than the old filter.

All22 injected loss windows reconverged to within0.1rad maximum of each controller's clean command for2 cached observations: median0.578s old-filter neural,0.681s neural-v6-safety,0.680s adaptive. The smoother recovery is slower. These are sample-resolution command-convergence measurements, not real-world motion latency.

## Prediction was screened and rejected

461 validation transitions, mean next-observed-frame horizon96.4ms. Canonical landmark prediction error: hold-last0.54750 palm widths; bounded constant velocity0.61927; lowpass tau.06s0.51984. Constant-velocity overshoot proxy0.01180 palm widths. The lowpass is applied to landmarks using the old time constant, not a claim about identical robot-filter execution. No runtime predictor, temporal network or hybrid-prediction mode was added.

No numerical command averaging Mode D was added; existing motion safety already provides smooth mode transitions, and no measured benefit justified another blend policy. One Euro filtering was not implemented; the retained tested filter/acceleration limiter was the bounded alternative.
