LandWatch AI - FC-Siam-diff, training run 2 (the default model in models/siamese_unet_levir)

Trained 2026-11-19/20 for all 13 planned epochs (83.4 min on Apple MPS), with the same code,
data, seed (42) and hyper-parameters as run 1 (results/deeplearning_run1_7ep/). Run 2 was
written to models/siamese_unet_levir_13ep and results/deeplearning_13ep, then promoted to the
default locations because its VALIDATION F1 (0.8738) beat run 1's (0.8554). Paths printed
inside these logs refer to those original folder names.

  training_log.csv / training_console.log   per-epoch loss and val metrics
  val_threshold_sweep.csv                   val F1 for thresholds 0.05..0.95 (picked 0.65)
  val_metrics.txt / val_eval_console.log    independent re-evaluation of the val split
  test_metrics.txt / test_eval_console.log  the single held-out test evaluation
  examples/                                 full 1024x1024 test_7 prediction and its F1
  device_benchmark.txt                      CPU vs MPS training-step benchmark (random
                                            weights, independent of either run)
