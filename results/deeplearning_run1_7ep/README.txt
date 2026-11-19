LandWatch AI - FC-Siam-diff, training run 1 (kept for comparison; superseded)

Run 1 was planned for 13 epochs but stopped after 7 by its 75-minute time budget
(see training.stop_reason in models/siamese_unet_levir_run1_7ep/model.properties).
Best val F1 0.8554 (epoch 7, threshold 0.80); its single test evaluation gave F1 0.8548.

On 2026-11-20 run 2 (same code, data, seed and hyper-parameters, all 13 epochs) reached
val F1 0.8738. Following the rule fixed before run 2 started (keep the model with the higher
VALIDATION F1; the test split plays no part in the choice), run 2 became the default model.
Run 1's files were then moved here unchanged:
  results/deeplearning/...     -> results/deeplearning_run1_7ep/...
  models/siamese_unet_levir/   -> models/siamese_unet_levir_run1_7ep/
Paths printed inside these logs refer to the original locations.
