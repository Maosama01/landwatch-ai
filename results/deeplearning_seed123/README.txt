LandWatch AI - FC-Siam-diff, repeat run with seed 123 (variance estimate only; NOT the shipped model)

Same code, data and settings as run 2 (results/deeplearning/), except --seed 123 and a
240-minute time budget; all 13 epochs completed. The rule was fixed before these repeats
started: they only measure run-to-run spread, the shipped model stays run 2 whatever they
score, and each repeat is evaluated on the test split exactly once.
Weights: models/siamese_unet_levir_seed123/. Mean and spread over the three full-schedule
runs (seeds 42, 7, 123): README.md and REPORT.md, Section 8.3.
