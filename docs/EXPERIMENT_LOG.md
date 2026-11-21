# Experiment log

Every training or evaluation run whose numbers appear in the documentation, in the order they were made. All metric values are copied from the files named; the full commands are in [REPORT.md, Appendix A](../REPORT.md#appendix-a-reproducing-every-result). The rules for every run: train on TRAIN, make every choice on VAL, and score each model on TEST at most once.

## Change detection on LEVIR-CD

| ID | Run | What changes | Split use | Main result | Files |
|---|---|---|---|---|---|
| CD-0 | Classical baseline (luminance difference, Otsu, `minRegionSize` 12) | no training | VAL; TEST once | test F1 0.1008, IoU 0.0531 | `results/benchmarks/baseline_{val,test}_metrics.txt` |
| CD-0s | Baseline `minRegionSize` sweep | post-processing size | VAL only | no setting helps meaningfully | `results/benchmarks/baseline_val_minregion_sweep.txt` |
| CD-1 | FC-Siam-diff run 1, seed 42 | stopped by time budget at epoch 7 of 13 | VAL; TEST once | val F1 0.8554, test F1 0.8548 | `models/siamese_unet_levir_run1_7ep/`, `results/deeplearning_run1_7ep/` |
| CD-2 | FC-Siam-diff run 2, seed 42 (**shipped**) | full 13 epochs; chosen over CD-1 on VAL | VAL; TEST once | val F1 0.8738, test F1 0.8693, IoU 0.7688 | `models/siamese_unet_levir/`, `results/deeplearning/` |
| CD-3 | FC-Siam-diff, seed 7 | seed (variance estimate only) | VAL; TEST once | val F1 0.8684 | `results/deeplearning_seed7/` |
| CD-4 | FC-Siam-diff, seed 123 | seed (variance estimate only) | VAL; TEST once | three seeds: test F1 0.8650 ± 0.0058 | `results/deeplearning_seed123/` |
| CD-5 | Error analysis of CD-2 | none (post-hoc breakdown) | TEST (same predictions) | outline errors 60.6 %, small regions 17.6 % found | `results/deeplearning/test_error_analysis.txt`, `test_per_image.csv` |
| CD-6 | Area estimation error of CD-0 and CD-2 | none (arithmetic on recorded counts) | TEST counts | −0.53 % vs +291.92 % | `results/deeplearning/test_area_error.txt` |
| CD-7 | Ablation: no Dice term, seed 42 | `--dice-weight 0` | VAL only | val F1 0.8725 (threshold 0.75); 0.8470 at 0.50 | `models/siamese_unet_levir_no_dice/`, `results/deeplearning_no_dice/` |
| CD-8 | FC-Siam-diff + Transformer, seed 42 | bottleneck Transformer | VAL; TEST once | val F1 0.8711, test F1 0.8662, IoU 0.7640; within seed spread of the convolutional FC-Siam-diff (test F1 0.8693) | `models/siamese_unet_levir_transformer/`, `results/deeplearning_transformer/` |

## EuroSAT

| ID | Run | Split use | Main result | Files |
|---|---|---|---|---|
| ES-1 | ResNet land-use classifier, 20 epochs, seed 42 | VAL for the checkpoint; TEST once | test accuracy 0.9798, macro F1 0.9792 | `models/eurosat_resnet/`, `results/eurosat/` |
| ES-2 | Variational autoencoder, 25 epochs, seed 42 | VAL for the checkpoint; TEST once | test PSNR 23.55 dB (mean-image baseline 15.99 dB) | `models/eurosat_vae/`, `results/eurosat_vae/` |

## Fundamentals (plain Java, deterministic)

| ID | Run | Result | Files |
|---|---|---|---|
| F-1 | XOR with a 2-4-1 MLP and gradient check | loss 0.139816 → 0.000043; 13/13 gradients pass | `results/fundamentals/xor_and_gradient_check_output.txt` |
| F-2 | Learning rules (Hebbian, Oja + SVD check, competitive, perceptron, LMS, Boltzmann) | Oja agrees with the SVD to \|cos\| = 0.99999 | `results/fundamentals/learning_rules_demo_output.txt` |
| F-3 | SVD low-rank approximation of a EuroSAT tile | rank 16: 34.91 dB with half the numbers | `results/fundamentals/svd_demo_output.txt`, `svd_low_rank.png` |

## Hardware and software

Apple M3 laptop, 8 GB, GPU through PyTorch's MPS backend; Java 17.0.21.1, DJL 0.40.0, PyTorch 2.7.1. Training is not bit-for-bit reproducible on the GPU, so the logs and saved weights are the reference; the fundamentals are exactly reproducible.
