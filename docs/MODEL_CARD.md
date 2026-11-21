# Model card: LandWatch AI models

This card follows "Model Cards for Model Reporting" (Mitchell et al., 2019). Every number is copied from a file in `results/` or `models/` (named in each section). The data are described in the [dataset card](DATASET_CARD.md).

| Model | Task | Files | Main test result |
|---|---|---|---|
| FC-Siam-diff (shipped) | building-change detection, LEVIR-CD | `models/siamese_unet_levir/` | change F1 0.8693, IoU 0.7688 |
| FC-Siam-diff + Transformer | building-change detection, LEVIR-CD | `models/siamese_unet_levir_transformer/` | change F1 0.8662, IoU 0.7640 (within seed spread of the shipped model) |
| EuroSAT ResNet | land-use scene classification | `models/eurosat_resnet/` | accuracy 0.9798, macro F1 0.9792 |
| EuroSAT VAE | generative model of land-cover patches | `models/eurosat_vae/` | reconstruction PSNR 23.55 dB (baseline 15.99 dB) |

All models were trained from scratch (no pretrained weights) in Java with DJL 0.40 and PyTorch 2.7.1 on an Apple M3 laptop GPU (MPS). Training is not bit-for-bit reproducible; the recorded runs and logs are the reference.

---

## FC-Siam-diff (the shipped change detector)

**Details.** Fully convolutional Siamese network after Daudt, Le Saux and Boulch (ICIP 2018): a shared 4-stage encoder (16-32-64-128 channels) for both dates, a decoder fed with the absolute feature differences |enc(A) − enc(B)| at every scale, and one change logit per pixel. 1,348,785 parameters. Weights `fc_siam_diff-0013.params`; every setting is in `model.properties`.

**Training.** LEVIR-CD train split (7,120 tiles of 256×256), 13 epochs, batch 8, Adam with cosine decay from 1e-3 to 1e-5, weighted BCE (positive weight 4.5597) plus soft Dice, random D4 rotations and flips, normalisation from train images only, seed 42; 83 minutes. The checkpoint and the decision threshold (0.65) were chosen on the validation split.

**Intended use.** Detecting where buildings appeared or disappeared between two co-registered 0.5 m RGB images similar to LEVIR-CD; teaching and demonstration.

**Out of scope.** Other resolutions or sensors without retraining; deforestation, roads, water or crops (never labelled); telling construction from demolition (the model is order-symmetric); any legal or enforcement decision about encroachment.

**Results** (`results/deeplearning/test_metrics.txt`, `test_error_analysis.txt`, `test_area_error.txt`; the test split was evaluated once).

| Measure | Value |
|---|---|
| Test change F1 / IoU | 0.8693 / 0.7688 |
| Test precision / recall / Cohen's κ | 0.8716 / 0.8670 / 0.8623 |
| Validation change F1 | 0.8738 |
| Test F1 over three seeds | 0.8650 ± 0.0058 |
| Per-image F1, median (quartiles) | 0.857 (0.803–0.904) |
| Change regions found: all / under 25 m² | 92.3 % / 17.6 % |
| Error of the total changed area | −0.53 % |
| Classical baseline on the same test pixels | F1 0.1008, IoU 0.0531 |

**Factors and limitations.**
- Errors concentrate on building outlines (60.6 % of wrong pixels lie within 2 px of a true outline).
- Small buildings are often missed: only 17.6 % of change regions under 25 m² are found. Where small or informal structures matter, the model would under-report them.
- Sites where one building replaces another are missed (both dates show a roof), and bright vehicles in yards can be mistaken for new buildings.
- All training and test images come from one region (Texas, USA) and one source; performance elsewhere is unknown.

**Ethical considerations.** The images show private property at 0.5 m. The model must not be used to monitor individuals, and any decision about a specific site needs a human check of the images.

## FC-Siam-diff + Transformer (architecture ablation)

**Details.** The same FC-Siam-diff encoder and decoder, but with two pre-norm Transformer encoder layers (4 heads, MLP 128→256→128, fixed 2-D sinusoidal position code) applied to the bottleneck tokens of both dates, following the idea of BIT (Chen, Qi and Shi, TGRS 2022). 1,614,001 parameters (+265,216 from the Transformer). Weights `fc_siam_diff_transformer-0012.params`; every setting is in `model.properties`.

**Training.** Identical recipe to the shipped FC-Siam-diff (same data, loss, augmentation, optimiser, 13 epochs, seed 42), trained on an NVIDIA GPU with the CUDA build of PyTorch; 25.5 minutes. The epoch-12 checkpoint and threshold 0.65 were chosen on VAL.

**Intended use.** Architecture ablation: testing whether global self-attention at the bottleneck improves on purely convolutional comparison.

**Results** (`results/deeplearning_transformer/test_metrics.txt`, `val_metrics.txt`; the test split was evaluated once).

| Measure | FC-Siam-diff + Transformer | FC-Siam-diff (shipped) | Difference |
|---|---|---|---|
| Test change F1 / IoU | 0.8662 / 0.7640 | 0.8693 / 0.7688 | −0.0031 / −0.0048 |
| Test precision / recall / κ | 0.8683 / 0.8641 / 0.8591 | 0.8716 / 0.8670 / 0.8623 | |
| Val change F1 | 0.8711 | 0.8738 | −0.0027 |
| Area error (test) | −0.48 % | −0.53 % | |

**Conclusion.** Both differences are within the seed spread (VAL F1 ±0.0028, TEST F1 ±0.0058 over three seeds). Self-attention at the bottleneck neither helps nor hurts on LEVIR-CD at this model size.

## EuroSAT ResNet (land-use classifier)

**Details.** ResNet-18-style classifier for 64×64 RGB tiles: residual stages of 32-64-128-256 channels, global average pooling and a linear layer to 10 classes; 2,797,610 parameters. Files in `models/eurosat_resnet/`.

**Training.** EuroSAT train split (18,900 images), 20 epochs, Adam with cosine decay, D4 augmentation, train-only normalisation, seed 42; 26.8 minutes. The epoch-19 checkpoint was chosen by validation accuracy.

**Intended use.** Labelling a whole 640 m × 640 m Sentinel-2 RGB scene (10 m pixels) with one of 10 land-use classes; the dashboard's Land use page.

**Out of scope.** Images of another resolution, such as LEVIR-CD's 0.5 m pixels (the dashboard refuses them); per-pixel maps; the multispectral bands.

**Results** (`results/eurosat/test_metrics.txt`, test split of 4,050 images evaluated once): accuracy 0.9798, macro F1 0.9792, 82 errors, mostly between similar vegetation classes. Validation accuracy 0.9726.

**Limitations.** The split is random, so tiles from the same area can be in train and test, and the score may overstate performance on new regions. Its probabilities were not checked for calibration, and it always answers with one of its 10 classes: the synthetic 400×300 demo image (a green field crossed by a blue stripe, resized to 64×64) gets "River" at 97.47 %, a confident answer for an image unlike any training tile (screenshot `results/ui_snapshots/21_page_land_use_eurosat_resized_input_note.png`).

## EuroSAT VAE (generative model)

**Details.** Convolutional variational autoencoder (Kingma and Welling, 2014): a 4-layer strided-convolution encoder to the mean and log-variance of a 64-number code, the reparameterisation trick, and a decoder of upsampling plus convolutions back to 64×64×3 logits. 1,878,211 parameters. Files in `models/eurosat_vae/` (`vae.properties` holds every setting).

**Training.** EuroSAT train split (18,900 images, labels not used), negative ELBO (Bernoulli reconstruction + KL), 25 epochs, batch 64, Adam with cosine decay, D4 augmentation, seed 42; 14.5 minutes. The epoch-24 checkpoint had the lowest validation negative ELBO.

**Intended use.** Teaching: showing how a generative model with a probabilistic code is trained and evaluated. It is not used by the change detector or the dashboard.

**Results** (`results/eurosat_vae/test_metrics.txt`, test split evaluated once): negative ELBO 7,608.70 nats per image (reconstruction 7,581.30, KL 27.40); reconstruction MSE 0.004420 (PSNR 23.55 dB), 82.4 % below the mean-image baseline (0.025164, 15.99 dB). Best class Forest (31.48 dB), worst Industrial (18.12 dB).

**Limitations.** Reconstructions are blurry and lose roads, field edges and buildings; images sampled from the prior are smooth colour patterns, not realistic scenes. It must not be used to create imagery that could pass for real observations.
