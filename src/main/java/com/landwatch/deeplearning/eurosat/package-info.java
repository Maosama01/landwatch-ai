/**
 * Land-use / land-cover scene classification on EuroSAT with a ResNet trained in Java with DJL
 * (Deep Java Library) and its PyTorch engine.
 *
 * <h2>Scope</h2>
 * <p>This is a separate experiment from the LEVIR-CD change detector in the parent package. EuroSAT
 * (Helber et al., IEEE JSTARS 2019) contains 27,000 Sentinel-2 RGB tiles of 64 x 64 pixels at 10 m
 * ground sample distance, each labelled with one of 10 classes (AnnualCrop, Forest,
 * HerbaceousVegetation, Highway, Industrial, Pasture, PermanentCrop, Residential, River, SeaLake).
 * The model answers "what kind of land is this whole tile?". It does not detect change and is not
 * meant for LEVIR-CD images, which are 0.5 m aerial photographs (a different sensor, a 20x finer
 * resolution and a different task).</p>
 *
 * <h2>Pieces</h2>
 * <ul>
 *   <li>{@link com.landwatch.deeplearning.eurosat.EuroSatSplit} - the stratified 70/15/15
 *       train / val / test split (fixed seed), written to {@code datasets/EuroSAT/splits/*.csv}.</li>
 *   <li>{@link com.landwatch.deeplearning.eurosat.ResidualBlock} and
 *       {@link com.landwatch.deeplearning.eurosat.EuroSatResNet} - the network: a compact
 *       ResNet-18-style CNN for 64 x 64 input; the Javadoc explains the residual idea
 *       {@code y = F(x) + x} and the architecture.</li>
 *   <li>{@link com.landwatch.deeplearning.eurosat.EuroSatImages} - decoding the JPEGs once into
 *       memory, TRAIN normalisation statistics, batches with D4 (flip / 90-degree rotation)
 *       augmentation.</li>
 *   <li>{@link com.landwatch.deeplearning.eurosat.TrainEuroSatClassifier} - training CLI (train on
 *       TRAIN, select the epoch on VAL).</li>
 *   <li>{@link com.landwatch.deeplearning.eurosat.EvaluateEuroSatClassifier} - evaluation CLI (VAL,
 *       and TEST exactly once), metrics from {@link com.landwatch.evaluation.MulticlassMetrics}.</li>
 *   <li>{@link com.landwatch.deeplearning.eurosat.EuroSatClassifier} - the inference API;
 *       {@link com.landwatch.deeplearning.eurosat.PredictEuroSatClass} - a small CLI around it.</li>
 *   <li>{@link com.landwatch.deeplearning.eurosat.EuroSatTraining} and
 *       {@link com.landwatch.deeplearning.eurosat.EuroSatModelConfig} - DJL plumbing (trainer,
 *       training step, checkpoints) and the saved {@code model.properties}.</li>
 * </ul>
 *
 * <p>Reused from the parent package: {@link com.landwatch.deeplearning.DjlRuntime} (device
 * selection), {@link com.landwatch.deeplearning.ImageTensors} (image to CHW floats),
 * {@link com.landwatch.deeplearning.Normalization} (per-channel standardisation),
 * {@link com.landwatch.deeplearning.D4Augmentation} (the 8 flips / rotations) and
 * {@link com.landwatch.deeplearning.TrainingSupport} (parameter counting, device-to-CPU copies).</p>
 */
package com.landwatch.deeplearning.eurosat;
