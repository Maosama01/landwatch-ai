/**
 * Deep-learning change detection for LandWatch AI: an FC-Siam-diff network trained on LEVIR-CD
 * with DJL (Deep Java Library) and its PyTorch engine.
 *
 * <h2>Pieces</h2>
 * <ul>
 *   <li>{@link com.landwatch.deeplearning.FcSiamDiff} - the network (a DJL custom block), with the
 *       architecture explained in its Javadoc.</li>
 *   <li>{@link com.landwatch.deeplearning.BceDiceLoss} - weighted cross-entropy + Dice loss for the
 *       heavily imbalanced change / no-change pixels.</li>
 *   <li>{@link com.landwatch.deeplearning.LevirCdPatch}, {@link com.landwatch.deeplearning.LevirCdDataset},
 *       {@link com.landwatch.deeplearning.ImageTensors}, {@link com.landwatch.deeplearning.Normalization},
 *       {@link com.landwatch.deeplearning.D4Augmentation} - reading the CSV metadata and PNG patches,
 *       normalisation with TRAIN statistics, and flip / rotation augmentation.</li>
 *   <li>{@link com.landwatch.deeplearning.TrainChangeDetector} - training CLI (train on TRAIN, choose
 *       checkpoint and threshold on VAL).</li>
 *   <li>{@link com.landwatch.deeplearning.EvaluateChangeDetector} - evaluation CLI (run once on TEST),
 *       metrics via {@link com.landwatch.evaluation.BinaryChangeMetrics}.</li>
 *   <li>{@link com.landwatch.deeplearning.DeepChangeDetector} - the inference API used by the dashboard
 *       (any image size: tiling, mirror padding, stitching).</li>
 *   <li>{@link com.landwatch.deeplearning.DjlRuntime}, {@link com.landwatch.deeplearning.TrainingSupport},
 *       {@link com.landwatch.deeplearning.ModelConfig}, {@link com.landwatch.deeplearning.ProbabilityHistogram},
 *       {@link com.landwatch.deeplearning.DeviceBenchmark} - plumbing: devices and versions, the training
 *       step, the saved configuration, the threshold sweep, and the CPU-vs-MPS benchmark.</li>
 * </ul>
 *
 * <h2>Data protocol</h2>
 * <p>TRAIN patches are used for learning (and for the normalisation constants), VAL patches for
 * choosing the checkpoint and the decision threshold, and TEST patches only once, for the final
 * numbers. Mixing these up would make the reported metrics optimistic.</p>
 */
package com.landwatch.deeplearning;
