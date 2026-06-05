# AutoDL Experiment Results Template

This directory contains results from the three experiments for paper revision.

## Experiment 1: OFF-ApexNet LOSO Reproduction

File: `offapexnet_loso_folds.csv`

Expected columns:
- fold: Fold number (1-24)
- test_subject: Subject ID (sub01-sub24)
- accuracy: Per-fold accuracy (%)
- predicted_labels: List of predicted class indices
- true_labels: List of true class indices

Summary: `offapexnet_loso_summary.txt`
- Mean accuracy
- Standard deviation
- Per-class F1 scores

## Experiment 2: rPPG Signal Quality Validation

File: `rppg_validation.json`

Expected fields:
- num_samples: Number of samples analyzed
- hr_mean: Mean heart rate (BPM)
- hr_std: Heart rate standard deviation (BPM)
- snr_mean: Mean signal-to-noise ratio (dB)
- valid_hr_rate: Percentage of samples with HR in 60-100 BPM range
- hr_distribution: Histogram data

## Experiment 3: Inference Latency Benchmark

File: `latency_benchmark.txt`

Expected fields per configuration:
- num_params: Number of parameters
- mean_ms: Mean latency (milliseconds)
- std_ms: Latency standard deviation
- p50_ms, p95_ms, p99_ms: Latency percentiles
- throughput: Frames per second
- gpu_memory_mb: Peak GPU memory usage (MB)

Configurations tested:
- Full Model (68.35M params)
- Fast-only (12.85M params)
- No-rPPG (57.90M params)
