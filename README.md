# SQL Injection Detection Pipeline

This repository provides a centroid-based SQL injection detection workflow built on top of sentence-transformer embeddings. It covers

* dataset validation and SQL-aware preprocessing;
* embedding extraction for full queries plus dynamic local signatures;
* centroid construction per attack technique with automatic threshold calibration;
* FAISS-backed nearest centroid search with 8-bit quantisation;
* inference with localisation of the suspicious fragment and centroid hot-reload for new techniques.

## Project Structure

```
.
├── artifacts/               # Generated artifacts (embeddings, centroids, metrics, notebooks)
├── sql_injection/           # Core Python package implementing the pipeline
├── train.py                 # Entry point for running the training pipeline
├── inference.py             # CLI for inference on new queries
└── requirements.txt         # Python dependencies
```

## Getting Started

1. Create and activate a Python 3.9+ virtual environment.
2. Install dependencies: `pip install -r requirements.txt`.
3. Place the labelled dataset at `data.csv` with columns `full_query`, `label`, `user_inputs`, and `attack_technique`.
4. Run the training pipeline:

   ```bash
   python train.py
   ```

   Artifacts such as embeddings, centroids, thresholds, metrics, and logs will be stored under `artifacts/`.

## Inference

After training, analyse a new query via the CLI:

```bash
python inference.py --query "SELECT * FROM users WHERE id = 1"
```

The command outputs a JSON structure containing the predicted label, technique, confidence, and the highlighted fragment most similar to the corresponding centroid.

To register a new attack technique from a handful of examples, load `SQLInjectionDetector` from `inference.py` and call `add_new_technique(technique_name, samples)`; the updated centroids and FAISS index are persisted automatically.

## Artifacts

Key generated files include:

* `artifacts/embeddings/embeddings_full.npy` – embeddings for each query.
* `artifacts/centroids/centroids.json` – class and position centroids with technique order.
* `artifacts/centroids/class_index.faiss` – FAISS scalar-quantised index over class centroids.
* `artifacts/metrics/thresholds.json` – calibrated detection thresholds.
* `artifacts/logs/run.log` – run logs.

The optional notebook `artifacts/analysis/validate_thresholds.ipynb` can be used to explore ROC/PR curves after training.

