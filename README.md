📦 This repository contains the full implementation and reproduction package of our project:
“Few-shot foundation-model mixture of experts for cereal mapping in data-scarce regions.”

The goal of this work is to evaluate whether combining multiple foundation models (Prithvi-EO V2 and Satlas) through a Mixture-of-Experts (MoE) strategy improves cereal crop mapping performance under extremely limited labeled data.

# We experiment with:
1. Foundation models as base encoders
2. Several data augmentation strategies
3. A small Algerian cereal dataset (published on Zenodo)
4. A final ONNX-exported MoE for fast, portable inference

# The repository is structured to be fully reproducible, allowing anyone to:
1. Re-run all experiments using the 9 provided notebooks
2. Inspect the intermediate results from each model and augmentation variant
3. Test the final MoE model locally through a lightweight demo app.
   
📁 Repository Structure
Finak-Year-Project/
│
├── notebooks/
│   ├── prithvi_baseline.ipynb
│   ├── prithvi_geom_aug.ipynb
│   ├── prithvi_rad_aug.ipynb
│   ├── prithvi_randaugment.ipynb
│   ├── satlas_baseline.ipynb
│   ├── satlas_geom_aug.ipynb
│   ├── satlas_rad_aug.ipynb
│   ├── satlas_randaugment.ipynb
│   ├── mixture_of_experts.ipynb
│   ├── README.md              ← explains how to re-run the experiments
│   └── requirements.txt       ← minimal environment for notebooks
│
├── demo_app/
│   ├── app.py                 ← simple Python demo for ONNX inference
│   ├── requirements.txt
│   ├── sample_inputs/
│   └── README.md              ← instructions for running the demo + download link for ONNX model (~1.6 GB)
│
└── README.md (this file)

📊 Dataset
The dataset used in this project is publicly available:
📥 Zenodo Dataset Link:
(paste your Zenodo link here)

📝 Citation
