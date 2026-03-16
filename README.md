# Few-Shot Foundation-Model Mixture of Experts for Cereal Mapping in Data-Scarce Regions


📦 This repository contains the full implementation and reproduction
package of my project:\
**"Few-shot foundation-model mixture of experts for cereal mapping in
data-scarce regions."**

The goal of this work is to evaluate whether combining multiple
foundation models (**Prithvi-EO V2** and **Satlas**) through a
**Mixture-of-Experts (MoE)** strategy improves cereal mapping
performance when labeled data is extremely limited.

------------------------------------------------------------------------

## 🌱 Project Highlights

We experiment with:

1.  **Foundation models as base encoders**\
2.  **Several data augmentation strategies** (geometric, radiometric,
    RandAugment)\
3.  **A small Algerian cereal dataset** (published on Zenodo)\
4.  **A final ONNX-exported MoE model** for fast and portable inference

------------------------------------------------------------------------

## 🔁 Reproducibility

This repository is fully reproducible, allowing anyone to:

1.  Re-run all experiments using the **9 provided notebooks**.\
2.  Inspect the **intermediate results** for each model and augmentation
    strategy.\
3.  Test the final **MoE ONNX model** through a lightweight local demo
    app.

------------------------------------------------------------------------

## 📁 Repository Structure

``` txt
Final-Year-Project/
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
│   ├── README.md              ← instructions for re-running experiments
│   └── requirements.txt       ← minimal environment for notebooks
│
├── demo_app/
│   ├── app.py                 ← simple Python demo for ONNX inference
│   ├── requirements.txt
│   ├── sample_inputs/
│   └── README.md              ← demo instructions + ONNX model link (~1.6 GB)
│
└── README.md                  ← this file
│
└── The-Paper                  ← the scientific publication produced by this work
```

------------------------------------------------------------------------

## 📊 Dataset

The dataset used in this work is publicly available on Zenodo.

🔗 **Zenodo Dataset Link:**\
https://doi.org/10.5281/zenodo.17674368

------------------------------------------------------------------------

## 📝 Citation

If you use this work or build upon it, please cite it appropriately:
