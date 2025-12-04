# Notebooks: Reproduce the experiments

This folder contains 9 notebooks used for the experiments:
1. `PrithviBaseline.ipynb`
2. `PrithviGeometricAugmentation.ipynb`
3. `PrithviRadiometricAugmentation.ipynb`
4. `PrithviRandAugment.ipynb`
5. `SatlasBaseline.ipynb`
6. `SatlasGeometric.ipynb`
7. `SatlasRadiometric.ipynb`
8. `SatlasRandAugment.ipynb`
9. `MixtureOfExperts.ipynb` (final integration demo; also can be run with ONNX model)

#Order to run (recommended)

Run the Prithvi or Satlas fine-tuning notebooks (1–4 for Prithvi or 5–8 for Satlas) to reproduce single-expert training and logs.

Each notebook is self-contained; set your DATA_DIR, CHECKPOINT_DIR variables at the top.

Run 'MixtureOfExperts.ipynb' to load the frozen experts and train the MoE gating network (or load the provided ONNX model for inference).

For full training you need the dataset (Zenodo DOI)


