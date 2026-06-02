# DVC Setup

Current environment:

- DVC version: 3.67.1
- Python: 3.11.9
- OS: Windows 10
- Cache: local NTFS on `C:`
- Remote: none configured

## Initialize DVC

```bash
dvc init
git add .dvc .gitignore .dvcignore dvc.yaml
```

## Track pipeline outputs

The preprocessing stage writes these outputs:

- `mimic_processed/X_train.npy`
- `mimic_processed/X_test.npy`
- `mimic_processed/y_train.npy`
- `mimic_processed/y_test.npy`
- `mimic_processed/feature_names_final.csv`
- `mimic_processed/preprocessing_pipeline.joblib`

If they already exist, track them with:

```bash
dvc add mimic_processed/X_train.npy mimic_processed/X_test.npy mimic_processed/y_train.npy mimic_processed/y_test.npy mimic_processed/feature_names_final.csv mimic_processed/preprocessing_pipeline.joblib
git add mimic_processed/*.dvc .gitignore
```

## Reproduce

```bash
dvc repro
```

## Optional remote

```bash
dvc remote add -d myremote <remote-url>
dvc push
```
