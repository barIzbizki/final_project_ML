# Heart Disease Project with PCA

Pipeline for every model:

preprocessing -> StandardScaler -> PCA (95% explained variance) -> model

The scaler and PCA are fitted only on the training data.

Run:

```bash
pip install -r requirements.txt
python run_all.py
```

Put the dataset at:

```text
data/heart_disease_dataset.csv
```
