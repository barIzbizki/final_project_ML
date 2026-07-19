"""GPU-backed model factory for the archive3 pipeline.

Mirrors create_models() in models.py, but KNN and SVM come from cuML (RAPIDS) and
XGBoost targets the GPU. Kept separate from models.py so the archive/archive2
pipelines keep running the CPU sklearn estimators unchanged.

Two things differ from the sklearn versions and are load-bearing:

* cuML's KNeighborsClassifier accepts `p` but silently ignores it -- the argument is
  swallowed by **kwargs and the distance is always Euclidean. Distance choice has to go
  through `metric`, which was verified to reproduce sklearn's p=1/p=2 results exactly.
  training_utils_archive3.py therefore grids over `model__metric`, not `model__p`.
* MLPClassifier has no GPU implementation in either sklearn or cuML, so it stays on CPU.

PCA is deliberately NOT taken from cuML: cuML's PCA requires an integer n_components and
raises on the fractional variance targets in PCA_VARIANCE_OPTIONS. It runs on 14-21
features here, so keeping sklearn's costs nothing.
"""
from cuml.neighbors import KNeighborsClassifier as KNeighborsClassifierGPU
from cuml.svm import SVC as SVCGPU
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier

from config_archive3 import RANDOM_STATE


def create_models():
    return {
        # n_neighbors and metric are overridden by GridSearchCV in training_utils_archive3.py;
        # these are only the fallback values used if a caller trains this model outside that search.
        "KNN": KNeighborsClassifierGPU(n_neighbors=21, weights="distance", metric="manhattan"),
        "SVM": SVCGPU(kernel="rbf", C=1.0, gamma="scale", probability=True, class_weight="balanced", random_state=RANDOM_STATE),
        "XGBoost": XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.05, subsample=0.85, colsample_bytree=0.85, objective="binary:logistic", eval_metric="logloss", random_state=RANDOM_STATE, n_jobs=-1, tree_method="hist", device="cuda"),
        "MLP": MLPClassifier(hidden_layer_sizes=(64,32), activation="relu", solver="adam", alpha=0.0001, learning_rate_init=0.001, max_iter=500, early_stopping=True, validation_fraction=0.15, random_state=RANDOM_STATE),
    }
