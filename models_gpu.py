
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
