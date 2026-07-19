from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from config import RANDOM_STATE


def create_models():
    return {
        # n_neighbors and p are overridden by GridSearchCV in training_utils.py (see PCA_VARIANCE_OPTIONS grid);
        # these are only the fallback values used if a caller trains this model outside that search.
        "KNN": KNeighborsClassifier(n_neighbors=21, weights="distance", p=1),
        "SVM": SVC(kernel="rbf", C=1.0, gamma="scale", probability=True, class_weight="balanced", random_state=RANDOM_STATE),
        "XGBoost": XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.05, subsample=0.85, colsample_bytree=0.85, objective="binary:logistic", eval_metric="logloss", random_state=RANDOM_STATE, n_jobs=-1),
        "MLP": MLPClassifier(hidden_layer_sizes=(64,32), activation="relu", solver="adam", alpha=0.0001, learning_rate_init=0.001, max_iter=500, early_stopping=True, validation_fraction=0.15, random_state=RANDOM_STATE),
    }
