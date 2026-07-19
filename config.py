from pathlib import Path

RANDOM_STATE = 42
TEST_SIZE = 0.20
PCA_VARIANCE = 0.95
PCA_VARIANCE_OPTIONS = [0.90, 0.95, 0.99]


SVM_MAX_TRAIN_ROWS = 40000

DATA_PATH = Path("archive (3)/heart_disease_health_indicators_BRFSS2015.csv")
OUTPUT_DIR = Path("outputs")
PLOTS_DIR = OUTPUT_DIR / "plots"
MODELS_DIR = OUTPUT_DIR / "models"

TARGET_COLUMN = "HeartDiseaseorAttack"
ID_COLUMN = "patient_id"

CLINICAL_COLUMNS = [
    "HighBP", "HighChol", "CholCheck", "BMI", "Stroke", "Diabetes",
    "GenHlth", "MentHlth", "PhysHlth", "DiffWalk", "Sex", "Age",
    "Education", "Income",
]

LIFESTYLE_COLUMNS = [
    "Smoker", "PhysActivity", "Fruits", "Veggies", "HvyAlcoholConsump",
    "AnyHealthcare", "NoDocbcCost",
]

ALL_FEATURE_COLUMNS = CLINICAL_COLUMNS + LIFESTYLE_COLUMNS
