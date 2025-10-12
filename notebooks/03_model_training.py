# Databricks notebook source
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import pandas as pd

CATALOG = "zg"
SCHEMA = "production_scheduling_demo"
MODEL_NAME = f"{CATALOG}.{SCHEMA}.predictive_routing_model"

SEED = 42
mlflow.set_registry_uri("databricks-uc")

# COMMAND ----------

# Load labeled dataset
df = spark.table(f"{CATALOG}.{SCHEMA}.historical_routes_labeled").toPandas()
print("Loaded training data:", df.shape)

# COMMAND ----------

# Prepare features and target
feature_cols = [
    "part_category",
    "machine_id",
    "estimated_runtime_hours",
    "quantity",
    "margin",
    "processing_hours",
    "daily_capacity_hours",
    "efficiency_factor",
    "base_confidence"
]
target_col = "is_best"

# Drop NAs and split
df = df[[*feature_cols, target_col]].dropna().reset_index(drop=True)
X_train, X_test, y_train, y_test = train_test_split(
    df[feature_cols], df[target_col],
    test_size=0.2, random_state=SEED, stratify=df[target_col]
)

# COMMAND ----------

# Build preprocessing pipeline
categorical_features = ["part_category", "machine_id"]
numeric_features = [c for c in feature_cols if c not in categorical_features]

preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_features),
        ("num", StandardScaler(), numeric_features)
    ],
    remainder="drop"
)

# COMMAND ----------

# Build RandomForest pipeline
clf = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    random_state=SEED,
    n_jobs=-1
)
pipe = Pipeline(steps=[("preprocessor", preprocessor), ("model", clf)])

# COMMAND ----------

# Train & log with MLflow
with mlflow.start_run(run_name="predictive_routing_rf", log_system_metrics=True):
    # Train
    pipe.fit(X_train, y_train)
    test_accuracy = pipe.score(X_test, y_test)

    # Log params and metrics
    mlflow.log_params({
        "model_type": "RandomForestClassifier",
        "n_estimators": 200,
        "max_depth": 10,
        "random_state": SEED
    })
    mlflow.log_metrics({"test_accuracy": test_accuracy})

    # Log and register model to Unity Catalog
    mlflow.sklearn.log_model(
        sk_model=pipe,
        artifact_path="model",
        registered_model_name=MODEL_NAME,
        input_example=X_train.head(5),
        signature=mlflow.models.infer_signature(X_train, pipe.predict(X_train))
    )

print(f"✅ Model trained and registered to Unity Catalog as {MODEL_NAME}")
print(f"Test Accuracy: {test_accuracy:.3f}")

# COMMAND ----------

client = mlflow.MlflowClient()
versions = client.search_model_versions(f"name='{MODEL_NAME}'")
latest_version = max(versions, key=lambda v: int(v.version))
client.set_registered_model_alias(name=MODEL_NAME, alias="champion", version=latest_version.version)
