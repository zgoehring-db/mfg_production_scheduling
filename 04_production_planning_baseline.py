# Databricks notebook source
import mlflow
import pandas as pd
import numpy as np
from pyspark.sql import functions as F
from utils.routing_helpers import greedy_assign_priority, compute_kpis

CATALOG = "zg"
SCHEMA = "production_scheduling_demo"
MODEL_NAME = f"{CATALOG}.{SCHEMA}.predictive_routing_model"
PLANNING_DAYS = 28  # scheduling horizon (can adjust)
SEED = 42

# COMMAND ----------

cand_df = spark.table(f"{CATALOG}.{SCHEMA}.candidate_routes")
machines_df = spark.table(f"{CATALOG}.{SCHEMA}.machines_catalog")

# Define features for model input
feature_cols = [
    "part_category",         # categorical
    "machine_id",            # categorical (candidate machine)
    "estimated_runtime_hours",
    "quantity",
    "margin",
    "processing_hours",
    "daily_capacity_hours",
    "efficiency_factor",
    "base_confidence"
]

# Convert to Pandas
cand_pd = cand_df.select(*feature_cols, "order_id", "promised_date").toPandas()

# Load model pipeline directly from Unity Catalog (sklearn flavor)
import mlflow.sklearn
model_uri = f"models:/{MODEL_NAME}@champion"
pipe = mlflow.sklearn.load_model(model_uri)

# Predict probability that candidate is best
probs = pipe.predict_proba(cand_pd[feature_cols])
class1_idx = list(pipe.named_steps["model"].classes_).index(1)
cand_pd["p_best"] = probs[:, class1_idx]

# Profit-aware score
cand_pd["profit_score"] = cand_pd["p_best"] * cand_pd["margin"]

# Sort orders by profit_score (high first)
cand_pd = cand_pd.sort_values(by="profit_score", ascending=False).reset_index(drop=True)

# COMMAND ----------

# saved scored candidates to delta table
cand_sdf = spark.createDataFrame(cand_pd)
cand_sdf.write.mode("overwrite").saveAsTable(f"{CATALOG}.{SCHEMA}.candidate_routes_scored")

# COMMAND ----------

# Baseline scenario (all machines operational)
machines_pd = machines_df.toPandas()
assigned_baseline, caps_baseline = greedy_assign_priority(cand_pd, machines_pd, PLANNING_DAYS, down_machines=[])

# Convert assigned baseline to Spark DF and save
assigned_baseline_sdf = spark.createDataFrame(assigned_baseline)
assigned_baseline_sdf.write.mode("overwrite").saveAsTable(f"{CATALOG}.{SCHEMA}.assigned_baseline")

# Convert capacity dict to a DataFrame and save
caps_baseline_df = pd.DataFrame([
    {"machine_id": k, "remaining_hours": v} for k, v in caps_baseline.items()
])
spark.createDataFrame(caps_baseline_df).write.mode("overwrite").saveAsTable(f"{CATALOG}.{SCHEMA}.capacity_baseline")

# COMMAND ----------

display(assigned_baseline_sdf)

# COMMAND ----------

display(caps_baseline_df)

# COMMAND ----------

baseline_kpis = kpi_base = compute_kpis(assigned_baseline, machines_pd)

# COMMAND ----------

print(baseline_kpis)

# COMMAND ----------


