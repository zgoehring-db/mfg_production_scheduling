# Databricks notebook source
import mlflow
import pandas as pd
import numpy as np
from pyspark.sql import functions as F

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

import sys
import os

# Add parent directory to the Python path so utils is discoverable
sys.path.append(os.path.abspath('.'))

from utils.routing_helpers import greedy_assign_priority, compute_kpis_from_assigned

# COMMAND ----------

# Greedy assignment with priority (profit, due date, hours)
def greedy_assign_priority(candidates_df, machines_catalog_df, planning_days=PLANNING_DAYS, down_machines=None):
    down_machines = set(down_machines or [])

    # Build capacity map
    if hasattr(machines_catalog_df, "toPandas"):  # Spark DF
        machines_list = machines_catalog_df.toPandas().to_dict(orient="records")
    elif isinstance(machines_catalog_df, pd.DataFrame):
        machines_list = machines_catalog_df.to_dict(orient="records")
    else:
        machines_list = list(machines_catalog_df)

    machine_caps = {
        m["machine_id"]: float(m["daily_capacity_hours"] * planning_days)
        for m in machines_list
    }
    for m in down_machines:
        machine_caps[m] = 0.0

    # Ensure pandas
    cand = candidates_df if isinstance(candidates_df, pd.DataFrame) else candidates_df.toPandas()

    # Priority: profit_score > earliest due > shortest hours
    order_priority = (
        cand.groupby("order_id")
            .agg(max_profit_score=("profit_score", "max"),
                 min_due=("promised_date", "min"),
                 min_hours=("processing_hours", "min"))
            .sort_values(by=["max_profit_score", "min_due", "min_hours"],
                         ascending=[False, True, True])
            .reset_index()
    )

    # Pre-split by order for speed
    cand_by_order = {
        oid: g.sort_values(by=["profit_score", "processing_hours"], ascending=[False, True])
        for oid, g in cand.groupby("order_id", sort=False)
    }

    assigned = []
    for _, rowp in order_priority.iterrows():
        oid = rowp["order_id"]
        group = cand_by_order[oid]
        for _, c in group.iterrows():
            mid = c["machine_id"]
            if mid in down_machines:
                continue
            need = float(c["processing_hours"])
            if machine_caps.get(mid, 0.0) >= need:
                machine_caps[mid] -= need
                assigned.append({
                    "order_id": oid,
                    "machine_id": mid,
                    "profit": float(c["margin"]),
                    "p_best": float(c["p_best"]),
                    "processing_hours": need,
                    "profit_score": float(c["profit_score"]),
                    "base_confidence": float(c["base_confidence"])
                })
                break  # assigned
    return pd.DataFrame(assigned), machine_caps

# COMMAND ----------

# Baseline scenario (all machines operational)
machines_pd = machines_df.toPandas()
assigned_baseline, caps_baseline = greedy_assign(cand_pd, machines_pd, PLANNING_DAYS, down_machines=[])

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