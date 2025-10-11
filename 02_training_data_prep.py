# Databricks notebook source
CATALOG = "zg"
SCHEMA = "production_scheduling_demo"

# COMMAND ----------

from pyspark.sql import functions as F, Window

# Create target table for labeled routes
spark.sql(f"""
CREATE TABLE IF NOT EXISTS {CATALOG}.{SCHEMA}.historical_routes_labeled (
  order_id STRING COMMENT 'Unique order identifier',
  part_id STRING COMMENT 'Part being manufactured',
  part_category STRING COMMENT 'Part category (metal, battery, electronics, assembly)',
  customer_name STRING COMMENT 'Customer placing the order',
  quantity INT COMMENT 'Units ordered',
  promised_date DATE COMMENT 'Promised delivery date',
  sale_price DOUBLE COMMENT 'Sale price per unit',
  material_cost DOUBLE COMMENT 'Material cost per unit',
  estimated_runtime_hours DOUBLE COMMENT 'Baseline runtime per unit (hours)',
  machine_id STRING COMMENT 'Machine candidate ID',
  machine_name STRING COMMENT 'Descriptive name of candidate machine',
  capability STRING COMMENT 'Machine capability (category match)',
  daily_capacity_hours DOUBLE COMMENT 'Machine available hours per day',
  efficiency_factor DOUBLE COMMENT 'Machine efficiency',
  is_operational BOOLEAN COMMENT 'Machine operational status',
  processing_hours DOUBLE COMMENT 'Estimated hours required on this machine',
  margin DOUBLE COMMENT 'Profit margin for the order',
  base_confidence DOUBLE COMMENT 'Historical route confidence (0–1)',
  profit_score DOUBLE COMMENT 'Synthetic performance score (margin × base_confidence)',
  is_best INT COMMENT 'Label: 1 if this was the best (historical) machine, else 0'
)
""")

cand_df = spark.table(f"{CATALOG}.{SCHEMA}.candidate_routes")

# Compute synthetic profit_score
cand_df = cand_df.withColumn("profit_score", F.col("margin") * F.col("base_confidence"))

# Rank machines per order
win = Window.partitionBy("order_id").orderBy(F.desc("profit_score"))
labeled_df = (
    cand_df
    .withColumn("rank", F.row_number().over(win))
    .withColumn("is_best", F.when(F.col("rank") == 1, F.lit(1)).otherwise(F.lit(0)))
    .drop("rank")
)

labeled_df.write.mode("overwrite").saveAsTable(f"{CATALOG}.{SCHEMA}.historical_routes_labeled")

display(labeled_df)

# COMMAND ----------

