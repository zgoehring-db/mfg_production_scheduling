# Databricks notebook source
# MAGIC %sql
# MAGIC drop schema if exists zg.production_scheduling_demo cascade

# COMMAND ----------

CATALOG = "zg"
SCHEMA = "production_scheduling_demo"
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.{SCHEMA}")

# COMMAND ----------

# Create tables
# Machines schema
spark.sql(f"""
CREATE TABLE IF NOT EXISTS {CATALOG}.{SCHEMA}.machines_catalog (
  machine_id STRING COMMENT 'Unique machine identifier',
  machine_name STRING COMMENT 'Descriptive name of the machine',
  capability STRING COMMENT 'Primary part category the machine can process (metal, battery, electronics, assembly)',
  daily_capacity_hours DOUBLE COMMENT 'Available production hours per day',
  efficiency_factor DOUBLE COMMENT 'Relative processing efficiency (1.0 = baseline)',
  is_operational BOOLEAN COMMENT 'Indicates if the machine is currently operational'
)
""")

# Parts schema
spark.sql(f"""
CREATE TABLE IF NOT EXISTS {CATALOG}.{SCHEMA}.parts_catalog (
  part_id STRING COMMENT 'Unique part identifier',
  part_name STRING COMMENT 'Part description or name',
  category STRING COMMENT 'Part type or category (metal, battery, electronics, assembly)',
  material_cost DOUBLE COMMENT 'Material cost per unit',
  default_price DOUBLE COMMENT 'Selling price per unit',
  estimated_runtime_hours DOUBLE COMMENT 'Baseline estimated runtime (hours) to produce one unit'
)
""")

# Orders schema
spark.sql(f"""
CREATE TABLE IF NOT EXISTS {CATALOG}.{SCHEMA}.orders_backlog (
  order_id STRING COMMENT 'Unique order identifier',
  part_id STRING COMMENT 'Part being ordered',
  category STRING COMMENT 'Part category (joins to parts.category)',
  customer_name STRING COMMENT 'Customer placing the order',
  quantity INT COMMENT 'Units ordered',
  promised_date DATE COMMENT 'Promised delivery date',
  sale_price DOUBLE COMMENT 'Sale price per unit',
  material_cost DOUBLE COMMENT 'Material cost per unit',
  estimated_runtime_hours DOUBLE COMMENT 'Baseline runtime per unit (hours)'
)
""")

# Candidate routes
spark.sql(f"""
CREATE TABLE IF NOT EXISTS {CATALOG}.{SCHEMA}.candidate_routes (
  order_id STRING COMMENT 'Unique order identifier',
  part_id STRING COMMENT 'Part being manufactured',
  part_category STRING COMMENT 'Part category (metal, battery, electronics, assembly)',
  customer_name STRING COMMENT 'Customer placing the order',
  quantity INT COMMENT 'Units ordered',
  promised_date DATE COMMENT 'Promised delivery date',
  sale_price DOUBLE COMMENT 'Sale price per unit',
  material_cost DOUBLE COMMENT 'Material cost per unit',
  estimated_runtime_hours DOUBLE COMMENT 'Baseline runtime (hours) per unit',
  machine_id STRING COMMENT 'Candidate machine ID for this order',
  machine_name STRING COMMENT 'Descriptive name of candidate machine',
  capability STRING COMMENT 'Machine capability (category match)',
  daily_capacity_hours DOUBLE COMMENT 'Machine’s available hours per day',
  efficiency_factor DOUBLE COMMENT 'Machine efficiency (higher = faster)',
  is_operational BOOLEAN COMMENT 'Current operational status of the machine',
  processing_hours DOUBLE COMMENT 'Estimated total hours required on this machine',
  margin DOUBLE COMMENT 'Profit margin for the order (price − cost) × quantity',
  base_confidence DOUBLE COMMENT 'Synthetic historical confidence score for this route (0–1)'
)
""")

# COMMAND ----------

# Load tables
from pyspark.sql.types import *
from pyspark.sql import functions as F, Window
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import random

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# Load machines
machines_data = [
    # --- Metal ---
    ("CNC-LATHE-A1", "CNC Lathe A1", "metal", 14.0, 1.0, True),
    ("CNC-LATHE-A2", "CNC Lathe A2", "metal", 13.0, 0.85, True),
    ("MILL-B1",       "Vertical Mill B1", "metal", 12.0, 0.95, True),
    ("MILL-B2",       "Vertical Mill B2", "metal", 11.5, 0.90, True),

    # --- Battery ---
    ("BAT-ASSEM-1", "Battery Assembly 1", "battery", 12.0, 1.0, True),
    ("BAT-ASSEM-2", "Battery Assembly 2", "battery", 11.0, 0.90, True),
    ("BAT-ASSEM-3", "Battery Assembly 3", "battery", 12.5, 0.95, True),

    # --- Electronics ---
    ("ELEC-ASSEM-1", "Electronics Assembly 1", "electronics", 10.0, 1.0, True),
    ("ELEC-ASSEM-2", "Electronics Assembly 2", "electronics", 12.0, 0.95, True),
    ("ELEC-ASSEM-3", "Electronics Assembly 3", "electronics", 12.5, 0.90, True),

    # --- Assembly ---
    ("ASSEMBLY-C1", "Final Assembly Line 1", "assembly", 14.0, 1.0, True),
    ("ASSEMBLY-C2", "Final Assembly Line 2", "assembly", 14.0, 0.95, True),
    ("ASSEMBLY-C3", "Final Assembly Line 3", "assembly", 13.5, 0.90, True),
]

machines_schema = StructType([
    StructField("machine_id", StringType(), True),
    StructField("machine_name", StringType(), True),
    StructField("capability", StringType(), True),
    StructField("daily_capacity_hours", DoubleType(), True),
    StructField("efficiency_factor", DoubleType(), True),
    StructField("is_operational", BooleanType(), True),
])

machines_df = spark.createDataFrame(machines_data, schema=machines_schema)
machines_df.write.mode("overwrite").saveAsTable(f"{CATALOG}.{SCHEMA}.machines_catalog")

# load parts 
parts_data = [
    ("GEAR-001", "Gear Housing", "metal", 50.0, 250.0, 1.2),
    ("HUB-001", "Hub Motor Core", "metal", 120.0, 500.0, 3.5),
    ("CTRL-001", "Controller Board", "electronics", 40.0, 220.0, 0.3),
    ("BAT48-001", "Battery 48V", "battery", 80.0, 400.0, 2.0),
    ("BAT60-001", "Battery 60V", "battery", 100.0, 450.0, 2.5),
    ("TORQ-001", "Torque Sensor", "electronics", 25.0, 150.0, 0.2),
    ("DRIVE-001", "Drive Belt", "assembly", 15.0, 75.0, 0.5),
    ("DISP-001", "LCD Display", "electronics", 60.0, 250.0, 0.4),
    ("FRAME-001", "Aluminum Frame", "metal", 200.0, 1000.0, 4.0),
    ("CRANK-001", "Crankset", "metal", 70.0, 350.0, 2.0)
]
parts_schema = StructType([
    StructField("part_id", StringType(), True),
    StructField("part_name", StringType(), True),
    StructField("category", StringType(), True),
    StructField("material_cost", DoubleType(), True),
    StructField("default_price", DoubleType(), True),
    StructField("estimated_runtime_hours", DoubleType(), True),
])
parts_df = spark.createDataFrame(parts_data, schema=parts_schema)
parts_df.write.mode("overwrite").saveAsTable(f"{CATALOG}.{SCHEMA}.parts_catalog")

# Customers
retail_customers = ["Alice Johnson","Bob Smith","Carol Lee","David Miller","Eva Brown","Frank Wilson","Grace Kim","Henry Adams","Irene Patel","Jack Davis"]
dealer_customers = ["Downtown Bikes LLC","Urban Riders Inc.","TrailMasters Co.","MountainGear Co.","CycleWorld Dist.","PeakMotion Ltd.","Revolution Wheels","GearUp Inc."]
customers = retail_customers + dealer_customers

# load orders
from random import choices

today = datetime(2025, 11, 1)
NUM_ORDERS = 1000        # # of orders to generate

order_rows = []
for i in range(NUM_ORDERS):
    order_id = f"ORD-{2000+i}"
    part_weights = {
        "metal": 1.0,
        "battery": 1.0,
        "electronics": 2.5,  # boost
        "assembly": 3.0       # boost
    }

    parts, weights = zip(*[(p, part_weights[p[2]]) for p in parts_data])

    part = choices(parts, weights=weights, k=1)[0]

    # part = random.choice(parts_data)
    part_id = part[0]
    category = part[2]
    # dealers more likely for electronics & battery
    if category in ("battery","electronics"):
        customer = random.choice(dealer_customers)
        quantity = random.randint(4, 6)
    else:
        customer = random.choice(customers)
        quantity = random.randint(1, 6)
    promised_date = today + timedelta(days=random.randint(1, 30))
    sale_price = part[4]
    material_cost = part[3]
    part_runtime = part[5]
    order_rows.append((order_id, part_id, category, customer, quantity, promised_date, sale_price, material_cost, part_runtime))

orders_schema = StructType([
    StructField("order_id", StringType(), True),
    StructField("part_id", StringType(), True),
    StructField("category", StringType(), True),
    StructField("customer_name", StringType(), True),
    StructField("quantity", IntegerType(), True),
    StructField("promised_date", DateType(), True),
    StructField("sale_price", DoubleType(), True),
    StructField("material_cost", DoubleType(), True),
    StructField("estimated_runtime_hours", DoubleType(), True)
])
orders_df = spark.createDataFrame(order_rows, schema=orders_schema)
orders_df.write.mode("overwrite").saveAsTable(f"{CATALOG}.{SCHEMA}.orders_backlog")

# COMMAND ----------

import pyspark.sql.functions as F
from pyspark.sql.functions import expr

# Load input tables
orders_df   = spark.table(f"{CATALOG}.{SCHEMA}.orders_backlog")
machines_df = spark.table(f"{CATALOG}.{SCHEMA}.machines_catalog")

# Join orders × machines on matching capability
candidates_df = (
    orders_df.alias("o")
    .join(
        machines_df.alias("m"),
        F.col("o.category") == F.col("m.capability"),
        "inner"
    )
    .select(
        F.col("o.order_id"),
        F.col("o.part_id"),
        F.col("o.category").alias("part_category"),
        F.col("o.customer_name"),
        F.col("o.quantity"),
        F.col("o.promised_date"),
        F.col("o.sale_price"),
        F.col("o.material_cost"),
        F.col("o.estimated_runtime_hours"),
        F.col("m.machine_id"),
        F.col("m.machine_name"),
        F.col("m.capability"),
        F.col("m.daily_capacity_hours"),
        F.col("m.efficiency_factor"),
        F.col("m.is_operational"),
    )
)

# Derived features
candidates_df = (
    candidates_df
    .withColumn("processing_hours",
                F.col("estimated_runtime_hours") * F.col("quantity") / F.col("efficiency_factor"))
    .withColumn("margin",
                (F.col("sale_price") - F.col("material_cost")) * F.col("quantity"))
    .withColumn("base_confidence",
        F.round(
            F.least(
                F.greatest(
                    F.lit(0.35)
                    + F.rand(seed=SEED) * F.lit(0.6)
                    + (F.col("efficiency_factor") - F.lit(1.0)) * F.lit(0.15),
                    F.lit(0.01)
                ),
                F.lit(0.99)
            ),
            2
        )
    )
)

# Write to Delta
candidates_df.write.mode("overwrite").saveAsTable(f"{CATALOG}.{SCHEMA}.candidate_routes")