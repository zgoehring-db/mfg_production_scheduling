import pandas as pd

# -------------------------------------------------------------------
# Greedy assignment algorithm
# -------------------------------------------------------------------
def greedy_assign_priority(candidates_df, machines_catalog_df, planning_days=21, down_machines=None):
    """
    Greedy assignment algorithm that routes orders to machines
    based on profit-weighted model scores and capacity limits.

    Parameters
    ----------
    candidates_df : pd.DataFrame
        Each row represents an order–machine candidate pair with columns:
        ['order_id', 'machine_id', 'processing_hours', 'margin', 'p_best',
         'profit_score', 'base_confidence', 'promised_date']
    machines_catalog_df : pd.DataFrame
        Machine metadata including ['machine_id', 'daily_capacity_hours'].
    planning_days : int
        Number of days to plan capacity over.
    down_machines : list[str], optional
        Machines unavailable for assignment.

    Returns
    -------
    assigned_df : pd.DataFrame
        Orders successfully assigned to machines.
    machine_caps : dict
        Remaining capacity (hours) per machine after assignment.
    """
    down_machines = set(down_machines or [])

    # Normalize inputs
    cand = candidates_df if isinstance(candidates_df, pd.DataFrame) else candidates_df.toPandas()
    mach = machines_catalog_df if isinstance(machines_catalog_df, pd.DataFrame) else machines_catalog_df.toPandas()

    # Initialize capacity map
    machine_caps = {
        m["machine_id"]: float(m["daily_capacity_hours"] * planning_days)
        for _, m in mach.iterrows()
    }
    for m in down_machines:
        machine_caps[m] = 0.0

    # Build per-order priority table
    order_priority = (
        cand.groupby("order_id")
            .agg(max_profit_score=("profit_score", "max"),
                 min_due=("promised_date", "min"),
                 min_hours=("processing_hours", "min"))
            .sort_values(by=["max_profit_score", "min_due", "min_hours"], ascending=[False, True, True])
            .reset_index()
    )

    # Pre-split candidates for efficiency
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
                    "base_confidence": float(c["base_confidence"]),
                })
                break  # move to next order
    return pd.DataFrame(assigned), machine_caps


# -------------------------------------------------------------------
# KPI computation helper
# -------------------------------------------------------------------
def compute_kpis(assigned_pd, machines_catalog_pd, planning_days=21):
    """
    Compute high-level KPIs from an assigned order–machine DataFrame.
    Replaces factory utilization with OEE (Overall Equipment Effectiveness).
    """

    if assigned_pd.empty:
        return {
            "expected_profit": 0.0,
            "expected_ontime_deliveries": 0.0,
            "factory_oee": 0.0,
            "expected_orders_completed": 0,
        }

    total_profit = assigned_pd["profit"].sum()
    expected_ontime = assigned_pd["p_best"].sum()  # proxy for on-time performance
    total_used = assigned_pd["processing_hours"].sum()
    total_capacity = machines_catalog_pd["daily_capacity_hours"].sum() * planning_days

    # --- OEE components ---
    availability = min(total_used / total_capacity, 1.0)                # machine uptime vs capacity
    performance = assigned_pd["p_best"].mean()                          # proxy for throughput efficiency
    quality = 0.98                                                      # assume 98% good parts for now
    factory_oee = availability * performance * quality

    orders_completed = assigned_pd["order_id"].nunique()

    return {
        "expected_profit": float(total_profit),
        "expected_ontime_deliveries": float(expected_ontime),
        "factory_oee": float(factory_oee),
        "expected_orders_completed": int(orders_completed),
    }