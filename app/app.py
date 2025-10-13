import os
from databricks import sql
from databricks.sdk.core import Config
from databricks.sdk import WorkspaceClient
import streamlit as st
import pandas as pd
from utils.routing_helpers import greedy_assign_priority, compute_kpis
from dotenv import load_dotenv

load_dotenv()
# -------------------------------------------------
# Environment setup
# -------------------------------------------------
CATALOG = os.getenv("CATALOG")
SCHEMA = os.getenv("SCHEMA")

# Ensure warehouse environment variable is defined
assert os.getenv("DATABRICKS_WAREHOUSE_ID"), "DATABRICKS_WAREHOUSE_ID must be set in app.yaml."

# -------------------------------------------------
# Databricks SQL connection helper (following cookbook pattern)
# -------------------------------------------------
cfg = Config()

def get_user_token():
    """Get OBO token from Streamlit context headers"""
    try:
        headers = st.context.headers
        user_token = headers.get("X-Forwarded-Access-Token")
        return user_token
    except Exception:
        return None

def get_sql_connection(user_token=None):
    """Create SQL connection with OBO or service principal auth"""
    st.write("🔍 DEBUG Connection Parameters:")
    st.write(f"  - server_hostname: {cfg.host}")
    st.write(f"  - http_path: /sql/1.0/warehouses/{os.getenv('DATABRICKS_WAREHOUSE_ID')}")
    st.write(f"  - has user_token: {bool(user_token)}")
    
    if user_token:
        st.write(f"  - token preview: {user_token[:50]}...")
    
    connect_kwargs = {
        "server_hostname": cfg.host,
        "http_path": f"/sql/1.0/warehouses/{os.getenv('DATABRICKS_WAREHOUSE_ID')}"
    }
    
    if user_token:
        connect_kwargs["access_token"] = user_token
    else:
        connect_kwargs["credentials_provider"] = lambda: cfg.authenticate
    
    try:
        st.write("🔄 Attempting connection...")
        conn = sql.connect(**connect_kwargs)
        st.success("✅ Connection successful!")
        return conn
    except Exception as e:
        st.error(f"❌ Connection failed!")
        st.error(f"Error type: {type(e).__name__}")
        st.error(f"Error message: {str(e)}")
        import traceback
        st.code(traceback.format_exc())
        raise

def sqlQuery(query: str, user_token=None) -> pd.DataFrame:
    """Execute a SQL query"""
    with get_sql_connection(user_token) as connection:
        with connection.cursor() as cursor:
            cursor.execute(query)
            return cursor.fetchall_arrow().to_pandas()

# Get user token for OBO authentication (at module level)
user_token = get_user_token()
if user_token:
    st.sidebar.success(f"✅ OBO Token Retrieved: {user_token[:20]}...")
else:
    st.sidebar.warning("⚠️ No OBO token - using service principal")

# -------------------------------------------------
# Data Load
# -------------------------------------------------
def load_data():
    """Load data from Databricks."""
    machines = sqlQuery(f"SELECT * FROM {CATALOG}.{SCHEMA}.machines_catalog", user_token)
    candidates = sqlQuery(f"SELECT * FROM {CATALOG}.{SCHEMA}.candidate_routes_scored", user_token)
    assigned_baseline = sqlQuery(f"SELECT * FROM {CATALOG}.{SCHEMA}.assigned_baseline", user_token)
    return machines, candidates, assigned_baseline

machines_df, cand_df, assigned_baseline_df = load_data()

# Compute baseline KPIs
kpi_baseline = compute_kpis(assigned_baseline_df, machines_df)
# -------------------------------------------------
# App layout
# -------------------------------------------------
# -------------------------------------------------
# Custom styles
# -------------------------------------------------
st.markdown("""
    <style>
        /* --- Balanced full-width layout --- */
        [data-testid="stAppViewContainer"] > .main {
            max-width: 90% !important;     /* ✅ gives ~5% margin on each side */
            margin: 0 auto !important;     /* centers the container */
            padding-left: 1.5rem !important;
            padding-right: 1.5rem !important;
        }

        /* --- Inner block adjustments for tables and KPI cards --- */
        .block-container {
            padding-left: 0 !important;
            padding-right: 0 !important;
            max-width: 90% !important;
        }

    
        /* --- GLOBAL PAGE STYLE --- */
        [data-testid="stAppViewContainer"] {
            background-color: #f5f7fb;
            # background-image: linear-gradient(180deg, #f8f9fb 0%, #eef2f6 100%);
        }

        [data-testid="stSidebar"] {
            background-color: #e9edf2 !important; /* slightly darker contrast to main background */
            border-right: 1px solid #d0d4da !important;
        }

        /* --- KPI CARDS --- */
        .kpi-card {
            background: linear-gradient(135deg, #ffffff 0%, #f3f6f9 100%);
            padding: 20px;
            border-radius: 14px;
            box-shadow: 0 3px 8px rgba(0,0,0,0.08);
            text-align: center;
            font-family: 'Inter', sans-serif;
        }
        .kpi-value {
            font-size: 1.8rem;
            font-weight: 600;
            color: #222;
        }
        .kpi-delta {
            font-size: 0.9rem;
            font-weight: 500;
        }
        .kpi-label {
            color: #666;
            font-size: 0.95rem;
        }
        /* --- SIDEBAR LOGO --- */
        [data-testid="stSidebar"] img {
            margin-top: 15px;
            border-radius: 10px;
        }
        /* --- SmartFab primary button --- */
        /* --- SmartFab primary button (deeper logo blue) --- */
        /* --- SmartFab primary button (flat warm orange) --- */
        div.stButton > button:first-child {
            background-color: #F59E0B !important;  /* warm amber-orange */
            color: white !important;
            border: none !important;
            border-radius: 8px !important;
            padding: 0.6rem 1.2rem !important;
            font-weight: 600 !important;
            box-shadow: 0px 3px 8px rgba(0, 0, 0, 0.15);
            transition: all 0.25s ease-in-out;
        }
        div.stButton > button:first-child:hover {
            transform: translateY(-2px);
            box-shadow: 0px 5px 12px rgba(0, 0, 0, 0.25);
            background-color: #EA580C !important;  /* darker on hover */
        }





    </style>
""", unsafe_allow_html=True)

# Sidebar logo + controls
st.sidebar.image("assets/SmartFab_logo_cropped.png", use_container_width=True)
st.sidebar.header("Scenario Controls")
down_machines = st.sidebar.multiselect(
    "Select machines that are down:",
    machines_df["machine_id"].tolist(),
)
recalc = st.sidebar.button("🔁 Recalculate Routes")

# Recalculate scenario
if recalc:
    assigned_scenario_df, caps_scenario = greedy_assign_priority(
        cand_df, machines_df, planning_days=21, down_machines=down_machines
    )
    kpi_scenario = compute_kpis(assigned_scenario_df, machines_df)
else:
    assigned_scenario_df, kpi_scenario = assigned_baseline_df, kpi_baseline

st.sidebar.markdown("---")
st.sidebar.subheader("Manual Override")

# Dropdown for order selection
order_selected = st.sidebar.selectbox(
    "Select Order to Override:",
    assigned_scenario_df["order_id"].unique(),
    index=None,
    placeholder="Choose an order..."
)

if order_selected:
    # Show machine options based on capability (optional)
    current_machine = assigned_scenario_df.loc[
        assigned_scenario_df["order_id"] == order_selected, "machine_id"
    ].iloc[0]
    st.sidebar.write(f"Current Machine: **{current_machine}**")

    new_machine = st.sidebar.selectbox(
        "Select New Machine:",
        machines_df["machine_id"].tolist(),
        index=None,
        placeholder="Choose new machine..."
    )

    if st.sidebar.button("💾 Save Override"):
        if new_machine and new_machine != current_machine:
            override_row = assigned_scenario_df[
                assigned_scenario_df["order_id"] == order_selected
            ].copy()
            override_row["machine_id"] = new_machine
            override_row["override_timestamp"] = pd.Timestamp.now()

            # Write to UC table
            with sql.connect(
                server_hostname=Config().host,
                http_path=f"/sql/1.0/warehouses/{os.getenv('DATABRICKS_WAREHOUSE_ID')}",
                credentials_provider=lambda: Config().authenticate
            ) as conn:
                override_row.to_sql(
                    name=f"{CATALOG}.{SCHEMA}.assigned_overrides",
                    con=conn._conn,  # reuse databricks connection
                    if_exists="append",
                    index=False,
                )
            st.sidebar.success(f"Override saved for {order_selected} → {new_machine}")
        else:
            st.sidebar.warning("Please choose a new machine before saving.")

# -------------------------------------------------
# KPI Cards
# -------------------------------------------------
st.markdown("### 📊 Key Performance Indicators")
cols = st.columns(4)
# -------------------------------------------------
# KPI Labels & Values
# -------------------------------------------------
kpi_labels = [
    "Expected Profit ($)",
    "On-Time Deliveries",
    "Factory Utilization",
    "Orders Completed"
]

baseline_vals = [
    kpi_baseline["expected_profit"],
    kpi_baseline["expected_ontime_deliveries"],
    kpi_baseline["factory_capacity_utilization"],
    kpi_baseline["expected_orders_completed"],
]

scenario_vals = [
    kpi_scenario["expected_profit"],
    kpi_scenario["expected_ontime_deliveries"],
    kpi_scenario["factory_capacity_utilization"],
    kpi_scenario["expected_orders_completed"],
]

for i, label in enumerate(kpi_labels):
    delta = scenario_vals[i] - baseline_vals[i]
    color = "green" if delta > 0 else "red" if delta < 0 else "#555"
    cols[i].markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-label">{label}</div>
            <div class="kpi-value">{scenario_vals[i]:,.2f}</div>
            <div class="kpi-delta" style="color:{color};">
                Δ {delta:+.2f}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# -------------------------------------------------
# Assigned Orders Table
# -------------------------------------------------
st.markdown("")  # Add spacing
st.markdown("### 🧾 Assigned Orders")
st.dataframe(assigned_scenario_df, use_container_width=True)
# -------------------------------------------------

