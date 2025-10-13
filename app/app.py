import os
from databricks import sql
from databricks.sdk.core import Config
from databricks.sdk import WorkspaceClient
import streamlit as st
import pandas as pd
from utils.routing_helpers import greedy_assign_priority, compute_kpis
from utils.lakebase_utils import add_override, fetch_overrides
from dotenv import load_dotenv

# -------------------------------------------------
# Page Configuration (must be first Streamlit command)
# -------------------------------------------------
st.set_page_config(
    page_title="SmartFab Production Scheduling",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded"
)

load_dotenv()
# -------------------------------------------------
# Environment setup
# -------------------------------------------------
CATALOG = os.getenv("CATALOG")
SCHEMA = os.getenv("SCHEMA")

# Ensure warehouse environment variable is defined
assert os.getenv("DATABRICKS_WAREHOUSE_ID"), "DATABRICKS_WAREHOUSE_ID must be set in app.yaml."

# -------------------------------------------------
# Databricks SQL connection helper
# -------------------------------------------------
cfg = Config()
workspace_client = WorkspaceClient()
current_user = workspace_client.current_user.me().user_name

def sqlQuery(query: str) -> pd.DataFrame:
    """Execute a SQL query using service principal authentication"""
    with sql.connect(
        server_hostname=cfg.host,
        http_path=f"/sql/1.0/warehouses/{os.getenv('DATABRICKS_WAREHOUSE_ID')}",
        credentials_provider=lambda: cfg.authenticate
    ) as connection:
        with connection.cursor() as cursor:
            cursor.execute(query)
            return cursor.fetchall_arrow().to_pandas()

# -------------------------------------------------
# Data Load
# -------------------------------------------------
@st.cache_data(ttl=60)
def load_data():
    """Load data from Databricks"""
    machines = sqlQuery(f"SELECT * FROM {CATALOG}.{SCHEMA}.machines_catalog")
    candidates = sqlQuery(f"SELECT * FROM {CATALOG}.{SCHEMA}.candidate_routes_scored")
    
    # Join assigned_baseline with orders_backlog to get enriched order details
    assigned_baseline = sqlQuery(f"""
        SELECT 
            a.order_id,
            a.machine_id,
            a.profit,
            a.p_best,
            a.processing_hours,
            o.part_id,
            o.category,
            o.customer_name,
            o.quantity
        FROM {CATALOG}.{SCHEMA}.assigned_baseline a
        LEFT JOIN {CATALOG}.{SCHEMA}.orders_backlog o
            ON a.order_id = o.order_id
    """)
    
    # Load orders_backlog separately for use in recalculation joins
    orders_backlog = sqlQuery(f"""
        SELECT 
            order_id, 
            part_id, 
            category, 
            customer_name, 
            quantity 
        FROM {CATALOG}.{SCHEMA}.orders_backlog
    """)
    
    return machines, candidates, assigned_baseline, orders_backlog

machines_df, cand_df, assigned_baseline_df, orders_backlog_df = load_data()

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
        /* --- Full width layout for Databricks Apps --- */
        .stApp {
            max-width: 100% !important;
        }
        
        .stApp > div {
            max-width: 100% !important;
        }
        
        .main {
            max-width: 100% !important;
        }
        
        .main .block-container {
            max-width: 100% !important;
            padding-left: 1rem !important;
            padding-right: 1rem !important;
            padding-top: 1rem !important;
            padding-bottom: 1rem !important;
        }
        
        .main .block-container > div {
            max-width: 100% !important;
        }
        
        /* --- GLOBAL PAGE STYLE --- */
        [data-testid="stAppViewContainer"] {
            background-color: #f5f7fb;
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
        
        /* --- BUTTON STYLING --- */
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
st.sidebar.header("Machine Downtime Simulator")
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
    
    # Enrich the recalculated assignments with order details (same join pattern as baseline)
    assigned_scenario_df = assigned_scenario_df.merge(
        orders_backlog_df, 
        on='order_id', 
        how='left'
    )
    
    kpi_scenario = compute_kpis(assigned_scenario_df, machines_df)
else:
    assigned_scenario_df, kpi_scenario = assigned_baseline_df, kpi_baseline

st.sidebar.markdown("---")
st.sidebar.subheader("Manual Override")

# Initialize session state for form reset
if 'override_form_key' not in st.session_state:
    st.session_state.override_form_key = 0

# Dropdown for order selection
order_selected = st.sidebar.selectbox(
    "Select Order to Override:",
    assigned_scenario_df["order_id"].unique(),
    index=None,
    placeholder="Choose an order...",
    key=f"order_select_{st.session_state.override_form_key}"
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
        placeholder="Choose new machine...",
        key=f"machine_select_{st.session_state.override_form_key}"
    )

    # Optional notes field
    notes = st.sidebar.text_input(
        "Notes (optional):", 
        placeholder="Reason for override...",
        key=f"notes_input_{st.session_state.override_form_key}"
    )
    
    if st.sidebar.button("💾 Save Override"):
        if new_machine and new_machine != current_machine:
            try:
                # Save override using Lakebase
                add_override(
                    part_id=order_selected,
                    assigned_machine_id=new_machine,
                    assigned_by=current_user,
                    notes=notes if notes else None
                )
                st.sidebar.success(f"✅ Override saved for {order_selected} → {new_machine}")
                # Increment form key to reset all widgets
                st.session_state.override_form_key += 1
                # Wait a moment for user to see the success message
                import time
                time.sleep(1.5)
                # Refresh the page to reset the form
                st.rerun()
            except Exception as e:
                st.sidebar.error(f"❌ Failed to save override: {str(e)}")
        else:
            st.sidebar.warning("⚠️ Please choose a new machine before saving.")

# -------------------------------------------------
# KPI Cards
# -------------------------------------------------
st.markdown("### 📊 Factory Performance Summary")
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
    
    # Only show delta if routes were recalculated
    if recalc:
        delta_html = f'<div class="kpi-delta" style="color:{color};">Δ {delta:+.2f}</div>'
    else:
        delta_html = ''
    
    cols[i].markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-label">{label}</div>
            <div class="kpi-value">{scenario_vals[i]:,.2f}</div>
            {delta_html}
        </div>
        """,
        unsafe_allow_html=True,
    )

# -------------------------------------------------
# Assigned Orders Table
# -------------------------------------------------
st.markdown("")  # Add spacing
st.markdown("### 🧾 Assigned Orders")
# Display only user-friendly columns (hide calculation-only columns)
display_columns = ['order_id', 'machine_id', 'part_id', 'category', 'customer_name', 'quantity']
st.dataframe(assigned_scenario_df[display_columns], use_container_width=True)

# -------------------------------------------------
# Override History Table
# -------------------------------------------------
st.markdown("### 📝 Override History")

try:
    overrides_data = fetch_overrides()
    if overrides_data:
        # Convert to DataFrame with proper column names
        overrides_df = pd.DataFrame(
            overrides_data,
            columns=["Order ID", "Machine ID", "Assigned By", "Assigned At", "Notes"]
        )
        st.dataframe(overrides_df, use_container_width=True)
    else:
        st.info("No overrides recorded yet.")
except Exception as e:
    st.warning(f"Unable to load override history: {str(e)}")
# -------------------------------------------------

