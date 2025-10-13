import os
import time
import uuid
from databricks import sdk
from psycopg_pool import ConnectionPool
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Database connection setup
workspace_client = sdk.WorkspaceClient()
postgres_password = None
last_password_refresh = 0
connection_pool = None

user = workspace_client.current_user.me().user_name

# Schema and table configuration
SCHEMA = os.getenv('LAKEBASE_SCHEMA')
OVERRIDES_TABLE_NAME = os.getenv('LAKEBASE_OVERRIDES_TABLE_NAME')
PART_LOOKUP_TABLE_NAME = os.getenv('LAKEBASE_PART_LOOKUP_TABLE_NAME')

def refresh_oauth_token():
    """Refresh OAuth token if expired."""
    global postgres_password, last_password_refresh
    if postgres_password is None or time.time() - last_password_refresh > 900:
        print("Refreshing PostgreSQL OAuth token")
        try:
            cred = workspace_client.database.generate_database_credential(
                request_id=str(uuid.uuid4()),
                instance_names=[os.getenv('LAKEBASE_INSTANCE_NAME')]
            )
            postgres_password = cred.token
            last_password_refresh = time.time()
        except Exception as e:
            raise Exception(f"Failed to refresh token: {str(e)}")

def get_connection_pool():
    """Get or create the connection pool."""
    global connection_pool
    if connection_pool is None:
        refresh_oauth_token()
        conn_string = (
            f"dbname={os.getenv('PGDATABASE')} "
            f"user={user} "
            f"password={postgres_password} "
            f"host={os.getenv('PGHOST')} "
            f"port={os.getenv('PGPORT')} "
            f"sslmode={os.getenv('PGSSLMODE', 'require')} "
            f"application_name={os.getenv('PGAPPNAME')}"
        )
        connection_pool = ConnectionPool(conn_string, min_size=2, max_size=10)
    return connection_pool

def get_connection():
    """Get a connection from the pool."""
    global connection_pool
    
    # Recreate pool if token expired
    if postgres_password is None or time.time() - last_password_refresh > 900:
        if connection_pool:
            connection_pool.close()
            connection_pool = None
    
    return get_connection_pool().connection()

def add_override(part_id, assigned_machine_id, assigned_by, notes):
    """Add or update an assignment override."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(f"""
                INSERT INTO {SCHEMA}.{OVERRIDES_TABLE_NAME} (order_id, machine_id, assigned_by, assigned_at, notes)
                VALUES (%s, %s, %s, CURRENT_TIMESTAMP, %s)
            """, (part_id, assigned_machine_id, assigned_by, notes))
            conn.commit()

def fetch_overrides():
    """Fetch assignment overrides from Lakebase for override history in the app."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(f"""
                SELECT order_id, machine_id, assigned_by, assigned_at, notes
                FROM {SCHEMA}.{OVERRIDES_TABLE_NAME}
                ORDER BY assigned_at DESC
            """)
            return cur.fetchall()