import pandas as pd
import numpy as np
import psycopg2
from scipy.spatial.distance import jensenshannon

DB_CONFIG = {
    'host': 'localhost',
    'port': 5432,
    'dbname': 'Mydatabase',
    'user': 'postgres',
    'password': 'sakthi'
}

def load_table(table_name):
    conn = psycopg2.connect(**DB_CONFIG)
    query = f"SELECT * FROM {table_name} ORDER BY timestamp"
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df

def js_divergence(p, q):
    p = p / np.sum(p)
    q = q / np.sum(q)
    return float(jensenshannon(p, q, base=2) ** 2)
# import pandas as pd
# import numpy as np
# import psycopg2
# from scipy.spatial.distance import jensenshannon

# DB_CONFIG = {
#     'host': 'localhost',
#     'port': 5432,
#     'dbname': 'Mydatabase',
#     'user': 'postgres',
#     'password': 'sakthi'
# }

# def load_table(schema: str, table_name: str, start_date=None, end_date=None):
#     conn = psycopg2.connect(**DB_CONFIG)
#     cursor = conn.cursor()

#     # Build SQL query with optional date filtering
#     query = f"""
#     SELECT *
#     FROM {schema}.{table_name}
#     """

#     if start_date and end_date:
#         query += " WHERE to_timestamp(timestamp / 1000) BETWEEN %s AND %s"

#     query += " ORDER BY timestamp"

#     if start_date and end_date:
#         df = pd.read_sql_query(query, conn, params=(start_date, end_date))
#     else:
#         df = pd.read_sql_query(query, conn)

#     conn.close()
#     return df

# def js_divergence(p, q):
#     p = p / np.sum(p)
#     q = q / np.sum(q)
#     return float(jensenshannon(p, q, base=2) ** 2)
