import pandas as pd
import sqlite3
import os

#Import Bracket Data Set
base_dir = os.path.dirname(os.path.abspath(__file__))
db_path = os.path.join(base_dir, 'data', 'bracket.db')
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

cursor.executescript("""
CREATE TABLE IF NOT EXISTS team_stats (id INTEGER PRIMARY KEY AUTOINCREMENT, team TEXT, seed INTEGER, current_round INTEGER,
    opponent_seed INTEGER, srs_basic REAL, sos_basic REAL, fg INTEGER, fga INTEGER, fg_pct REAL,
    three_p INTEGER, three_pa INTEGER, three_p_pct REAL, ft INTEGER, fta INTEGER, ft_pct REAL,
    orb INTEGER, trb INTEGER, ast INTEGER, stl INTEGER, blk INTEGER, tov INTEGER, pf INTEGER, pace REAL, ortg_x REAL,
    ftr REAL, three_par REAL, ts_pct REAL, trb_pct REAL, ast_pct REAL, stl_pct REAL, blk_pct REAL, efg_pct REAL,
    tov_pct REAL, orb_pct REAL, ft_per_fga REAL, rank INTEGER, net_rtg REAL, ortg_y REAL, adj_t REAL, luck REAL, sos_net_rtg REAL,
    sos_ortg REAL, ncsos_net_rtg REAL, current_ap_pre INTEGER, current_wl_pct REAL, co_wl_pct REAL, co_ncaa INTEGER,
    co_s16 INTEGER, co_ff INTEGER, co_chmp INTEGER, ortg REAL, drtg REAL, sos_drtg REAL, rf_prob REAL, combined_prob REAL,
    draft_pick_count INTEGER, total_pick_score REAL, avg_pick_number REAL, inverse_pick_sum REAL, ranked_player_count INTEGER,
    total_rank_score REAL, avg_rank_score REAL, inverse_rank_sum REAL, avg_plus_minus REAL, avg_bayesian_pr REAL,
    net_eff_dev REAL, starting5_bpr REAL, starting5_dbpr REAL, starting5_obpr REAL, starting5_plus_minus REAL,
    bench_bpr REAL, bench_dbpr REAL, bench_obpr REAL, bench_plus_minus REAL, pass_score REAL, def_sync_score REAL,
    bpr_spread REAL, scoring_cohesion REAL, teamwork_index REAL);
""")
# Commit and close
conn.commit()

# Load CSV into DataFrame
df = pd.read_csv(r"C:\Users\Colt Meyer\OneDrive\Documents\Academics\MSDS\DTSC 691 - Capstone Project\Machine Learning Project\Data Sets\Bracket Build Data Sets\test_bracket_data_set.csv")

# Save DataFrame to SQLite
df.to_sql('bracket_data', conn, if_exists='replace', index=False)
conn.close()

# Testing confirm commit
#conn = sqlite3.connect(db_path)
#df_check = pd.read_sql("SELECT * FROM bracket_data LIMIT 5", conn)
#print(df_check)
#conn.close()

#Import Linear Data Set
base_dir = os.path.dirname(os.path.abspath(__file__))
db_path = os.path.join(base_dir, 'data', 'bracket.db')

# Connect and query
conn = sqlite3.connect(db_path)

# Check actual column names
#df_all = pd.read_sql("SELECT * FROM bracket_data LIMIT 1", conn)
#print(df_all.columns)

lookup_cols = [
    'Team', 'Seed', 'Opponent Seed', 'Current Round', 'NetRtg', 'ORtg_y',
    'Bench_BPR', 'DRtg', 'Bench_OBPR', 'SoS_NetRtg', 'eFG%',
    'Starting5_BPR', 'FGA', 'SoS_ORtg', '3PA', 'TS%', 'ORtg_x'
]

# Quote columns with special characters
quoted_cols = [f'"{col}"' for col in lookup_cols]
query = f"SELECT {', '.join(quoted_cols)} FROM bracket_data"

# Execute and store in new DataFrame
linear = pd.read_sql(query, conn)

# Define path for new database
linear_db_path = os.path.join(base_dir, 'data', 'linear.db')

# Connect to new database
linear_conn = sqlite3.connect(linear_db_path)

# Save DataFrame to new database
linear.to_sql('linear_data', linear_conn, if_exists='replace', index=False)

linear_conn.close()

#conn = sqlite3.connect(linear_db_path)
#df_check = pd.read_sql("SELECT * FROM linear_data LIMIT 5", conn)
#print(df_check.head())
conn.close()

# Import XGBoost Data Set
base_dir = os.path.dirname(os.path.abspath(__file__))
db_path = os.path.join(base_dir, 'data', 'bracket.db')
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

cursor.executescript("""
CREATE TABLE IF NOT EXISTS XGBoost_data (id INTEGER PRIMARY KEY AUTOINCREMENT, srs_basic REAL, sos_basic REAL, fg INTEGER,
    fga INTEGER, three_p_pct REAL, orb INTEGER, trb INTEGER, ast INTEGER, stl INTEGER, blk INTEGER, tov INTEGER, pf INTEGER,
    ortg_x REAL, three_par REAL, ts_pct REAL, efg_pct REAL, tov_pct REAL, net_rtg REAL, ortg_y REAL, sos_net_rtg REAL,
    draft_pick_count INTEGER, total_rank_score REAL, avg_rank_score REAL, inverse_rank_sum REAL, starting5_bpr REAL,
    bench_bpr REAL, bench_obpr REAL, bench_plus_minus REAL, team TEXT, seed INTEGER, opponent_seed INTEGER);
""")

# Load CSV into DataFrame
df = pd.read_csv(r"C:\Users\Colt Meyer\OneDrive\Documents\Academics\MSDS\DTSC 691 - Capstone Project\Machine Learning Project\Data Sets\Bracket Build Data Sets\test_XGBoost_features_ds.csv")

# Save DataFrame to SQLite
df.to_sql('XGBoost', conn, if_exists='replace', index=False)

# Execute and store in new DataFrame
XGBoost = pd.read_sql(query, conn)

# Define path for new database
XGBoost_db_path = os.path.join(base_dir, 'data', 'XGBoost.db')

# Connect to new database
XGBoost_conn = sqlite3.connect(XGBoost_db_path)

# Save DataFrame to new database
XGBoost.to_sql('XGBoost_data', XGBoost_conn, if_exists='replace', index=False)

XGBoost_conn.close()

conn = sqlite3.connect(XGBoost_db_path)
df_check = pd.read_sql("SELECT * FROM XGBoost_data LIMIT 5", conn)
print(df_check.head())
conn.close()
