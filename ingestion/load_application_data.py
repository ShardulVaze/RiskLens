import pandas as pd
from sqlalchemy import create_engine
from database.db_config import DATABASE_URL

engine = create_engine(DATABASE_URL)

df = pd.read_csv("data/raw/application_data.csv")

print(f" {len(df)} rows")
df.columns = df.columns.str.upper().str.strip()

df.to_sql(
    "raw_application",
    engine,
    if_exists="append",
    index=False,
    chunksize=5000  
)

print("DONE")
