import os
import pandas as pd
from sqlalchemy import create_engine
from database.db_config import DATABASE_URL

engine = create_engine(DATABASE_URL)

csv_path = "data/raw/previous_application.csv"

if not os.path.exists(csv_path):
  
    exit()

df = pd.read_csv(csv_path)
df.columns = df.columns.str.upper().str.strip()
total_rows = len(df)


chunksize = 10000

inserted = 0
for i in range(0, total_rows, chunksize):
    chunk = df.iloc[i:i+chunksize]
    chunk.to_sql(
        "raw_previous_application",
        engine,
        if_exists="append",
        index=False
    )
    inserted += len(chunk)
    print(f"Inserted rows: {inserted}/{total_rows}")

print("DONE")
