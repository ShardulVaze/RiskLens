import pandas as pd
from sqlalchemy import create_engine
from database.db_config import DATABASE_URL


csv_path = "data/raw/creditcard.csv"
df = pd.read_csv(csv_path)


df.columns = df.columns.str.lower()

engine = create_engine(DATABASE_URL)

df.to_sql(
    name="raw_transactions",
    con=engine,
    if_exists="append",
    index=False
)

print("Raw data successfully ingested into PostgreSQL!")
