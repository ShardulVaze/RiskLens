from sqlalchemy import create_engine
from database.db_config import DATABASE_URL

try:
    engine = create_engine(DATABASE_URL)
    conn = engine.connect()

    print("Database connected successfully!")

    conn.close()

except Exception as e:
    print("Error:", e)