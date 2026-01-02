from urllib.parse import quote_plus

DB_USER = "postgres"
DB_PASSWORD = quote_plus("Shrvaz@123")
DB_HOST = "localhost"
DB_PORT = "5432"
DB_NAME = "fraud_db"  

DATABASE_URL = f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
