from sqlalchemy import create_engine, text
from database.db_config import DATABASE_URL

def test_connection():
    engine = create_engine(DATABASE_URL)
    with engine.connect() as conn:
        result = conn.execute(text("SELECT 1")).scalar()
        assert result == 1, "DB connection test failed"
    print("test_connection PASSED")

def test_view_exists():
    engine = create_engine(DATABASE_URL)
    with engine.connect() as conn:
        result = conn.execute(text(
            "SELECT COUNT(*) FROM vw_risk_model_dataset LIMIT 1"
        )).scalar()
        assert result > 0, "View is empty or doesn't exist"
    print(f"test_view_exists PASSED — {result} rows found")

if __name__ == "__main__":
    test_connection()
    test_view_exists()