import psycopg2

def connect_to_db():
    """Connect to the TimescaleDB database."""
    try:
        conn = psycopg2.connect(
            dbname="trading_data",
            user="postgres",
            password="Kaye0315",
            host="172.18.0.2",  # Change this if using a remote server
            port="5432"
        )
        return conn
    except Exception as e:
        print(f"[ERROR] Failed to connect to the database: {e}")
        return None
