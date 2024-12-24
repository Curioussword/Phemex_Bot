import psycopg2

def connect_to_db():
    """Connect to the TimescaleDB database."""
    try:
        conn = psycopg2.connect(
            dbname="trading_data",
            user="postgres",
            password="xxxx",
            host="xxxxxxxx",  # Change this if using a remote server
            port="xxxx"
        )
        return conn
    except Exception as e:
        print(f"[ERROR] Failed to connect to the database: {e}")
        return None
