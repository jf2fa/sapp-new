import os
import pandas as pd
from sqlalchemy import create_engine
import psycopg2

# Database configuration
DB_NAME = "testdb"
DB_USER = "testuser"
DB_PASSWORD = "password"
DB_HOST = "localhost"
DB_PORT = "5432"
SUPERUSER_PASSWORD = "cheese"  # Superuser password for 'postgres'

# Connection string for creating the database
db_connection_string_admin = f"postgresql://postgres:{SUPERUSER_PASSWORD}@{DB_HOST}:{DB_PORT}/postgres"

# Connection string for connecting to the created database
db_connection_string = f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

# SQLAlchemy engine
engine = create_engine(db_connection_string)

# List of CSV files to import
csv_files = [
    "synthetic_survey_responses_1.csv",
    "synthetic_survey_responses_2.csv",
    "synthetic_survey_responses_3.csv",
    "synthetic_survey_responses.csv"
]

def create_database():
    """Create the database and user if they don't exist."""
    try:
        # Connect to the default 'postgres' database as the 'postgres' superuser
        conn = psycopg2.connect(db_connection_string_admin)
        conn.autocommit = True
        cursor = conn.cursor()
        
        # Create the database
        cursor.execute(f"CREATE DATABASE {DB_NAME};")
        
        # Create the user
        cursor.execute(f"CREATE USER {DB_USER} WITH PASSWORD '{DB_PASSWORD}';")
        
        # Grant privileges
        cursor.execute(f"GRANT ALL PRIVILEGES ON DATABASE {DB_NAME} TO {DB_USER};")
        
        # Grant all privileges on the public schema to the user
        cursor.execute(f"GRANT ALL PRIVILEGES ON SCHEMA public TO {DB_USER};")
        cursor.execute(f"GRANT USAGE ON SCHEMA public TO {DB_USER};")
        
        # Close the connection
        cursor.close()
        conn.close()
        
        print("Database and user created successfully.")
    except psycopg2.errors.DuplicateDatabase:
        print(f"Database {DB_NAME} already exists.")
    except psycopg2.errors.DuplicateObject:
        print(f"User {DB_USER} already exists.")
    except Exception as e:
        print(f"Error creating database or user: {e}")

def import_csv_to_db(csv_folder="."):
    """Import CSV files into the database."""
    for csv_file in csv_files:
        table_name = os.path.splitext(csv_file)[0]
        try:
            df = pd.read_csv(os.path.join(csv_folder, csv_file))
            df.to_sql(table_name, engine, if_exists='replace', index=False)
            print(f"Imported {csv_file} into {table_name} table.")
        except Exception as e:
            print(f"Error importing {csv_file}: {e}")

if __name__ == "__main__":
    # Step 1: Create the database and user (run only once)
    create_database()

    # Step 2: Import CSV files into the database
    import_csv_to_db()
