#!/usr/bin/env python3
"""
Setup script to create the sky_music database and tables for Aero Melody
"""

import pymysql
import os
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def create_database():
    """Create the sky_music database if it doesn't exist"""
    try:
        # Connect to MySQL server (without specifying database)
        conn = pymysql.connect(
            host="localhost",
            user="root",
            password=os.getenv("DB_PASSWORD", "your_password_here"),
            port=3306
        )

        cursor = conn.cursor()

        # Create database if it doesn't exist
        cursor.execute("CREATE DATABASE IF NOT EXISTS sky_music")
        print("✅ Database 'sky_music' created or already exists")

        # Use the database
        cursor.execute("USE sky_music")

        # Check if tables already exist
        cursor.execute("SHOW TABLES")
        existing_tables = [table[0] for table in cursor.fetchall()]

        if existing_tables:
            print(f"⚠️  Tables already exist: {', '.join(existing_tables)}")
            print("Skipping table creation. Drop tables first if you want to recreate them.")
        else:
            print("📋 Creating tables...")

            # Read and execute the SQL file
            sql_file_path = os.path.join(os.path.dirname(__file__), "..", "sql", "create_tables.sql")

            if os.path.exists(sql_file_path):
                with open(sql_file_path, 'r') as f:
                    sql_content = f.read()

                # Split by semicolon and execute each statement
                statements = [stmt.strip() for stmt in sql_content.split(';') if stmt.strip() and not stmt.strip().startswith('--')]

                for statement in statements:
                    if statement.strip():
                        cursor.execute(statement)

                print("✅ Tables created successfully!")
            else:
                print(f"❌ Error: SQL file not found at {sql_file_path}")

        conn.commit()
        cursor.close()
        conn.close()

        return True

    except pymysql.Error as e:
        print(f"❌ MySQL Error: {e}")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    print("🔧 Setting up Aero Melody database...")
    print("=" * 50)

    if create_database():
        print("\n🎉 Database setup completed successfully!")
        print("You can now run the FastAPI server.")
    else:
        print("\n❌ Database setup failed!")
        print("Please check your MySQL installation and credentials.")
        sys.exit(1)
