"""
Quick launcher for DuckDB Analytics
Run this from the duckdb_analytics directory or backend directory
"""

import sys
import os

# Add backend to path (go up one level if in duckdb_analytics folder)
backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, backend_dir)

# Import and run analytics
from duckdb_analytics.analytics import main

if __name__ == "__main__":
    main()
