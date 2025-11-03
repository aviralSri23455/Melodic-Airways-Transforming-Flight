"""
Interactive DuckDB Query Tool for Aero Melody
Quick queries and data exploration
"""

import duckdb
import sys
from app.core.config import settings


def run_query(query: str, db_path: str = None):
    """Execute a DuckDB query and display results"""
    db_path = db_path or settings.DUCKDB_PATH
    
    conn = duckdb.connect(db_path)
    
    try:
        result = conn.execute(query).fetchall()
        columns = [desc[0] for desc in conn.description] if conn.description else []
        
        # Print column headers
        if columns:
            print("\n" + " | ".join(columns))
            print("-" * (len(" | ".join(columns))))
        
        # Print rows
        for row in result:
            print(" | ".join(str(val) for val in row))
        
        print(f"\n✅ {len(result)} rows returned")
        
    except Exception as e:
        print(f"❌ Query error: {e}")
    finally:
        conn.close()


def quick_stats():
    """Display quick statistics"""
    db_path = settings.DUCKDB_PATH
    conn = duckdb.connect(db_path)
    
    print("\n" + "=" * 60)
    print("📊 QUICK STATISTICS")
    print("=" * 60)
    
    # List all tables
    print("\n📋 Available Tables:")
    try:
        tables = conn.execute("""
            SELECT table_schema, table_name, 
                   (SELECT COUNT(*) FROM information_schema.tables t2 
                    WHERE t2.table_schema = t1.table_schema 
                    AND t2.table_name = t1.table_name) as row_count
            FROM information_schema.tables t1
            WHERE table_schema NOT IN ('information_schema', 'pg_catalog')
            ORDER BY table_schema, table_name
        """).fetchall()
        
        for schema, table, count in tables:
            print(f"  • {schema}.{table}")
    except Exception as e:
        print(f"  ⚠️  {e}")
    
    # Database size
    try:
        import os
        if os.path.exists(db_path):
            size_mb = os.path.getsize(db_path) / (1024 * 1024)
            print(f"\n💾 Database Size: {size_mb:.2f} MB")
    except:
        pass
    
    conn.close()


def interactive_mode():
    """Interactive query mode"""
    print("\n" + "=" * 60)
    print("🦆 DUCKDB INTERACTIVE MODE")
    print("=" * 60)
    print("Enter SQL queries (type 'exit' to quit, 'stats' for quick stats)")
    print("=" * 60)
    
    db_path = settings.DUCKDB_PATH
    conn = duckdb.connect(db_path)
    
    while True:
        try:
            query = input("\nDuckDB> ").strip()
            
            if query.lower() == 'exit':
                break
            elif query.lower() == 'stats':
                quick_stats()
                continue
            elif not query:
                continue
            
            result = conn.execute(query).fetchall()
            columns = [desc[0] for desc in conn.description] if conn.description else []
            
            # Print results
            if columns:
                print("\n" + " | ".join(columns))
                print("-" * (len(" | ".join(columns))))
            
            for row in result:
                print(" | ".join(str(val) for val in row))
            
            print(f"\n✅ {len(result)} rows")
            
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")
    
    conn.close()


if __name__ == "__main__":
    if len(sys.argv) > 1:
        if sys.argv[1] == "stats":
            quick_stats()
        elif sys.argv[1] == "interactive":
            interactive_mode()
        else:
            # Run provided query
            query = " ".join(sys.argv[1:])
            run_query(query)
    else:
        # Default: show stats
        quick_stats()
        print("\n💡 Usage:")
        print("  python duckdb_query.py stats              - Show quick statistics")
        print("  python duckdb_query.py interactive         - Interactive query mode")
        print("  python duckdb_query.py 'SELECT * FROM ...' - Run a query")
