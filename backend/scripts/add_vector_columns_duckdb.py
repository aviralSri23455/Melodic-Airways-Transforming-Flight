"""
Add vector embedding columns to DuckDB analytics database
"""

import duckdb
import sys
from pathlib import Path

# Add parent directory to path to import settings
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.core.config import settings


def add_vector_columns():
    """Add vector embedding columns to DuckDB tables"""
    
    db_path = getattr(settings, 'DUCKDB_PATH', './data/analytics.duckdb')
    
    print(f"Connecting to DuckDB at: {db_path}")
    
    try:
        # Create data directory if it doesn't exist
        db_dir = Path(db_path).parent
        db_dir.mkdir(parents=True, exist_ok=True)
        
        # Connect to DuckDB
        conn = duckdb.connect(database=db_path, read_only=False)
        
        print("✅ Connected to DuckDB")
        
        # Check if route_analytics table exists
        tables = conn.execute("SHOW TABLES").fetchall()
        table_names = [table[0] for table in tables]
        
        print(f"Existing tables: {table_names}")
        
        # Add embedding column to route_analytics if it doesn't exist
        if 'route_analytics' in table_names:
            # Check if column already exists
            columns = conn.execute("DESCRIBE route_analytics").fetchall()
            column_names = [col[0] for col in columns]
            
            if 'route_embedding' not in column_names:
                print("Adding route_embedding column to route_analytics...")
                conn.execute("""
                    ALTER TABLE route_analytics 
                    ADD COLUMN route_embedding VARCHAR
                """)
                print("✅ Added route_embedding column")
            else:
                print("ℹ️  route_embedding column already exists")
        else:
            print("Creating route_analytics table with embedding column...")
            conn.execute("""
                CREATE TABLE route_analytics (
                    id INTEGER PRIMARY KEY,
                    origin VARCHAR,
                    destination VARCHAR,
                    distance_km DOUBLE,
                    complexity_score DOUBLE,
                    path_length INTEGER,
                    intermediate_stops INTEGER,
                    route_embedding VARCHAR,
                    created_at TIMESTAMP
                )
            """)
            print("✅ Created route_analytics table")
        
        # Add embedding column to music_analytics if it doesn't exist
        if 'music_analytics' in table_names:
            columns = conn.execute("DESCRIBE music_analytics").fetchall()
            column_names = [col[0] for col in columns]
            
            if 'music_vector' not in column_names:
                print("Adding music_vector column to music_analytics...")
                conn.execute("""
                    ALTER TABLE music_analytics 
                    ADD COLUMN music_vector VARCHAR
                """)
                print("✅ Added music_vector column")
            else:
                print("ℹ️  music_vector column already exists")
        else:
            print("Creating music_analytics table with vector column...")
            conn.execute("""
                CREATE TABLE music_analytics (
                    id INTEGER PRIMARY KEY,
                    route_id INTEGER,
                    tempo INTEGER,
                    key VARCHAR,
                    scale VARCHAR,
                    duration_seconds DOUBLE,
                    note_count INTEGER,
                    harmony_complexity DOUBLE,
                    genre VARCHAR,
                    embedding_vector VARCHAR,
                    music_vector VARCHAR,
                    created_at TIMESTAMP
                )
            """)
            print("✅ Created music_analytics table")
        
        # Create vector embeddings table for FAISS indices
        if 'vector_embeddings' not in table_names:
            print("Creating vector_embeddings table...")
            conn.execute("""
                CREATE TABLE vector_embeddings (
                    id INTEGER PRIMARY KEY,
                    entity_type VARCHAR,
                    entity_id INTEGER,
                    embedding_vector VARCHAR,
                    metadata VARCHAR,
                    created_at TIMESTAMP,
                    updated_at TIMESTAMP
                )
            """)
            print("✅ Created vector_embeddings table")
        else:
            print("ℹ️  vector_embeddings table already exists")
        
        # Create indices for faster lookups
        print("Creating indices...")
        try:
            conn.execute("CREATE INDEX IF NOT EXISTS idx_route_analytics_origin ON route_analytics(origin)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_route_analytics_destination ON route_analytics(destination)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_music_analytics_route_id ON music_analytics(route_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_vector_embeddings_entity ON vector_embeddings(entity_type, entity_id)")
            print("✅ Created indices")
        except Exception as e:
            print(f"⚠️  Warning: Could not create some indices: {e}")
        
        conn.close()
        
        print("\n" + "="*50)
        print("✅ Vector columns added successfully!")
        print("="*50)
        
        return True
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = add_vector_columns()
    sys.exit(0 if success else 1)
