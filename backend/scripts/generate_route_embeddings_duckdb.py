"""
Generate vector embeddings for routes in DuckDB
Uses sentence transformers to create semantic embeddings
"""

import duckdb
import sys
import json
import numpy as np
from pathlib import Path
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.core.config import settings

try:
    import torch
    from sentence_transformers import SentenceTransformer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("⚠️  Warning: sentence-transformers not available, using simple embeddings")


def generate_simple_embedding(origin: str, destination: str, distance: float) -> list:
    """Generate a simple embedding based on route characteristics"""
    # Simple hash-based embedding for demonstration
    origin_hash = sum(ord(c) for c in origin) / 1000.0
    dest_hash = sum(ord(c) for c in destination) / 1000.0
    distance_norm = min(distance / 10000.0, 1.0)  # Normalize distance
    
    # Create a 128-dimensional embedding
    embedding = []
    for i in range(128):
        val = (origin_hash * (i + 1) + dest_hash * (i + 2) + distance_norm * (i + 3)) % 1.0
        embedding.append(float(val))
    
    return embedding


def generate_transformer_embedding(model, origin: str, destination: str, distance: float) -> list:
    """Generate embedding using sentence transformer"""
    # Create a text representation of the route
    route_text = f"Flight from {origin} to {destination}, distance {distance:.0f} km"
    
    # Generate embedding
    embedding = model.encode(route_text, convert_to_numpy=True)
    
    return embedding.tolist()


def generate_embeddings():
    """Generate embeddings for all routes in DuckDB"""
    
    db_path = getattr(settings, 'DUCKDB_PATH', './data/analytics.duckdb')
    
    print(f"Connecting to DuckDB at: {db_path}")
    
    try:
        # Connect to DuckDB
        conn = duckdb.connect(database=db_path, read_only=False)
        
        print("✅ Connected to DuckDB")
        
        # Load transformer model if available
        model = None
        if TRANSFORMERS_AVAILABLE:
            try:
                print("Loading sentence transformer model...")
                model = SentenceTransformer('all-MiniLM-L6-v2')
                print("✅ Model loaded")
            except Exception as e:
                print(f"⚠️  Could not load transformer model: {e}")
                print("Using simple embeddings instead")
        
        # Get all routes without embeddings
        print("\nFetching routes from database...")
        routes = conn.execute("""
            SELECT id, origin, destination, distance_km
            FROM route_analytics
            WHERE route_embedding IS NULL OR route_embedding = ''
        """).fetchall()
        
        if not routes:
            print("ℹ️  No routes found without embeddings")
            print("Checking total routes...")
            total = conn.execute("SELECT COUNT(*) FROM route_analytics").fetchone()[0]
            print(f"Total routes in database: {total}")
            
            if total == 0:
                print("\n⚠️  No routes in database. Creating sample routes...")
                # Create sample routes
                sample_routes = [
                    ('JFK', 'LAX', 3983.0),
                    ('LHR', 'CDG', 344.0),
                    ('SIN', 'HKG', 2590.0),
                    ('DXB', 'LHR', 5476.0),
                    ('SYD', 'MEL', 713.0),
                ]
                
                for origin, dest, distance in sample_routes:
                    conn.execute("""
                        INSERT INTO route_analytics 
                        (origin, destination, distance_km, complexity_score, path_length, intermediate_stops, created_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                    """, [origin, dest, distance, 0.5, 1, 0, datetime.now()])
                
                print(f"✅ Created {len(sample_routes)} sample routes")
                
                # Fetch the newly created routes
                routes = conn.execute("""
                    SELECT id, origin, destination, distance_km
                    FROM route_analytics
                    WHERE route_embedding IS NULL OR route_embedding = ''
                """).fetchall()
        
        print(f"Found {len(routes)} routes to process")
        
        # Generate embeddings
        processed = 0
        for route_id, origin, destination, distance in routes:
            try:
                # Generate embedding
                if model:
                    embedding = generate_transformer_embedding(model, origin, destination, distance or 0)
                else:
                    embedding = generate_simple_embedding(origin, destination, distance or 0)
                
                # Store as JSON string
                embedding_json = json.dumps(embedding)
                
                # Update database
                conn.execute("""
                    UPDATE route_analytics
                    SET route_embedding = ?
                    WHERE id = ?
                """, [embedding_json, route_id])
                
                processed += 1
                if processed % 10 == 0:
                    print(f"Processed {processed}/{len(routes)} routes...")
                
            except Exception as e:
                print(f"⚠️  Error processing route {route_id}: {e}")
                continue
        
        conn.close()
        
        print("\n" + "="*50)
        print(f"✅ Generated embeddings for {processed} routes!")
        print("="*50)
        
        return True
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = generate_embeddings()
    sys.exit(0 if success else 1)
