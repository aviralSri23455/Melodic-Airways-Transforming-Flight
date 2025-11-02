"""
Test similarity search functionality with DuckDB
"""

import duckdb
import sys
import json
import numpy as np
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.core.config import settings


def cosine_similarity(vec1: list, vec2: list) -> float:
    """Calculate cosine similarity between two vectors"""
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    return float(dot_product / (norm1 * norm2))


def test_similarity_search():
    """Test similarity search on routes"""
    
    db_path = getattr(settings, 'DUCKDB_PATH', './data/analytics.duckdb')
    
    print(f"Connecting to DuckDB at: {db_path}")
    
    try:
        # Connect to DuckDB
        conn = duckdb.connect(database=db_path, read_only=True)
        
        print("✅ Connected to DuckDB")
        
        # Get all routes with embeddings
        print("\nFetching routes with embeddings...")
        routes = conn.execute("""
            SELECT id, origin, destination, distance_km, route_embedding
            FROM route_analytics
            WHERE route_embedding IS NOT NULL AND route_embedding != ''
            LIMIT 100
        """).fetchall()
        
        if not routes:
            print("❌ No routes with embeddings found!")
            return False
        
        print(f"Found {len(routes)} routes with embeddings")
        
        # Test similarity search with first route
        test_route = routes[0]
        test_id, test_origin, test_dest, test_distance, test_embedding_str = test_route
        
        print(f"\n{'='*60}")
        print(f"Testing similarity search for route:")
        print(f"  ID: {test_id}")
        print(f"  Route: {test_origin} → {test_dest}")
        print(f"  Distance: {test_distance:.0f} km")
        print(f"{'='*60}\n")
        
        # Parse embedding
        test_embedding = json.loads(test_embedding_str)
        
        # Calculate similarities with all other routes
        similarities = []
        for route_id, origin, dest, distance, embedding_str in routes:
            if route_id == test_id:
                continue
            
            try:
                embedding = json.loads(embedding_str)
                similarity = cosine_similarity(test_embedding, embedding)
                similarities.append((route_id, origin, dest, distance, similarity))
            except Exception as e:
                print(f"⚠️  Error processing route {route_id}: {e}")
                continue
        
        # Sort by similarity
        similarities.sort(key=lambda x: x[4], reverse=True)
        
        # Display top 5 similar routes
        print("Top 5 most similar routes:")
        print(f"{'Rank':<6} {'Origin':<8} {'Dest':<8} {'Distance':<12} {'Similarity':<12}")
        print("-" * 60)
        
        for i, (route_id, origin, dest, distance, similarity) in enumerate(similarities[:5], 1):
            print(f"{i:<6} {origin:<8} {dest:<8} {distance:<12.0f} {similarity:<12.4f}")
        
        print("\n" + "="*60)
        print("✅ Similarity search test completed successfully!")
        print("="*60)
        
        # Test statistics
        if similarities:
            avg_similarity = np.mean([s[4] for s in similarities])
            max_similarity = max([s[4] for s in similarities])
            min_similarity = min([s[4] for s in similarities])
            
            print(f"\nStatistics:")
            print(f"  Average similarity: {avg_similarity:.4f}")
            print(f"  Max similarity: {max_similarity:.4f}")
            print(f"  Min similarity: {min_similarity:.4f}")
        
        conn.close()
        return True
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_similarity_search()
    sys.exit(0 if success else 1)
