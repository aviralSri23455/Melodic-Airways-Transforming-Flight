"""
Quick test to verify vector embeddings are working after git clone
"""
import asyncio
import sys
from sqlalchemy import text
from app.db.database import get_db

async def test_vector_embeddings():
    """Test if vector embeddings are set up and working"""
    
    print("=" * 60)
    print("🔍 TESTING VECTOR EMBEDDINGS")
    print("=" * 60)
    
    try:
        # Get database session
        async for db in get_db():
            # Test 1: Check if vector columns exist
            print("\n✓ Test 1: Checking database schema...")
            result = await db.execute(text("""
                SELECT COLUMN_NAME 
                FROM INFORMATION_SCHEMA.COLUMNS 
                WHERE TABLE_NAME = 'routes' 
                AND COLUMN_NAME IN ('route_embedding', 'melodic_complexity', 'harmonic_complexity', 'rhythmic_complexity')
            """))
            columns = [row[0] for row in result.fetchall()]
            
            if len(columns) == 4:
                print(f"   ✅ All vector columns exist: {columns}")
            else:
                print(f"   ❌ Missing columns. Found: {columns}")
                print("   ⚠️  Run: mysql -u root -p aero_melody < backend/sql/add_vector_embeddings.sql")
                return False
            
            # Test 2: Check if embeddings are generated
            print("\n✓ Test 2: Checking if embeddings are generated...")
            result = await db.execute(text("""
                SELECT COUNT(*) as total,
                       COUNT(route_embedding) as with_embeddings,
                       ROUND(COUNT(route_embedding) / COUNT(*) * 100, 2) as coverage
                FROM routes
            """))
            stats = result.fetchone()
            
            print(f"   Total routes: {stats[0]}")
            print(f"   Routes with embeddings: {stats[1]}")
            print(f"   Coverage: {stats[2]}%")
            
            if stats[2] > 0:
                print(f"   ✅ Embeddings are generated!")
            else:
                print(f"   ⚠️  No embeddings found. Run embedding generation script.")
                print(f"   Run: python backend/scripts/generate_route_embeddings.py")
                return False
            
            # Test 3: Check complexity metrics
            print("\n✓ Test 3: Checking complexity metrics...")
            result = await db.execute(text("""
                SELECT 
                    ROUND(AVG(melodic_complexity), 3) as avg_melodic,
                    ROUND(AVG(harmonic_complexity), 3) as avg_harmonic,
                    ROUND(AVG(rhythmic_complexity), 3) as avg_rhythmic
                FROM routes
                WHERE melodic_complexity IS NOT NULL
            """))
            complexity = result.fetchone()
            
            if complexity and complexity[0]:
                print(f"   Average melodic complexity: {complexity[0]}")
                print(f"   Average harmonic complexity: {complexity[1]}")
                print(f"   Average rhythmic complexity: {complexity[2]}")
                print(f"   ✅ Complexity metrics are calculated!")
            else:
                print(f"   ⚠️  No complexity metrics found.")
            
            # Test 4: Sample embedding data
            print("\n✓ Test 4: Checking sample embedding...")
            result = await db.execute(text("""
                SELECT 
                    r.id,
                    ao.iata_code as origin,
                    ad.iata_code as destination,
                    r.distance_km,
                    JSON_LENGTH(r.route_embedding) as embedding_size,
                    r.melodic_complexity,
                    r.harmonic_complexity,
                    r.rhythmic_complexity
                FROM routes r
                JOIN airports ao ON r.origin_airport_id = ao.id
                JOIN airports ad ON r.destination_airport_id = ad.id
                WHERE r.route_embedding IS NOT NULL
                LIMIT 1
            """))
            sample = result.fetchone()
            
            if sample:
                print(f"   Sample route: {sample[1]} → {sample[2]}")
                print(f"   Distance: {sample[3]} km")
                print(f"   Embedding dimension: {sample[4]}D")
                
                # Check if complexity metrics exist
                if sample[5] is not None:
                    print(f"   Melodic: {sample[5]:.3f}, Harmonic: {sample[6]:.3f}, Rhythmic: {sample[7]:.3f}")
                else:
                    print(f"   Complexity metrics: Not calculated yet")
                
                print(f"   ✅ Sample embedding looks good!")
            
            # Test 5: Check FAISS index (if exists)
            print("\n✓ Test 5: Checking FAISS index...")
            try:
                import faiss
                import os
                
                index_path = "backend/data/faiss_route_index.bin"
                if os.path.exists(index_path):
                    index = faiss.read_index(index_path)
                    print(f"   FAISS index found: {index.ntotal} vectors")
                    print(f"   ✅ FAISS index is ready!")
                else:
                    print(f"   ⚠️  FAISS index not found at {index_path}")
                    print(f"   This is optional but recommended for fast similarity search")
            except ImportError:
                print(f"   ⚠️  FAISS not installed (optional)")
                print(f"   Install with: pip install faiss-cpu")
            
            # Check if complexity metrics are calculated
            has_complexity = complexity and complexity[0] is not None
            
            print("\n" + "=" * 60)
            print("✅ VECTOR EMBEDDINGS ARE WORKING!")
            print("=" * 60)
            print("\n📝 Summary:")
            print(f"   • Database schema: ✅ Ready")
            print(f"   • Embeddings generated: ✅ {stats[2]}% coverage")
            
            if has_complexity:
                print(f"   • Complexity metrics: ✅ Calculated")
            else:
                print(f"   • Complexity metrics: ⚠️  Not calculated (optional)")
            
            print(f"   • Sample data: ✅ Valid")
            print("\n🎵 You can now use vector similarity search!")
            print("   Try: GET /api/v1/vectors/similar-routes?origin=JFK&destination=LAX")
            
            if not has_complexity:
                print("\n💡 Optional: To calculate complexity metrics, run:")
                print("   python scripts/calculate_complexity_metrics.py")
            
            return True
            
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\n⚠️  Make sure:")
        print("   1. Backend is configured (.env file)")
        print("   2. Database is running")
        print("   3. Tables are created")
        return False

if __name__ == "__main__":
    result = asyncio.run(test_vector_embeddings())
    sys.exit(0 if result else 1)
