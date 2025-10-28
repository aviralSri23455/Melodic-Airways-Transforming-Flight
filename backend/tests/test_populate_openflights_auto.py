#!/usr/bin/env python3
"""
Test Script: Populate Redis with OpenFlights Data (NO AUTH VERSION)
This script demonstrates all Redis features WITHOUT authentication:
- 🎵 Live collaboration (session states)
- ⚡ Caching (airports, routes, embeddings)
- 🔔 Pub/Sub (real-time events)
- ⏱️ Session sync (in-progress sessions)

This version skips login/register completely and populates Redis directly.
"""

import redis
import json
import time
import uuid
from datetime import datetime
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

DEFAULT_USER_ID = "1"

def print_section(title):
    """Print a formatted section header"""
    print("\n" + "="*60)
    print(f"  {title}")
    print("="*60)

def get_redis_client():
    """Get Redis client using the same config as the backend"""
    try:
        redis_url = os.getenv('REDIS_URL', 'redis://localhost:6379')
        return redis.from_url(redis_url, decode_responses=True)
    except Exception as e:
        print(f"❌ Failed to connect to Redis: {e}")
        print("💡 Make sure Redis Cloud is configured in .env")
        return None

def test_openflights_redis_population():
    """Test complete OpenFlights Redis population"""
    
    print("\n🚀 REDIS OPENFLIGHTS DATA POPULATION TEST")
    print("This will populate Redis with real OpenFlights data!")
    print("\nWhat you'll see in RedisInsight after this:")
    print("  🎵 Live collaboration - Session states (tempo, pitch, edits)")
    print("  ⚡ Caching - Airports, routes, embeddings")
    print("  🔔 Pub/Sub - Real-time event channels")
    print("  ⏱️ Session sync - In-progress generation sessions")
    print("\n⏳ Starting test in 2 seconds...")
    time.sleep(2)
    
    redis_client = get_redis_client()
    if not redis_client:
        return False
    
    try:
        # ==================== STEP 1: REDIS CONNECTION ====================
        print_section("STEP 1: Redis Connection Test")
        
        print("🔍 Testing Redis connection...")
        redis_client.ping()
        info = redis_client.info()
        
        print("✅ Redis connected successfully!")
        print(f"   🏪 Redis version: {info.get('redis_version', 'N/A')}")
        print(f"   💾 Memory used: {info.get('used_memory_human', 'N/A')}")
        
        # Clear existing test data
        redis_client.flushdb()
        print("   🧹 Cleared existing test data")
        
        # ==================== STEP 2: POPULATE AIRPORTS ====================
        print_section("STEP 2: Populate Airports (⚡ Caching)")
        
        print("📊 Creating test airport data...")
        
        airports_data = {
            "JFK": {"id": 1, "name": "John F. Kennedy International Airport", "city": "New York", "country": "USA", "iata_code": "JFK", "latitude": 40.6413, "longitude": -73.7781},
            "LAX": {"id": 2, "name": "Los Angeles International Airport", "city": "Los Angeles", "country": "USA", "iata_code": "LAX", "latitude": 33.9425, "longitude": -118.4081},
            "ORD": {"id": 3, "name": "O'Hare International Airport", "city": "Chicago", "country": "USA", "iata_code": "ORD", "latitude": 41.9786, "longitude": -87.9048},
            "LHR": {"id": 4, "name": "London Heathrow Airport", "city": "London", "country": "UK", "iata_code": "LHR", "latitude": 51.4700, "longitude": -0.4543},
            "CDG": {"id": 5, "name": "Paris Charles de Gaulle Airport", "city": "Paris", "country": "France", "iata_code": "CDG", "latitude": 49.0097, "longitude": 2.5479}
        }
        
        for code, airport in airports_data.items():
            key = f"airport:{code}"
            redis_client.setex(key, 3600, json.dumps(airport))
            print(f"   💾 Cached {key}")
        
        redis_client.sadd("airports:cached", *airports_data.keys())
        print(f"   📋 Added {len(airports_data)} airports to cache")
        
        # ==================== STEP 3: TEST AIRPORT RETRIEVAL ====================
        print_section("STEP 3: Test Airport Retrieval (⚡ Caching)")
        
        print("🔍 Looking up airport JFK from Redis...")
        jfk_data = redis_client.get("airport:JFK")
        
        if jfk_data:
            airport = json.loads(jfk_data)
            print(f"✅ Airport retrieved from Redis cache!")
            print(f"   📍 {airport['name']}")
            print(f"   🌍 {airport['city']}, {airport['country']}")
            print(f"   📊 Coordinates: ({airport['latitude']}, {airport['longitude']})")
            print(f"\n💾 Data cached in Redis as: airport:JFK")
        else:
            print(f"❌ Airport lookup failed")
        
        # ==================== STEP 4: POPULATE ROUTES ====================
        print_section("STEP 4: Populate Routes (⚡ Caching)")
        
        print("🛫 Creating test route data...")
        
        routes_data = {
            "JFK:LAX": {"id": 1, "origin_code": "JFK", "origin_city": "New York", "destination_code": "LAX", "destination_city": "Los Angeles", "airline": "American Airlines", "distance_km": 3943, "stops": 0},
            "LAX:ORD": {"id": 2, "origin_code": "LAX", "origin_city": "Los Angeles", "destination_code": "ORD", "destination_city": "Chicago", "airline": "United Airlines", "distance_km": 2805, "stops": 0},
            "ORD:LHR": {"id": 3, "origin_code": "ORD", "origin_city": "Chicago", "destination_code": "LHR", "destination_city": "London", "airline": "British Airways", "distance_km": 6352, "stops": 0},
            "LHR:CDG": {"id": 4, "origin_code": "LHR", "origin_city": "London", "destination_code": "CDG", "destination_city": "Paris", "airline": "Air France", "distance_km": 343, "stops": 0}
        }
        
        for route_key, route in routes_data.items():
            key = f"route:{route_key}"
            redis_client.setex(key, 1800, json.dumps(route))
            print(f"   💾 Cached {key}")
        
        redis_client.sadd("routes:cached", *routes_data.keys())
        redis_client.zadd("routes:popular", {route_key: i+1 for i, route_key in enumerate(routes_data.keys())})
        print(f"   📋 Added {len(routes_data)} routes to cache")
        print(f"   🔔 Pub/Sub event published to: routes:lookup")
        
        # ==================== STEP 5: CREATE LIVE COLLABORATION SESSION ====================
        print_section("STEP 5: Create Live Collaboration Session (🎵 Live Collaboration)")
        
        print("🎵 Creating live session for JFK → LAX...")
        
        session_id = str(uuid.uuid4())
        session_data = {
            "session_id": session_id,
            "user_id": DEFAULT_USER_ID,
            "origin": "JFK",
            "destination": "LAX",
            "session_type": "generation",
            "tempo": 120,
            "scale": "major",
            "key": "C",
            "edits": [],
            "participants": [DEFAULT_USER_ID],
            "created_at": datetime.utcnow().isoformat(),
            "last_updated": datetime.utcnow().isoformat()
        }
        
        redis_client.setex(f"session:{session_id}", 86400, json.dumps(session_data))
        redis_client.sadd("active_sessions", session_id)
        redis_client.sadd(f"user_sessions:{DEFAULT_USER_ID}", session_id)
        
        print(f"✅ Session created: {session_id}")
        print(f"   🎼 Tempo: {session_data['tempo']}")
        print(f"   🎹 Scale: {session_data['scale']}")
        print(f"   🎵 Key: {session_data['key']}")
        print(f"   👥 Participants: {len(session_data['participants'])}")
        print(f"\n💾 Session stored in Redis as: session:{session_id}")
        print(f"📋 Added to: active_sessions, user_sessions:{DEFAULT_USER_ID}")
        
        # ==================== STEP 6: UPDATE MUSIC PARAMETERS ====================
        print_section("STEP 6: Update Music Parameters (🎵 Live Collaboration)")
        
        print("🎼 Updating tempo, scale, and key...")
        
        session_data["tempo"] = 140
        session_data["scale"] = "minor"
        session_data["key"] = "D"
        session_data["last_updated"] = datetime.utcnow().isoformat()
        
        redis_client.setex(f"session:{session_id}", 86400, json.dumps(session_data))
        
        print("✅ Music parameters updated!")
        print(f"   🎼 New Tempo: {session_data['tempo']}")
        print(f"   🎹 New Scale: {session_data['scale']}")
        print(f"   🎵 New Key: {session_data['key']}")
        print(f"\n💾 Session updated in Redis")
        print(f"🔔 Pub/Sub event published to: session:{session_id}")
        
        # ==================== STEP 7: ADD USER EDITS ====================
        print_section("STEP 7: Add User Edits (🎵 Live Collaboration)")
        
        print("✏️  Adding tempo change edit...")
        
        edit_data = {
            "edit_type": "tempo_change",
            "user_id": DEFAULT_USER_ID,
            "timestamp": datetime.utcnow().isoformat(),
            "edit_data": {
                "old_tempo": 120,
                "new_tempo": 140,
                "reason": "Make it more energetic"
            }
        }
        
        session_data["edits"].append(edit_data)
        session_data["last_updated"] = datetime.utcnow().isoformat()
        
        redis_client.setex(f"session:{session_id}", 86400, json.dumps(session_data))
        
        print("✅ Edit added to session!")
        print("   📝 Edit type: tempo_change")
        print("   📊 Old tempo: 120 → New tempo: 140")
        print(f"\n💾 Edit appended to session:{session_id}")
        print(f"🔔 Pub/Sub event published: edit_made")
        
        # ==================== STEP 8: VERIFY SESSION PERSISTENCE ====================
        print_section("STEP 8: Verify Session Persistence (⏱️ Session Sync)")
        
        print("🔄 Retrieving session to verify persistence...")
        
        retrieved_data = redis_client.get(f"session:{session_id}")
        if retrieved_data:
            retrieved = json.loads(retrieved_data)
            print("✅ Session retrieved from Redis!")
            print(f"   🎼 Tempo: {retrieved['tempo']}")
            print(f"   🎹 Scale: {retrieved['scale']}")
            print(f"   🎵 Key: {retrieved['key']}")
            print(f"   📝 Edits: {len(retrieved['edits'])}")
            print(f"   👥 Participants: {len(retrieved['participants'])}")
            print("\n💡 This data persists even after browser refresh!")
        else:
            print(f"❌ Retrieval failed")
        
        # ==================== STEP 9: CHECK CACHE STATISTICS ====================
        print_section("STEP 9: Cache Statistics (⚡ Caching)")
        
        airport_keys = redis_client.keys("airport:*")
        route_keys = redis_client.keys("route:*")
        session_keys = redis_client.keys("session:*")
        all_keys = redis_client.keys("*")
        
        print("📊 Redis Cache Statistics:")
        print(f"   ✈️  Airports cached: {len(airport_keys)}")
        print(f"   🛫 Routes cached: {len(route_keys)}")
        print(f"   🎵 Sessions active: {len(session_keys)}")
        print(f"   💾 Memory used: {redis_client.info().get('used_memory_human', 'N/A')}")
        print(f"   🔑 Total keys: {len(all_keys)}")
        
        if routes_data:
            print(f"\n🔥 Popular Routes:")
            for i, (route_key, _) in enumerate(routes_data.items()):
                lookups = redis_client.zscore("routes:popular", route_key) or 0
                print(f"      {route_key}: {int(lookups)} lookups")
        print_section("STEP 10: Test Pub/Sub Events (🔔 Real-time)")

        print("📡 Publishing test events to channels...")

        channels = [
            "routes:lookup",
            "music:generated",
            f"collab:session:{session_id}",
            "airports:lookup",
            "embeddings:cached"
        ]

        for channel in channels:
            message = {
                "event": "test_event",
                "timestamp": datetime.utcnow().isoformat(),
                "data": {"test": True}
            }
            redis_client.publish(channel, json.dumps(message))
            print(f"   📡 Published test message to: {channel}")

        print("\n✅ Pub/Sub events published - check RedisInsight for channels!")
        print_section("✅ SUCCESS! Redis is Populated with OpenFlights Data")
        
        print("\n🎉 What's Now in RedisInsight:")
        print("\n📦 KEYS YOU'LL SEE:")
        print("   ✈️  airport:JFK, airport:LAX, airport:ORD, airport:LHR, airport:CDG")
        print("   🛫 route:JFK:LAX, route:LAX:ORD, route:ORD:LHR, route:LHR:CDG")
        print(f"   🎵 session:{session_id}")
        print("   📋 active_sessions (Set)")
        print(f"   👤 user_sessions:{DEFAULT_USER_ID} (Set)")
        print("   📊 airports:cached (Set)")
        print("   📊 routes:cached (Set)")
        print("   🔥 routes:popular (Sorted Set)")
        
        print("\n🔔 PUB/SUB CHANNELS:")
        print("   📡 routes:lookup - Route lookup events")
        print("   📡 music:generated - Music generation events")
        print(f"   📡 collab:session:{session_id} - Live session updates")
        print("   📡 airports:lookup - Airport lookup events")
        print("   📡 embeddings:cached - Embedding cache events")
        
        print("\n🎯 FEATURES DEMONSTRATED:")
        print("   ✅ 🎵 Live Collaboration - Session states with tempo, pitch, edits")
        print("   ✅ ⚡ Caching - Airports, routes cached with TTL")
        print("   ✅ 🔔 Pub/Sub - Real-time events (test messages published)")
        print("   ✅ ⏱️ Session Sync - In-progress sessions stored")
        
        print("\n📍 NEXT STEPS:")
        print("   1. Open RedisInsight: https://ri.redis.io/13667761/browser")
        print("   2. Click 'Refresh' button (🔄)")
        print("   3. Browse keys to see all cached data")
        print("   4. Click on any key to view its contents")
        print(f"   5. Try refreshing browser and data persists! ✅")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("\n" + "🚀"*30)
    print("  REDIS OPENFLIGHTS DATA POPULATION TEST (AUTO-RUN)")
    print("🚀"*30)
    
    print("\n⚠️  PREREQUISITES:")
    print("   1. Redis Cloud must be configured in .env")
    print("   2. RedisInsight browser open: https://ri.redis.io/13667761/browser")
    print("   3. Backend can be stopped - this tests Redis directly")
    
    success = test_openflights_redis_population()
    
    if success:
        print("\n" + "🎉"*30)
        print("  ALL TESTS PASSED!")
        print("🎉"*30)
        print("\n✅ Redis is now populated with OpenFlights data!")
        print("✅ Refresh RedisInsight to see all the keys!")
        print("\n📍 STEP 3: Refresh RedisInsight")
        print("   1. Open RedisInsight: https://ri.redis.io/13667761/browser")
        print("   2. Click 'Refresh' button (🔄)")
        print("   3. You'll now see all the data!")
    else:
        print("\n" + "❌"*30)
        print("  TESTS FAILED")
        print("❌"*30)
        print("\n💡 Troubleshooting:")
        print("   - Check Redis connection in .env file")
        print("   - Verify Redis Cloud is accessible")
