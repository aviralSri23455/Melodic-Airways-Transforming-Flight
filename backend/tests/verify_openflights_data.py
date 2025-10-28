#!/usr/bin/env python3
"""
🔍 OpenFlights Data Verification Script
Checks if real OpenFlights data is loaded in your database
"""

import asyncio
import sys
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from sqlalchemy.future import select
from sqlalchemy import func, text
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import your models
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from app.models.models import Airport, Route
from app.core.config import settings

async def verify_openflights_data():
    """Verify OpenFlights data is properly loaded"""
    
    print("🔍 OPENFLIGHTS DATA VERIFICATION")
    print("=" * 50)
    
    try:
        # Create database connection
        engine = create_async_engine(settings.DATABASE_URL)
        async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
        
        async with async_session() as db:
            print("✅ Database connection successful")
            
            # Check airports table
            print("\n📍 AIRPORTS DATA:")
            print("-" * 30)
            
            # Total airports
            total_airports = await db.execute(select(func.count(Airport.id)))
            total_count = total_airports.scalar()
            print(f"Total airports: {total_count}")
            
            # Airports with IATA codes
            airports_with_iata = await db.execute(
                select(func.count(Airport.id)).where(Airport.iata_code.isnot(None))
            )
            iata_count = airports_with_iata.scalar()
            print(f"Airports with IATA codes: {iata_count}")
            
            # Airports with coordinates
            airports_with_coords = await db.execute(
                select(func.count(Airport.id)).where(
                    Airport.latitude.isnot(None) & Airport.longitude.isnot(None)
                )
            )
            coords_count = airports_with_coords.scalar()
            print(f"Airports with coordinates: {coords_count}")
            
            # Sample airports
            sample_airports = await db.execute(
                select(Airport).where(Airport.iata_code.isnot(None)).limit(5)
            )
            airports = sample_airports.scalars().all()
            
            print(f"\n📋 Sample airports:")
            for airport in airports:
                print(f"  {airport.iata_code} - {airport.name} ({airport.city}, {airport.country})")
            
            # Check specific test airports
            print(f"\n🎯 Test airports:")
            test_codes = ["DEL", "LHR", "JFK", "LAX", "NRT", "SYD"]
            for code in test_codes:
                airport = await db.execute(
                    select(Airport).where(Airport.iata_code == code).limit(1)
                )
                airport = airport.scalar_one_or_none()
                if airport:
                    print(f"  ✅ {code} - {airport.name} ({airport.city})")
                else:
                    print(f"  ❌ {code} - NOT FOUND")
            
            # Check routes table
            print(f"\n✈️ ROUTES DATA:")
            print("-" * 30)
            
            # Total routes
            total_routes = await db.execute(select(func.count(Route.id)))
            routes_count = total_routes.scalar()
            print(f"Total routes: {routes_count}")
            
            # Routes with distance data
            routes_with_distance = await db.execute(
                select(func.count(Route.id)).where(Route.distance_km.isnot(None))
            )
            distance_count = routes_with_distance.scalar()
            print(f"Routes with distance data: {distance_count}")
            
            # Sample routes
            sample_routes = await db.execute(
                select(Route).limit(5)
            )
            routes = sample_routes.scalars().all()
            
            print(f"\n📋 Sample routes:")
            for route in routes[:3]:
                origin = await db.execute(
                    select(Airport).where(Airport.id == route.origin_airport_id)
                )
                origin = origin.scalar_one_or_none()
                
                dest = await db.execute(
                    select(Airport).where(Airport.id == route.destination_airport_id)
                )
                dest = dest.scalar_one_or_none()
                
                origin_code = origin.iata_code if origin else "Unknown"
                dest_code = dest.iata_code if dest else "Unknown"
                distance = route.distance_km or "No data"
                
                print(f"  {origin_code} → {dest_code} ({distance} km)")
            
            # Data quality assessment
            print(f"\n📊 DATA QUALITY ASSESSMENT:")
            print("-" * 40)
            
            airport_quality = (coords_count / total_count * 100) if total_count > 0 else 0
            route_quality = (distance_count / routes_count * 100) if routes_count > 0 else 0
            
            print(f"Airport data quality: {airport_quality:.1f}% complete")
            print(f"Route data quality: {route_quality:.1f}% complete")
            
            # OpenFlights benchmark
            print(f"\n🎯 OPENFLIGHTS BENCHMARK:")
            print("-" * 35)
            print(f"Expected airports: ~3,000+")
            print(f"Your airports: {total_count}")
            print(f"Expected routes: ~67,000+") 
            print(f"Your routes: {routes_count}")
            
            # Verdict
            print(f"\n🏆 VERDICT:")
            print("-" * 15)
            
            if total_count >= 3000 and routes_count >= 60000:
                print("✅ REAL OPENFLIGHTS DATA DETECTED!")
                print("✅ Your database contains the full OpenFlights dataset")
                print("✅ No mock data - this is production-ready!")
            elif total_count >= 1000 and routes_count >= 10000:
                print("⚠️  PARTIAL OPENFLIGHTS DATA")
                print("⚠️  You have substantial data but may be missing some records")
                print("💡 Consider re-importing the full dataset")
            else:
                print("❌ LIMITED DATA DETECTED")
                print("❌ This appears to be test/mock data")
                print("💡 You need to import the full OpenFlights dataset")
                print("💡 Download from: https://github.com/MariaDB/openflights")
            
            # Recommendations
            print(f"\n💡 RECOMMENDATIONS:")
            print("-" * 25)
            
            if total_count < 3000:
                print("📥 Import airports.dat from OpenFlights GitHub")
                print("🔗 https://raw.githubusercontent.com/jpatokal/openflights/master/data/airports.dat")
            
            if routes_count < 60000:
                print("📥 Import routes.dat from OpenFlights GitHub") 
                print("🔗 https://raw.githubusercontent.com/jpatokal/openflights/master/data/routes.dat")
            
            if airport_quality < 90:
                print("🔧 Clean up airport data - remove entries without IATA codes")
            
            if route_quality < 80:
                print("🔧 Calculate missing route distances using haversine formula")
            
            print(f"\n✅ Verification complete!")
            
    except Exception as e:
        print(f"❌ Error connecting to database: {e}")
        print(f"💡 Check your DATABASE_URL in .env file")
        print(f"💡 Make sure MariaDB is running")
        return False
    
    return True

if __name__ == "__main__":
    asyncio.run(verify_openflights_data())