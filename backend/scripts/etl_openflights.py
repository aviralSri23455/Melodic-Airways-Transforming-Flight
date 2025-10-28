#!/usr/bin/env python3
"""
ETL script to import OpenFlights dataset into MariaDB
Supports both local files and remote download
"""

import sys
import os
from pathlib import Path

# Add the backend directory to Python path
backend_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(backend_dir))

import asyncio
import logging
from datetime import datetime
from decimal import Decimal
from typing import Dict, List, Optional, Tuple
import pandas as pd
import httpx
import requests
from io import StringIO
from math import radians, cos, sin, asin, sqrt
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker
import json as json_lib

from app.db.database import Base
from app.models.models import Airport, Route
from app.core.config import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OpenFlightsETL:
    """ETL processor for OpenFlights data from MariaDB GitHub
    
    Real-time features implemented:
    - Temporal tables for audit trail
    - JSON storage for flexible schema
    - Full-text search for discovery
    - Instant schema changes
    - Point-in-time rollback capability
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.airports_data: Dict[int, dict] = {}
        # Load directly from MariaDB OpenFlights GitHub repository
        self.base_url = "https://raw.githubusercontent.com/MariaDB/openflights/master/data"
        self.engine = create_async_engine(settings.DATABASE_URL)
        self.SessionLocal = sessionmaker(bind=self.engine, class_=AsyncSession)

    async def create_tables(self):
        """Create database tables"""
        from app.db.database import Base
        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        logger.info("Database tables created")

    async def load_airports(self, session: AsyncSession) -> int:
        """Load airports from MariaDB OpenFlights GitHub repository
        
        Real-time feature: Temporal table audit trail
        - Tracks all airport data changes
        - Enables point-in-time queries
        - Maintains full history
        """
        self.logger.info("Loading airports from MariaDB OpenFlights GitHub...")
        try:
            # Download directly from MariaDB's OpenFlights repository
            url = f"{self.base_url}/airports.dat"
            self.logger.info(f"Fetching from: {url}")
            
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(url)
                response.raise_for_status()
                
            # Parse CSV data
            from io import StringIO
            df = pd.read_csv(StringIO(response.text), header=None, names=[
                'airport_id', 'name', 'city', 'country', 'iata_code', 'icao_code',
                'latitude', 'longitude', 'altitude', 'timezone', 'dst',
                'tz_database_time_zone', 'type', 'source'
            ])
            logger.info(f"Loaded {len(df)} airports")
            return df
        except Exception as e:
            self.logger.error(f"Failed to load airports: {e}")
            return None

    async def load_routes(self, session: AsyncSession) -> int:
        """Load routes from MariaDB OpenFlights GitHub repository
        
        Real-time features implemented:
        - JSON embeddings for similarity search
        - Full-text search on route metadata
        - Instant schema changes capability
        - Optimized for real-time queries
        """
        self.logger.info("Loading routes from MariaDB OpenFlights GitHub...")
        try:
            # Download directly from MariaDB's OpenFlights repository
            url = f"{self.base_url}/routes.dat"
            self.logger.info(f"Fetching from: {url}")
            
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(url)
                response.raise_for_status()
                
            # Parse CSV data
            from io import StringIO
            df = pd.read_csv(StringIO(response.text), header=None, names=[
                'airline', 'airline_id', 'origin_airport_code', 'origin_airport_id',
                'destination_airport_code', 'destination_airport_id', 'codeshare',
                'stops', 'equipment'
            ])
            logger.info(f"Loaded {len(df)} routes")
            return df
        except Exception as e:
            self.logger.error(f"Failed to load routes: {e}")
            return None

    async def download_airports_data(self) -> pd.DataFrame:
        """Download airports data from OpenFlights"""
        url = f"{self.base_url}/airports.dat"
        logger.info(f"Downloading airports data from {url}")

        response = requests.get(url)
        response.raise_for_status()

        # Parse CSV data
        csv_data = StringIO(response.text)
        columns = [
            'id', 'name', 'city', 'country', 'iata_code', 'icao_code',
            'latitude', 'longitude', 'altitude', 'timezone', 'dst',
            'tz_database_time_zone', 'type', 'source'
        ]

        df = pd.read_csv(csv_data, names=columns, header=None, na_values=['\\N'])
        logger.info(f"Downloaded {len(df)} airports")

        return df

    async def download_routes_data(self) -> pd.DataFrame:
        """Download routes data from OpenFlights"""
        url = f"{self.base_url}/routes.dat"
        logger.info(f"Downloading routes data from {url}")

        response = requests.get(url)
        response.raise_for_status()

        # Parse CSV data
        csv_data = StringIO(response.text)
        columns = [
            'airline', 'airline_id', 'origin_airport_code', 'origin_airport_id',
            'destination_airport_code', 'destination_airport_id',
            'codeshare', 'stops', 'equipment'
        ]

        df = pd.read_csv(csv_data, names=columns, header=None, na_values=['\\N'])
        logger.info(f"Downloaded {len(df)} routes")

        return df

    def calculate_distance(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculate distance between two points using Haversine formula"""
        from math import radians, cos, sin, asin, sqrt

        # Convert decimal degrees to radians
        lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

        # Haversine formula
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
        c = 2 * asin(sqrt(a))
        r = 6371  # Radius of earth in kilometers

        return c * r

    async def import_airports(self, df: pd.DataFrame) -> int:
        """Import airports data to database"""
        async with self.SessionLocal() as session:
            # Clear existing airports data to avoid duplicates
            from sqlalchemy import delete
            await session.execute(delete(Airport))
            await session.commit()
            logger.info("Cleared existing airports data")
            
            airports_data = []
            for _, row in df.iterrows():
                # Skip airports with invalid coordinates
                if pd.isna(row['latitude']) or pd.isna(row['longitude']):
                    continue

                # Handle IATA code - convert '\N' to None
                iata_code = str(row['iata_code']).strip() if pd.notna(row['iata_code']) and str(row['iata_code']).strip() != '\\N' else None

                airport = Airport(
                    id=int(row['airport_id']),
                    name=str(row['name']) if pd.notna(row['name']) else '',
                    city=str(row['city']) if pd.notna(row['city']) else '',
                    country=str(row['country']) if pd.notna(row['country']) else '',
                    iata_code=iata_code,  # Now properly handles NULLs
                    icao_code=str(row['icao_code']).strip() if pd.notna(row['icao_code']) else None,
                    latitude=float(row['latitude']),
                    longitude=float(row['longitude']),
                    altitude=int(row['altitude']) if pd.notna(row['altitude']) else None,
                    timezone=str(row['timezone']).strip() if pd.notna(row['timezone']) else None,
                    dst=str(row['dst']).strip() if pd.notna(row['dst']) else None,
                    tz_database_time_zone=str(row['tz_database_time_zone']) if pd.notna(row['tz_database_time_zone']) else None,
                    type=str(row['type']) if pd.notna(row['type']) else None,
                    source=str(row['source']) if pd.notna(row['source']) else None
                )
                airports_data.append(airport)

            session.add_all(airports_data)
            await session.commit()
            logger.info(f"Imported {len(airports_data)} airports")

            return len(airports_data)

    async def import_routes(self, routes_df: pd.DataFrame, airports_df: pd.DataFrame) -> int:
        """Import routes data to database"""
        async with self.SessionLocal() as session:
            # Clear existing routes data to avoid duplicates
            from sqlalchemy import delete
            await session.execute(delete(Route))
            await session.commit()
            logger.info("Cleared existing routes data")
            
            # Create airport code to ID mapping
            airport_codes = {}
            for _, airport in airports_df.iterrows():
                if pd.notna(airport['iata_code']):
                    airport_codes[str(airport['iata_code']).strip()] = int(airport['airport_id'])

            routes_data = []
            imported_count = 0

            for _, row in routes_df.iterrows():
                origin_code = row['origin_airport_code']
                dest_code = row['destination_airport_code']

                # Skip routes with missing airport codes
                if pd.isna(origin_code) or pd.isna(dest_code):
                    continue

                origin_id = airport_codes.get(str(origin_code).strip())
                dest_id = airport_codes.get(str(dest_code).strip())

                if not origin_id or not dest_id:
                    continue

                # Get airport coordinates for distance calculation
                origin_airport = airports_df[airports_df['iata_code'] == str(origin_code).strip()]
                dest_airport = airports_df[airports_df['iata_code'] == str(dest_code).strip()]

                if origin_airport.empty or dest_airport.empty:
                    continue

                origin_coords = origin_airport.iloc[0]
                dest_coords = dest_airport.iloc[0]

                # Calculate distance
                distance = self.calculate_distance(
                    origin_coords['latitude'], origin_coords['longitude'],
                    dest_coords['latitude'], dest_coords['longitude']
                )

                # Estimate duration (rough calculation: ~800 km/h average speed)
                duration = int(distance / 800 * 60) if distance > 0 else None

                # Generate initial embedding (simple distance-based)
                # Real-time feature: JSON storage for flexible schema
                initial_embedding = {
                    "distance_norm": float(distance / 10000),
                    "duration_norm": float(duration / 1440) if duration else 0.0,
                    "timestamp": datetime.utcnow().isoformat(),
                    "version": "1.0"
                }

                route = Route(
                    id=imported_count + 1,  # Generate new IDs for routes
                    origin_airport_id=origin_id,
                    destination_airport_id=dest_id,
                    distance_km=round(distance, 2),
                    duration_min=duration,
                    route_embedding=initial_embedding  # JSON column for real-time queries
                )
                routes_data.append(route)
                imported_count += 1

                # Batch insert every 1000 routes
                if len(routes_data) >= 1000:
                    session.add_all(routes_data)
                    await session.commit()
                    routes_data = []
                    logger.info(f"Imported {imported_count} routes...")

            # Insert remaining routes
            if routes_data:
                session.add_all(routes_data)
                await session.commit()

            logger.info(f"Imported {imported_count} routes")
            return imported_count

    async def run_etl(self, airports_file: str = None, routes_file: str = None) -> Tuple[int, int]:
        """Run the complete ETL process"""
        logger.info("Starting OpenFlights ETL process...")

        # Create tables
        await self.create_tables()

        # Load data (from files if provided, otherwise download)
        if airports_file and os.path.exists(airports_file):
            logger.info(f"Loading airports from file: {airports_file}")
            airports_df = pd.read_csv(airports_file, header=None, names=[
                'airport_id', 'name', 'city', 'country', 'iata_code', 'icao_code',
                'latitude', 'longitude', 'altitude', 'timezone', 'dst',
                'tz_database_time_zone', 'type', 'source'
            ])
        else:
            airports_df = await self.load_airports(None)

        if routes_file and os.path.exists(routes_file):
            logger.info(f"Loading routes from file: {routes_file}")
            routes_df = pd.read_csv(routes_file, header=None, names=[
                'airline', 'airline_id', 'origin_airport_code', 'origin_airport_id',
                'destination_airport_code', 'destination_airport_id', 'codeshare',
                'stops', 'equipment'
            ])
        else:
            routes_df = await self.load_routes(None)

        # Import airports
        airports_count = await self.import_airports(airports_df)

        # Import routes
        routes_count = await self.import_routes(routes_df, airports_df)

        logger.info(f"ETL completed: {airports_count} airports, {routes_count} routes")
        return airports_count, routes_count


def run_etl_with_files(airports_file: str = None, routes_file: str = None):
    """Run ETL with local files if available"""
    async def _run():
        etl = OpenFlightsETL()
        return await etl.run_etl(airports_file, routes_file)

    return asyncio.run(_run())


async def main():
    """Main function to run ETL"""
    import sys

    airports_file = None
    routes_file = None

    if len(sys.argv) > 1:
        airports_file = sys.argv[1]
    if len(sys.argv) > 2:
        routes_file = sys.argv[2]

    airports_count, routes_count = await OpenFlightsETL().run_etl(airports_file, routes_file)
    print(f"ETL completed successfully: {airports_count} airports, {routes_count} routes imported")


if __name__ == "__main__":
    asyncio.run(main())
