"""
DuckDB Analytics for Aero Melody - Final Version
Analyzes your actual MariaDB schema with routes and airports
"""

import duckdb
import os
import sys
from datetime import datetime
import mysql.connector
import pandas as pd

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from app.core.config import settings


class AeroMelodyAnalytics:
    """DuckDB analytics for Aero Melody"""
    
    def __init__(self, db_path: str = None):
        """Initialize DuckDB connection"""
        self.db_path = db_path or settings.DUCKDB_PATH
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        
        self.conn = duckdb.connect(self.db_path)
        self.conn.execute(f"SET memory_limit='{settings.DUCKDB_MEMORY_LIMIT}'")
        self.conn.execute(f"SET threads={settings.DUCKDB_THREADS}")
        
        print(f"✅ Connected to DuckDB at: {self.db_path}")
        
        self.mariadb_config = {
            'host': settings.DB_HOST,
            'port': settings.DB_PORT,
            'user': settings.DB_USER,
            'password': settings.DB_PASSWORD,
            'database': settings.DB_NAME
        }
    
    def import_from_mariadb(self):
        """Import data from MariaDB"""
        print("\n📥 Importing data from MariaDB...")
        
        try:
            maria_conn = mysql.connector.connect(**self.mariadb_config)
            cursor = maria_conn.cursor()
            
            # Import airports
            print("  Fetching airports...")
            cursor.execute("DESCRIBE airports")
            airports_cols = [col[0] for col in cursor.fetchall()]
            cursor.execute("SELECT * FROM airports")
            airports_data = cursor.fetchall()
            
            if airports_data:
                self.conn.execute("DROP TABLE IF EXISTS airports")
                df = pd.DataFrame(airports_data, columns=airports_cols)
                self.conn.execute("CREATE TABLE airports AS SELECT * FROM df")
                print(f"✅ Imported {len(airports_data):,} airports")
            
            # Import routes
            print("  Fetching routes...")
            cursor.execute("DESCRIBE routes")
            routes_cols = [col[0] for col in cursor.fetchall()]
            cursor.execute("SELECT * FROM routes")
            routes_data = cursor.fetchall()
            
            if routes_data:
                self.conn.execute("DROP TABLE IF EXISTS routes")
                df = pd.DataFrame(routes_data, columns=routes_cols)
                self.conn.execute("CREATE TABLE routes AS SELECT * FROM df")
                print(f"✅ Imported {len(routes_data):,} routes")
            
            # Import travel_logs if exists
            try:
                print("  Fetching travel logs...")
                cursor.execute("DESCRIBE travel_logs")
                logs_cols = [col[0] for col in cursor.fetchall()]
                cursor.execute("SELECT * FROM travel_logs")
                logs_data = cursor.fetchall()
                
                self.conn.execute("DROP TABLE IF EXISTS travel_logs")
                
                if logs_data and len(logs_data) > 0:
                    df = pd.DataFrame(logs_data, columns=logs_cols)
                    self.conn.execute("CREATE TABLE travel_logs AS SELECT * FROM df")
                    print(f"✅ Imported {len(logs_data):,} travel logs")
                else:
                    # Create empty table with actual schema from MariaDB
                    df = pd.DataFrame(columns=logs_cols)
                    self.conn.execute("CREATE TABLE travel_logs AS SELECT * FROM df")
                    print("✅ Travel logs table imported (0 records - waiting for users to save logs)")
            except Exception as e:
                print(f"⚠️  Could not import travel_logs: {e}")
            
            cursor.close()
            maria_conn.close()
            
        except Exception as e:
            print(f"❌ MariaDB import error: {e}")
    
    def analyze_routes(self):
        """Analyze flight routes"""
        print("\n✈️  ROUTE ANALYSIS")
        print("=" * 60)
        
        # Total routes
        result = self.conn.execute("SELECT COUNT(*) FROM routes").fetchone()
        print(f"Total Routes: {result[0]:,}")
        
        # Distance statistics
        result = self.conn.execute("""
            SELECT 
                MIN(distance_km) as min_dist,
                AVG(distance_km) as avg_dist,
                MAX(distance_km) as max_dist
            FROM routes
        """).fetchone()
        print(f"\n📏 Distance Statistics:")
        print(f"  Min: {float(result[0]):,.2f} km")
        print(f"  Avg: {float(result[1]):,.2f} km")
        print(f"  Max: {float(result[2]):,.2f} km")
        
        # Duration statistics
        result = self.conn.execute("""
            SELECT 
                MIN(duration_min) as min_dur,
                AVG(duration_min) as avg_dur,
                MAX(duration_min) as max_dur
            FROM routes
        """).fetchone()
        print(f"\n⏱️  Duration Statistics:")
        print(f"  Min: {result[0]:.0f} minutes")
        print(f"  Avg: {result[1]:.0f} minutes")
        print(f"  Max: {result[2]:.0f} minutes")
        
        # Routes by distance category
        print("\n📊 Routes by Distance Category:")
        results = self.conn.execute("""
            SELECT 
                CASE 
                    WHEN distance_km < 500 THEN 'Very Short (<500km)'
                    WHEN distance_km < 1500 THEN 'Short (500-1500km)'
                    WHEN distance_km < 4000 THEN 'Medium (1500-4000km)'
                    WHEN distance_km < 8000 THEN 'Long (4000-8000km)'
                    ELSE 'Very Long (>8000km)'
                END as category,
                COUNT(*) as count,
                AVG(duration_min) as avg_duration
            FROM routes
            GROUP BY category
            ORDER BY 
                CASE category
                    WHEN 'Very Short (<500km)' THEN 1
                    WHEN 'Short (500-1500km)' THEN 2
                    WHEN 'Medium (1500-4000km)' THEN 3
                    WHEN 'Long (4000-8000km)' THEN 4
                    ELSE 5
                END
        """).fetchall()
        
        for category, count, avg_dur in results:
            print(f"  {category:25s} {count:7,} routes (avg {avg_dur:.0f} min)")
        
        # Most connected airports (as origins)
        print("\n🌐 Top 10 Origin Airports by Route Count:")
        results = self.conn.execute("""
            SELECT 
                a.iata_code,
                a.name,
                a.city,
                COUNT(*) as route_count
            FROM routes r
            JOIN airports a ON r.origin_airport_id = a.id
            WHERE a.iata_code IS NOT NULL
            GROUP BY a.iata_code, a.name, a.city
            ORDER BY route_count DESC
            LIMIT 10
        """).fetchall()
        
        for i, (code, name, city, count) in enumerate(results, 1):
            print(f"  {i:2d}. {code} - {city:20s} {count:5,} routes")
        
        # Most connected airports (as destinations)
        print("\n🎯 Top 10 Destination Airports by Route Count:")
        results = self.conn.execute("""
            SELECT 
                a.iata_code,
                a.name,
                a.city,
                COUNT(*) as route_count
            FROM routes r
            JOIN airports a ON r.destination_airport_id = a.id
            WHERE a.iata_code IS NOT NULL
            GROUP BY a.iata_code, a.name, a.city
            ORDER BY route_count DESC
            LIMIT 10
        """).fetchall()
        
        for i, (code, name, city, count) in enumerate(results, 1):
            print(f"  {i:2d}. {code} - {city:20s} {count:5,} routes")
    
    def analyze_airports(self):
        """Analyze airports"""
        print("\n🛫 AIRPORT ANALYSIS")
        print("=" * 60)
        
        # Total airports
        result = self.conn.execute("SELECT COUNT(*) FROM airports").fetchone()
        print(f"Total Airports: {result[0]:,}")
        
        # Airports by country
        print("\n🌍 Top 15 Countries by Airport Count:")
        results = self.conn.execute("""
            SELECT country, COUNT(*) as count
            FROM airports
            GROUP BY country
            ORDER BY count DESC
            LIMIT 15
        """).fetchall()
        
        for i, (country, count) in enumerate(results, 1):
            print(f"  {i:2d}. {country:30s} {count:5,} airports")
        
        # Altitude statistics
        result = self.conn.execute("""
            SELECT 
                MIN(altitude) as min_alt,
                AVG(altitude) as avg_alt,
                MAX(altitude) as max_alt
            FROM airports
            WHERE altitude IS NOT NULL
        """).fetchone()
        print(f"\n⛰️  Altitude Statistics:")
        print(f"  Min: {result[0]:,.0f} ft")
        print(f"  Avg: {result[1]:,.0f} ft")
        print(f"  Max: {result[2]:,.0f} ft")
        
        # Highest altitude airports
        print("\n🏔️  Top 5 Highest Altitude Airports:")
        results = self.conn.execute("""
            SELECT name, city, country, altitude
            FROM airports
            WHERE altitude IS NOT NULL
            ORDER BY altitude DESC
            LIMIT 5
        """).fetchall()
        
        for i, (name, city, country, alt) in enumerate(results, 1):
            print(f"  {i}. {name} ({city}, {country}) - {alt:,} ft")
        
        # Hub scores
        print("\n🌟 Top 10 Airports by Hub Score:")
        results = self.conn.execute("""
            SELECT name, city, country, hub_score
            FROM airports
            WHERE hub_score IS NOT NULL
            ORDER BY hub_score DESC
            LIMIT 10
        """).fetchall()
        
        for i, (name, city, country, score) in enumerate(results, 1):
            print(f"  {i:2d}. {name} ({city}, {country}) - Score: {score}")
    
    def analyze_travel_logs(self):
        """Analyze travel logs if available"""
        print("\n🎵 TRAVEL LOGS ANALYSIS")
        print("=" * 60)
        
        try:
            # Check if table exists
            tables = self.conn.execute("SHOW TABLES").fetchall()
            table_names = [t[0] for t in tables]
            
            if 'travel_logs' not in table_names:
                print("⚠️  Travel logs table not found in DuckDB.")
                print("💡 Run the analytics import to sync from MariaDB.")
                return
            
            result = self.conn.execute("SELECT COUNT(*) FROM travel_logs").fetchone()
            
            if result[0] == 0:
                print("✅ Travel logs table ready (imported from MariaDB)")
                print("📊 Current records: 0")
                print("")
                print("💡 Waiting for users to save travel logs in the app.")
                print("   Once users save routes, you'll see:")
                print("   • Most popular routes")
                print("   • Travel patterns by date")
                print("   • Public vs private logs")
                print("   • Most used tags")
                return
            
            print(f"Total Travel Logs: {result[0]:,}")
            
            # Most active users
            print("\n👥 Most Active Users:")
            results = self.conn.execute("""
                SELECT user_id, COUNT(*) as log_count
                FROM travel_logs
                GROUP BY user_id
                ORDER BY log_count DESC
                LIMIT 10
            """).fetchall()
            
            for i, (user_id, count) in enumerate(results, 1):
                print(f"  {i:2d}. User {user_id:5d} - {count:3,} logs")
            
            # Public vs Private logs
            print("\n🔓 Public vs Private Logs:")
            results = self.conn.execute("""
                SELECT 
                    CASE WHEN is_public = 1 THEN 'Public' ELSE 'Private' END as visibility,
                    COUNT(*) as count
                FROM travel_logs
                GROUP BY is_public
            """).fetchall()
            
            for visibility, count in results:
                print(f"  {visibility:10s} {count:5,} logs")
            
            # Recent activity
            print("\n📅 Recent Travel Logs:")
            results = self.conn.execute("""
                SELECT title, user_id, created_at
                FROM travel_logs
                ORDER BY created_at DESC
                LIMIT 5
            """).fetchall()
            
            for i, (title, user_id, created) in enumerate(results, 1):
                print(f"  {i}. '{title}' by User {user_id} - {created}")
            
        except Exception as e:
            print(f"⚠️  Travel logs analysis unavailable")
            print(f"💡 Save some routes in the app to see this data!")
    
    def generate_insights(self):
        """Generate interesting insights"""
        print("\n💡 INTERESTING INSIGHTS")
        print("=" * 60)
        
        # Longest route
        result = self.conn.execute("""
            SELECT 
                ao.iata_code as origin,
                ao.city as origin_city,
                ad.iata_code as dest,
                ad.city as dest_city,
                r.distance_km,
                r.duration_min
            FROM routes r
            JOIN airports ao ON r.origin_airport_id = ao.id
            JOIN airports ad ON r.destination_airport_id = ad.id
            ORDER BY r.distance_km DESC
            LIMIT 1
        """).fetchone()
        
        if result:
            print(f"🌍 Longest Route:")
            print(f"  {result[0]} ({result[1]}) → {result[2]} ({result[3]})")
            print(f"  Distance: {float(result[4]):,.2f} km")
            print(f"  Duration: {result[5]:.0f} minutes ({result[5]/60:.1f} hours)")
        
        # Shortest route
        result = self.conn.execute("""
            SELECT 
                ao.iata_code as origin,
                ao.city as origin_city,
                ad.iata_code as dest,
                ad.city as dest_city,
                r.distance_km,
                r.duration_min
            FROM routes r
            JOIN airports ao ON r.origin_airport_id = ao.id
            JOIN airports ad ON r.destination_airport_id = ad.id
            WHERE r.distance_km > 0
            ORDER BY r.distance_km ASC
            LIMIT 1
        """).fetchone()
        
        if result:
            print(f"\n✈️  Shortest Route:")
            print(f"  {result[0]} ({result[1]}) → {result[2]} ({result[3]})")
            print(f"  Distance: {float(result[4]):,.2f} km")
            print(f"  Duration: {result[5]:.0f} minutes")
        
        # Average speed
        result = self.conn.execute("""
            SELECT AVG(distance_km / (duration_min / 60.0)) as avg_speed
            FROM routes
            WHERE duration_min > 0
        """).fetchone()
        
        if result:
            print(f"\n🚀 Average Flight Speed: {result[0]:.0f} km/h")
    
    def export_to_csv(self, output_dir: str = "./analytics_export"):
        """Export analytics to CSV"""
        print(f"\n📤 Exporting analytics to CSV...")
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Export routes by distance
        self.conn.execute(f"""
            COPY (
                SELECT 
                    ao.iata_code as origin,
                    ao.city as origin_city,
                    ao.country as origin_country,
                    ad.iata_code as destination,
                    ad.city as dest_city,
                    ad.country as dest_country,
                    r.distance_km,
                    r.duration_min
                FROM routes r
                JOIN airports ao ON r.origin_airport_id = ao.id
                JOIN airports ad ON r.destination_airport_id = ad.id
                ORDER BY r.distance_km DESC
                LIMIT 1000
            ) TO '{output_dir}/top_routes_by_distance.csv' (HEADER, DELIMITER ',')
        """)
        print(f"✅ Exported: {output_dir}/top_routes_by_distance.csv")
        
        # Export airports by country
        self.conn.execute(f"""
            COPY (
                SELECT country, COUNT(*) as airport_count
                FROM airports
                GROUP BY country
                ORDER BY airport_count DESC
            ) TO '{output_dir}/airports_by_country.csv' (HEADER, DELIMITER ',')
        """)
        print(f"✅ Exported: {output_dir}/airports_by_country.csv")
        
        print(f"\n✅ All exports completed in: {output_dir}/")
    
    def generate_report(self):
        """Generate full analytics report"""
        print("\n" + "=" * 60)
        print("📈 AERO MELODY ANALYTICS REPORT")
        print("=" * 60)
        print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Database: {self.db_path}")
        print("=" * 60)
        
        self.analyze_routes()
        self.analyze_airports()
        self.analyze_travel_logs()
        self.generate_insights()
        
        print("\n" + "=" * 60)
        print("✅ Analytics Complete!")
        print("=" * 60)
    
    def close(self):
        """Close connection"""
        self.conn.close()
        print("\n✅ DuckDB connection closed")


def main():
    """Main execution"""
    print("\n" + "=" * 60)
    print("🦆 AERO MELODY DUCKDB ANALYTICS")
    print("=" * 60)
    
    analytics = AeroMelodyAnalytics()
    analytics.import_from_mariadb()
    analytics.generate_report()
    analytics.export_to_csv()
    analytics.close()


if __name__ == "__main__":
    main()
