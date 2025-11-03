import pymysql

try:
    conn = pymysql.connect(
        host='localhost',
        user='root',
        password='SkyMusiC@2026!',
        port=3306
    )
    cursor = conn.cursor()
    cursor.execute("SHOW DATABASES LIKE 'melody_aero'")
    result = cursor.fetchone()
    
    if result:
        print("✅ Database 'melody_aero' exists")
    else:
        print("❌ Database 'melody_aero' does not exist")
        print("Creating database...")
        cursor.execute("CREATE DATABASE melody_aero")
        print("✅ Database created successfully")
    
    conn.close()
except Exception as e:
    print(f"❌ Error: {e}")
