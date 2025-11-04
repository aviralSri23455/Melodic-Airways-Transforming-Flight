-- Fix Authentication Plugin for aero_user
-- This runs before other initialization scripts (00- prefix ensures it runs first)

-- Ensure aero_user uses mysql_native_password plugin
ALTER USER IF EXISTS 'aero_user'@'%' IDENTIFIED VIA mysql_native_password USING PASSWORD('melody_aero_db_2024');
ALTER USER IF EXISTS 'aero_user'@'localhost' IDENTIFIED VIA mysql_native_password USING PASSWORD('melody_aero_db_2024');

-- Flush privileges
FLUSH PRIVILEGES;

SELECT 'Authentication fixed - using mysql_native_password' AS Status;
