-- Aero Melody Database Initialization Script
-- This script runs automatically when the Galera cluster starts

-- Create database if it doesn't exist
CREATE DATABASE IF NOT EXISTS aero_melody CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;

-- Use the database
USE aero_melody;

-- Grant privileges to aero_user
GRANT ALL PRIVILEGES ON aero_melody.* TO 'aero_user'@'%';
GRANT ALL PRIVILEGES ON aero_melody.* TO 'aero_user'@'localhost';

-- Flush privileges
FLUSH PRIVILEGES;

-- Log initialization
SELECT 'Aero Melody database initialized successfully!' AS Status;
