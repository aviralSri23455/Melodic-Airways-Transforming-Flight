-- Fix ETL issues: coordinate precision and ensure JSON column type
-- Run this before executing the ETL script

USE melody_aero;

-- Drop and recreate tables to ensure correct schema
DROP TABLE IF EXISTS routes;
DROP TABLE IF EXISTS airports;

-- Recreate airports table with correct precision
CREATE TABLE airports (
    id INT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    city VARCHAR(255),
    country VARCHAR(255) NOT NULL,
    iata_code VARCHAR(3),
    icao_code VARCHAR(4) UNIQUE,
    latitude DECIMAL(12,10) NOT NULL,
    longitude DECIMAL(13,10) NOT NULL,
    altitude INT,
    timezone VARCHAR(50),
    dst VARCHAR(10),
    tz_database_time_zone VARCHAR(100),
    type VARCHAR(50),
    source VARCHAR(50),
    INDEX idx_iata (iata_code),
    INDEX idx_country (country),
    INDEX idx_coordinates (latitude, longitude)
) ENGINE=InnoDB;

-- Recreate routes table with proper JSON column
CREATE TABLE routes (
    id INT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
    origin_airport_id INT UNSIGNED NOT NULL,
    destination_airport_id INT UNSIGNED NOT NULL,
    distance_km DECIMAL(10,2),
    duration_min INT,
    route_embedding JSON,
    INDEX idx_origin_dest (origin_airport_id, destination_airport_id),
    INDEX idx_distance (distance_km),
    CONSTRAINT fk_routes_origin FOREIGN KEY (origin_airport_id)
        REFERENCES airports(id) ON DELETE CASCADE ON UPDATE CASCADE,
    CONSTRAINT fk_routes_destination FOREIGN KEY (destination_airport_id)
        REFERENCES airports(id) ON DELETE CASCADE ON UPDATE CASCADE
) ENGINE=InnoDB;

-- Verify table structures
SHOW CREATE TABLE airports;
SHOW CREATE TABLE routes;

SELECT 'Tables recreated successfully with correct schema' as status;
