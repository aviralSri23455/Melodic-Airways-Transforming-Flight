-- Aero Melody Database Schema - FREE MariaDB Features Only
-- Docker initialization script with JSON-based embeddings

USE aero_melody;

-- Drop all tables in correct order (due to foreign key dependencies)
DROP TABLE IF EXISTS composition_remixes;
DROP TABLE IF EXISTS user_activities;
DROP TABLE IF EXISTS collaboration_sessions;
DROP TABLE IF EXISTS user_collections;
DROP TABLE IF EXISTS user_datasets;
DROP TABLE IF EXISTS music_compositions;
DROP TABLE IF EXISTS routes;
DROP TABLE IF EXISTS airports;
DROP TABLE IF EXISTS users;

-- 1. USERS TABLE (no foreign keys needed here)
CREATE TABLE users (
    id INT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
    username VARCHAR(50) UNIQUE NOT NULL,
    email VARCHAR(255) UNIQUE NOT NULL,
    hashed_password VARCHAR(255) NOT NULL,
    role VARCHAR(20) DEFAULT 'user',
    is_active TINYINT(1) DEFAULT 1,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    INDEX idx_username (username),
    INDEX idx_email (email)
) ENGINE=InnoDB;

-- 2. AIRPORTS TABLE (no foreign keys needed here)
CREATE TABLE airports (
    id INT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    city VARCHAR(255),
    country VARCHAR(255) NOT NULL,
    iata_code VARCHAR(3) UNIQUE,
    icao_code VARCHAR(4) UNIQUE,
    latitude DECIMAL(10,8) NOT NULL,
    longitude DECIMAL(11,8) NOT NULL,
    altitude INT,
    timezone VARCHAR(50),
    dst VARCHAR(1),
    tz_database_time_zone VARCHAR(100),
    type VARCHAR(50),
    source VARCHAR(50),
    INDEX idx_iata (iata_code),
    INDEX idx_country (country),
    INDEX idx_coordinates (latitude, longitude)
) ENGINE=InnoDB;

-- 3. ROUTES TABLE (with foreign keys to airports and JSON embedding)
CREATE TABLE routes (
    id INT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
    origin_airport_id INT UNSIGNED NOT NULL,
    destination_airport_id INT UNSIGNED NOT NULL,
    distance_km DECIMAL(10,2),
    duration_min INT,
    route_embedding JSON,
    INDEX idx_origin_dest (origin_airport_id, destination_airport_id),
    INDEX idx_distance (distance_km),
    INDEX idx_route_embedding (route_embedding(255)),
    CONSTRAINT fk_routes_origin FOREIGN KEY (origin_airport_id)
        REFERENCES airports(id) ON DELETE CASCADE ON UPDATE CASCADE,
    CONSTRAINT fk_routes_destination FOREIGN KEY (destination_airport_id)
        REFERENCES airports(id) ON DELETE CASCADE ON UPDATE CASCADE
) ENGINE=InnoDB;

-- 4. MUSIC COMPOSITIONS TABLE (with foreign keys to routes and users)
CREATE TABLE music_compositions (
    id INT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
    route_id INT UNSIGNED NOT NULL,
    user_id INT UNSIGNED,
    title VARCHAR(255),
    genre VARCHAR(100),
    tempo INT NOT NULL,
    pitch FLOAT NOT NULL,
    harmony FLOAT NOT NULL,
    midi_path VARCHAR(500) NOT NULL,
    complexity_score FLOAT,
    harmonic_richness FLOAT,
    duration_seconds INT,
    unique_notes INT,
    musical_key VARCHAR(2),
    scale VARCHAR(20),
    music_vector JSON,
    is_public TINYINT(1) DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    INDEX idx_route_user (route_id, user_id),
    INDEX idx_complexity (complexity_score),
    INDEX idx_music_genre (genre),
    INDEX idx_music_public (is_public),
    INDEX idx_music_title (title),
    INDEX idx_music_vector_harmonic (music_vector(255)),
    FULLTEXT INDEX idx_music_search (title, genre),
    CONSTRAINT fk_music_route FOREIGN KEY (route_id)
        REFERENCES routes(id) ON DELETE CASCADE ON UPDATE CASCADE,
    CONSTRAINT fk_music_user FOREIGN KEY (user_id)
        REFERENCES users(id) ON DELETE SET NULL ON UPDATE CASCADE
) ENGINE=InnoDB;

-- 5. USER DATASETS TABLE
CREATE TABLE user_datasets (
    id INT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
    user_id INT UNSIGNED NOT NULL,
    name VARCHAR(255) NOT NULL,
    route_data JSON NOT NULL,
    metadata JSON,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    CONSTRAINT fk_datasets_user FOREIGN KEY (user_id)
        REFERENCES users(id) ON DELETE CASCADE ON UPDATE CASCADE,
    INDEX idx_user_datasets_user_id (user_id),
    INDEX idx_user_datasets_created (created_at),
    INDEX idx_user_datasets_name (name)
) ENGINE=InnoDB;

-- 6. USER COLLECTIONS TABLE
CREATE TABLE user_collections (
    id INT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
    user_id INT UNSIGNED NOT NULL,
    name VARCHAR(255) NOT NULL,
    description TEXT,
    composition_ids JSON,
    tags JSON,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT fk_collections_user FOREIGN KEY (user_id)
        REFERENCES users(id) ON DELETE CASCADE ON UPDATE CASCADE,
    INDEX idx_collections_user_id (user_id),
    INDEX idx_collections_name (name),
    INDEX idx_collections_created (created_at)
) ENGINE=InnoDB;

-- 7. COLLABORATION SESSIONS TABLE
CREATE TABLE collaboration_sessions (
    id INT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
    creator_id INT UNSIGNED NOT NULL,
    composition_id INT UNSIGNED,
    session_state JSON,
    participants JSON,
    is_active TINYINT(1) DEFAULT 1,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    expires_at TIMESTAMP,
    CONSTRAINT fk_sessions_creator FOREIGN KEY (creator_id)
        REFERENCES users(id) ON DELETE CASCADE ON UPDATE CASCADE,
    CONSTRAINT fk_sessions_composition FOREIGN KEY (composition_id)
        REFERENCES music_compositions(id) ON DELETE SET NULL ON UPDATE CASCADE,
    INDEX idx_sessions_creator (creator_id),
    INDEX idx_sessions_active (is_active),
    INDEX idx_sessions_expires (expires_at),
    INDEX idx_sessions_composition (composition_id)
) ENGINE=InnoDB;

-- 8. COMPOSITION REMIXES TABLE
CREATE TABLE composition_remixes (
    id INT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
    original_composition_id INT UNSIGNED NOT NULL,
    remix_composition_id INT UNSIGNED NOT NULL,
    remix_type ENUM('variation', 'genre_change', 'tempo_change', 'full_remix'),
    attribution_data JSON,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT fk_remixes_original FOREIGN KEY (original_composition_id)
        REFERENCES music_compositions(id) ON DELETE CASCADE ON UPDATE CASCADE,
    CONSTRAINT fk_remixes_remix FOREIGN KEY (remix_composition_id)
        REFERENCES music_compositions(id) ON DELETE CASCADE ON UPDATE CASCADE,
    INDEX idx_remixes_original (original_composition_id),
    INDEX idx_remixes_remix (remix_composition_id),
    INDEX idx_remixes_type (remix_type),
    UNIQUE KEY unique_remix_pair (original_composition_id, remix_composition_id)
) ENGINE=InnoDB;

-- 9. USER ACTIVITIES TABLE
CREATE TABLE user_activities (
    id INT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
    user_id INT UNSIGNED NOT NULL,
    activity_type ENUM(
        'composition_created', 'composition_updated', 'composition_deleted',
        'dataset_created', 'dataset_updated', 'dataset_deleted',
        'collection_created', 'collection_updated', 'collection_deleted',
        'remix_created', 'remix_updated',
        'collaboration_joined', 'collaboration_left', 'collaboration_updated',
        'search_performed', 'profile_updated', 'login', 'logout'
    ) NOT NULL,
    target_id INT UNSIGNED,
    target_type VARCHAR(50),
    activity_data JSON,
    ip_address VARCHAR(45),
    user_agent TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT fk_activities_user FOREIGN KEY (user_id)
        REFERENCES users(id) ON DELETE CASCADE ON UPDATE CASCADE,
    INDEX idx_activities_user_id (user_id),
    INDEX idx_activities_type (activity_type),
    INDEX idx_activities_target (target_id, target_type),
    INDEX idx_activities_created (created_at),
    INDEX idx_activities_user_created (user_id, created_at)
) ENGINE=InnoDB;

-- 10. UPDATE music_compositions to add dataset_id foreign key
ALTER TABLE music_compositions
ADD COLUMN dataset_id INT UNSIGNED AFTER user_id,
ADD CONSTRAINT fk_music_dataset FOREIGN KEY (dataset_id)
    REFERENCES user_datasets(id) ON DELETE SET NULL ON UPDATE CASCADE;

-- Verify ALL tables were created
SHOW TABLES;

-- Show table structures to confirm foreign keys and JSON columns
SHOW CREATE TABLE routes;
SHOW CREATE TABLE music_compositions;
SHOW CREATE TABLE user_datasets;

-- Verify JSON columns work
SELECT 'JSON columns and free MariaDB features working correctly' as init_status;
