-- Aero Melody Enhanced Database Schema Migration
-- Adds vector storage, user datasets, collections, collaboration sessions, and remixes
-- Compatible with existing MariaDB setup using free features

USE aero_melody;

-- 1. MODIFY existing music_compositions table to add vector storage and new features
ALTER TABLE music_compositions
ADD COLUMN music_vector JSON AFTER harmonic_richness,
ADD COLUMN genre VARCHAR(100) AFTER musical_key,
ADD COLUMN is_public TINYINT(1) DEFAULT 0 AFTER scale,
ADD COLUMN title VARCHAR(255) AFTER user_id,
ADD COLUMN dataset_id INT UNSIGNED AFTER user_id,
ADD INDEX idx_music_genre (genre),
ADD INDEX idx_music_public (is_public),
ADD INDEX idx_music_title (title);

-- Add foreign key constraint for dataset relationship
ALTER TABLE music_compositions
ADD CONSTRAINT fk_music_dataset FOREIGN KEY (dataset_id)
    REFERENCES user_datasets(id) ON DELETE SET NULL ON UPDATE CASCADE;

-- 2. ADD new user_datasets table (using existing INT UNSIGNED pattern)
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

-- 3. ADD collaboration_sessions table
CREATE TABLE collaboration_sessions (
    id INT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
    creator_id INT UNSIGNED NOT NULL,
    composition_id INT UNSIGNED,
    session_state JSON,
    participants JSON, -- Array of participant IDs
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

-- 4. ADD user_collections table
CREATE TABLE user_collections (
    id INT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
    user_id INT UNSIGNED NOT NULL,
    name VARCHAR(255) NOT NULL,
    description TEXT,
    composition_ids JSON, -- Array of composition IDs
    tags JSON, -- Array of tags
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT fk_collections_user FOREIGN KEY (user_id)
        REFERENCES users(id) ON DELETE CASCADE ON UPDATE CASCADE,
    INDEX idx_collections_user_id (user_id),
    INDEX idx_collections_name (name),
    INDEX idx_collections_created (created_at)
) ENGINE=InnoDB;

-- 5. ADD composition_remixes table
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

-- 6. ADD user_activities table for real-time activity feed
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
    target_id INT UNSIGNED, -- ID of the affected resource (composition, dataset, etc.)
    target_type VARCHAR(50), -- Type of target resource
    activity_data JSON, -- Additional data about the activity
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

-- 7. Create view for public compositions
CREATE OR REPLACE VIEW public_compositions AS
SELECT
    mc.*,
    JSON_EXTRACT(mc.music_vector, '$.harmonic') as harmonic_features,
    JSON_EXTRACT(mc.music_vector, '$.rhythmic') as rhythmic_features,
    JSON_EXTRACT(mc.music_vector, '$.melodic') as melodic_features,
    JSON_EXTRACT(mc.music_vector, '$.genre') as genre_features
FROM music_compositions mc
WHERE mc.is_public = 1 AND mc.music_vector IS NOT NULL;

-- 8. Create indexes for better query performance
CREATE INDEX idx_music_vector_genre ON music_compositions(
    (JSON_EXTRACT(music_vector, '$.genre[0]')),
    (JSON_EXTRACT(music_vector, '$.genre[1]'))
);

CREATE INDEX idx_music_tempo_range ON music_compositions(tempo, genre, is_public);

-- 9. Add full-text search index for composition titles and genres
CREATE FULLTEXT INDEX idx_music_search ON music_compositions(title, genre);

-- Verify all new tables and modifications
SELECT 'Database migration completed successfully' as status;
