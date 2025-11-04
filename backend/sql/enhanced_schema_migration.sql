-- Aero Melody Enhanced Database Schema Migration
-- IMPORTANT: Run this AFTER create_tables.sql has been executed
-- This script ONLY adds enhancements (views, additional indexes) to existing tables
-- Using only FREE MariaDB features: JSON storage, Full-text search, and application-level real-time features
-- Compatible with standard MariaDB without paid extensions

USE melody_aero;

-- ============================================================================
-- SECTION 1: ADD ENHANCED INDEXES FOR JSON-BASED SIMILARITY SEARCH
-- ============================================================================

-- Check if tables exist before adding indexes
SET @tables_exist = (SELECT COUNT(*) FROM information_schema.tables 
                     WHERE table_schema = 'melody_aero' 
                     AND table_name IN ('music_compositions', 'routes', 'user_datasets', 
                                       'collaboration_sessions', 'user_collections', 
                                       'composition_remixes', 'user_activities'));

-- Only proceed if all base tables exist (created by create_tables.sql)
-- If tables don't exist, this script will fail gracefully

-- 1. Add JSON extraction indexes for music_compositions (if not already present)
-- These indexes improve performance for similarity searches on music vectors
-- MariaDB-compatible approach: Use generated columns instead of functional indexes

-- Drop old indexes if they exist from previous runs
DROP INDEX IF EXISTS idx_music_vector_harmonic ON music_compositions;
DROP INDEX IF EXISTS idx_music_vector_rhythmic ON music_compositions;
DROP INDEX IF EXISTS idx_music_vector_melodic ON music_compositions;
DROP INDEX IF EXISTS idx_music_vector_genre ON music_compositions;
DROP INDEX IF EXISTS idx_music_tempo_range ON music_compositions;
DROP INDEX IF EXISTS idx_music_complexity ON music_compositions;

-- Drop generated columns if they exist (for clean re-run)
ALTER TABLE music_compositions DROP COLUMN IF EXISTS harmonic_feature;
ALTER TABLE music_compositions DROP COLUMN IF EXISTS rhythmic_feature;
ALTER TABLE music_compositions DROP COLUMN IF EXISTS melodic_feature;
ALTER TABLE music_compositions DROP COLUMN IF EXISTS genre_feature;

-- Add generated columns for JSON features (MariaDB compatible)
ALTER TABLE music_compositions
ADD COLUMN harmonic_feature DECIMAL(10,6)
    AS (CAST(JSON_UNQUOTE(JSON_EXTRACT(music_vector, '$.harmonic')) AS DECIMAL(10,6)))
    STORED;

ALTER TABLE music_compositions
ADD COLUMN rhythmic_feature DECIMAL(10,6)
    AS (CAST(JSON_UNQUOTE(JSON_EXTRACT(music_vector, '$.rhythmic')) AS DECIMAL(10,6)))
    STORED;

ALTER TABLE music_compositions
ADD COLUMN melodic_feature DECIMAL(10,6)
    AS (CAST(JSON_UNQUOTE(JSON_EXTRACT(music_vector, '$.melodic')) AS DECIMAL(10,6)))
    STORED;

ALTER TABLE music_compositions
ADD COLUMN genre_feature DECIMAL(10,6)
    AS (CAST(JSON_UNQUOTE(JSON_EXTRACT(music_vector, '$.genre_score')) AS DECIMAL(10,6)))
    STORED;

-- Create indexes on the generated columns
CREATE INDEX idx_music_vector_harmonic ON music_compositions(harmonic_feature);
CREATE INDEX idx_music_vector_rhythmic ON music_compositions(rhythmic_feature);
CREATE INDEX idx_music_vector_melodic ON music_compositions(melodic_feature);
CREATE INDEX idx_music_vector_genre_feature ON music_compositions(genre_feature);

-- 2. Add composite indexes for common query patterns
DROP INDEX IF EXISTS idx_music_tempo_genre_public ON music_compositions;
DROP INDEX IF EXISTS idx_music_complexity_harmonic ON music_compositions;
DROP INDEX IF EXISTS idx_music_created_public ON music_compositions;

CREATE INDEX idx_music_tempo_genre_public ON music_compositions(tempo, genre, is_public);
CREATE INDEX idx_music_complexity_harmonic ON music_compositions(complexity_score, harmonic_richness);
CREATE INDEX idx_music_created_public ON music_compositions(created_at, is_public);

-- 3. Add route embedding indexes for similarity search (MariaDB compatible)
DROP INDEX IF EXISTS idx_routes_embedding_first ON routes;

-- Drop generated column if it exists (for clean re-run)
ALTER TABLE routes DROP COLUMN IF EXISTS embedding_first;

-- Add generated column for first embedding value
ALTER TABLE routes
ADD COLUMN embedding_first DECIMAL(10,6)
    AS (CAST(JSON_UNQUOTE(JSON_EXTRACT(route_embedding, '$[0]')) AS DECIMAL(10,6)))
    STORED;

-- Create index on the generated column
CREATE INDEX idx_routes_embedding_first ON routes(embedding_first);

-- ============================================================================
-- SECTION 2: CREATE OPTIMIZED VIEWS FOR REAL-TIME FEATURES
-- ============================================================================

-- View 1: Public compositions with extracted JSON features for faster querying
DROP VIEW IF EXISTS public_compositions;
CREATE VIEW public_compositions AS
SELECT
    mc.id,
    mc.route_id,
    mc.user_id,
    mc.title,
    mc.genre,
    mc.tempo,
    mc.pitch,
    mc.harmony,
    mc.midi_path,
    mc.complexity_score,
    mc.harmonic_richness,
    mc.duration_seconds,
    mc.unique_notes,
    mc.musical_key,
    mc.scale,
    mc.is_public,
    mc.created_at,
    mc.updated_at,
    mc.harmonic_feature,
    mc.rhythmic_feature,
    mc.melodic_feature,
    mc.genre_feature
FROM music_compositions mc
WHERE mc.is_public = 1 AND mc.music_vector IS NOT NULL;

-- View 2: Active collaboration sessions with participant details
DROP VIEW IF EXISTS active_collaborations;
CREATE VIEW active_collaborations AS
SELECT
    cs.id,
    cs.creator_id,
    cs.composition_id,
    cs.is_active,
    cs.created_at,
    cs.expires_at,
    JSON_LENGTH(cs.participants) as participant_count,
    mc.title as composition_title,
    mc.genre as composition_genre,
    u.username as creator_username
FROM collaboration_sessions cs
LEFT JOIN music_compositions mc ON cs.composition_id = mc.id
LEFT JOIN users u ON cs.creator_id = u.id
WHERE cs.is_active = 1 AND (cs.expires_at IS NULL OR cs.expires_at > NOW());

-- View 3: User activity summary for real-time dashboard
DROP VIEW IF EXISTS user_activity_summary;
CREATE VIEW user_activity_summary AS
SELECT
    ua.user_id,
    u.username,
    ua.activity_type,
    COUNT(*) as activity_count,
    MAX(ua.created_at) as last_activity,
    DATE(ua.created_at) as activity_date
FROM user_activities ua
JOIN users u ON ua.user_id = u.id
WHERE ua.created_at >= DATE_SUB(NOW(), INTERVAL 30 DAY)
GROUP BY ua.user_id, u.username, ua.activity_type, DATE(ua.created_at);

-- View 4: Popular compositions by remix count
DROP VIEW IF EXISTS popular_compositions;
CREATE VIEW popular_compositions AS
SELECT
    mc.id,
    mc.title,
    mc.genre,
    mc.tempo,
    mc.complexity_score,
    mc.created_at,
    mc.user_id,
    u.username,
    COUNT(DISTINCT cr.id) as remix_count,
    COUNT(DISTINCT uc.id) as collection_count
FROM music_compositions mc
LEFT JOIN composition_remixes cr ON mc.id = cr.original_composition_id
LEFT JOIN user_collections uc ON JSON_CONTAINS(uc.composition_ids, CONCAT('"', mc.id, '"'))
LEFT JOIN users u ON mc.user_id = u.id
WHERE mc.is_public = 1
GROUP BY mc.id, mc.title, mc.genre, mc.tempo, mc.complexity_score, mc.created_at, mc.user_id, u.username
HAVING remix_count > 0 OR collection_count > 0
ORDER BY remix_count DESC, collection_count DESC;

-- View 5: User collection details with composition count
DROP VIEW IF EXISTS user_collection_details;
CREATE VIEW user_collection_details AS
SELECT
    uc.id,
    uc.user_id,
    u.username,
    uc.name,
    uc.description,
    uc.created_at,
    JSON_LENGTH(uc.composition_ids) as composition_count,
    JSON_LENGTH(uc.tags) as tag_count
FROM user_collections uc
JOIN users u ON uc.user_id = u.id;

-- ============================================================================
-- SECTION 3: ADD PERFORMANCE INDEXES FOR REAL-TIME QUERIES
-- ============================================================================

-- Indexes for collaboration features
DROP INDEX IF EXISTS idx_sessions_active_expires ON collaboration_sessions;
CREATE INDEX idx_sessions_active_expires ON collaboration_sessions(is_active, expires_at);

-- Indexes for activity tracking
DROP INDEX IF EXISTS idx_activities_recent ON user_activities;
CREATE INDEX idx_activities_recent ON user_activities(created_at DESC, user_id);

-- Indexes for collection queries
DROP INDEX IF EXISTS idx_collections_user_created ON user_collections;
CREATE INDEX idx_collections_user_created ON user_collections(user_id, created_at DESC);

-- Indexes for remix relationships
DROP INDEX IF EXISTS idx_remixes_created ON composition_remixes;
CREATE INDEX idx_remixes_created ON composition_remixes(created_at DESC);

-- ============================================================================
-- SECTION 4: VERIFICATION AND COMPLETION
-- ============================================================================

-- Verify all views were created
SELECT 
    TABLE_NAME as view_name,
    TABLE_TYPE
FROM information_schema.TABLES
WHERE TABLE_SCHEMA = 'melody_aero' 
AND TABLE_TYPE = 'VIEW'
ORDER BY TABLE_NAME;

-- Verify enhanced indexes were created
SELECT 
    TABLE_NAME,
    INDEX_NAME,
    INDEX_TYPE
FROM information_schema.STATISTICS
WHERE TABLE_SCHEMA = 'melody_aero'
AND INDEX_NAME LIKE 'idx_music_vector%'
OR INDEX_NAME LIKE 'idx_routes_embedding%'
OR INDEX_NAME LIKE 'idx_sessions_active%'
OR INDEX_NAME LIKE 'idx_activities_recent%'
ORDER BY TABLE_NAME, INDEX_NAME;

SELECT 'Enhanced schema migration completed successfully - FREE MariaDB features only' as status;
