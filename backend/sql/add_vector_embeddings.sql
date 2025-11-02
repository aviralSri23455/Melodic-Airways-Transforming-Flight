-- Add vector embedding support to OpenFlights dataset
-- This script adds columns for storing vector embeddings and complexity metrics

-- Add embedding column to routes table
ALTER TABLE routes 
ADD COLUMN IF NOT EXISTS route_embedding JSON COMMENT 'Vector embedding for similarity search (128D)';

-- Add melodic complexity metrics to routes
ALTER TABLE routes
ADD COLUMN IF NOT EXISTS melodic_complexity FLOAT COMMENT 'Melodic complexity score (0-1)',
ADD COLUMN IF NOT EXISTS harmonic_complexity FLOAT COMMENT 'Harmonic complexity score (0-1)',
ADD COLUMN IF NOT EXISTS rhythmic_complexity FLOAT COMMENT 'Rhythmic complexity score (0-1)';

-- Add embedding column to airports table
ALTER TABLE airports 
ADD COLUMN IF NOT EXISTS airport_embedding JSON COMMENT 'Vector embedding for airport characteristics (128D)';

-- Add airport characteristics for embedding generation
ALTER TABLE airports
ADD COLUMN IF NOT EXISTS hub_score FLOAT COMMENT 'Hub importance score (0-1)',
ADD COLUMN IF NOT EXISTS connectivity_score FLOAT COMMENT 'Connectivity score based on routes (0-1)';

-- Create index on route_embedding for faster JSON queries
CREATE INDEX IF NOT EXISTS idx_routes_embedding ON routes((CAST(route_embedding AS CHAR(255))));

-- Create index on complexity metrics
CREATE INDEX IF NOT EXISTS idx_routes_melodic_complexity ON routes(melodic_complexity);
CREATE INDEX IF NOT EXISTS idx_routes_harmonic_complexity ON routes(harmonic_complexity);
CREATE INDEX IF NOT EXISTS idx_routes_rhythmic_complexity ON routes(rhythmic_complexity);

-- Create index on airport_embedding
CREATE INDEX IF NOT EXISTS idx_airports_embedding ON airports((CAST(airport_embedding AS CHAR(255))));

-- Create view for routes with embeddings
CREATE OR REPLACE VIEW routes_with_embeddings AS
SELECT 
    r.id,
    r.origin_airport_id,
    r.destination_airport_id,
    r.distance_km,
    r.stops,
    r.route_embedding,
    r.melodic_complexity,
    r.harmonic_complexity,
    r.rhythmic_complexity,
    ao.iata_code as origin_code,
    ao.name as origin_name,
    ao.city as origin_city,
    ao.country as origin_country,
    ad.iata_code as dest_code,
    ad.name as dest_name,
    ad.city as dest_city,
    ad.country as dest_country
FROM routes r
JOIN airports ao ON r.origin_airport_id = ao.id
JOIN airports ad ON r.destination_airport_id = ad.id
WHERE r.route_embedding IS NOT NULL;

-- Create view for high complexity routes
CREATE OR REPLACE VIEW complex_routes AS
SELECT 
    r.id,
    r.origin_airport_id,
    r.destination_airport_id,
    r.distance_km,
    r.stops,
    r.melodic_complexity,
    r.harmonic_complexity,
    r.rhythmic_complexity,
    (r.melodic_complexity * 0.4 + r.harmonic_complexity * 0.3 + r.rhythmic_complexity * 0.3) as overall_complexity,
    ao.iata_code as origin_code,
    ad.iata_code as dest_code
FROM routes r
JOIN airports ao ON r.origin_airport_id = ao.id
JOIN airports ad ON r.destination_airport_id = ad.id
WHERE r.melodic_complexity IS NOT NULL
AND (r.melodic_complexity * 0.4 + r.harmonic_complexity * 0.3 + r.rhythmic_complexity * 0.3) > 0.7
ORDER BY overall_complexity DESC;

-- Create statistics table for embedding analytics
CREATE TABLE IF NOT EXISTS embedding_statistics (
    id INT AUTO_INCREMENT PRIMARY KEY,
    total_routes INT,
    routes_with_embeddings INT,
    embedding_coverage FLOAT,
    avg_melodic_complexity FLOAT,
    avg_harmonic_complexity FLOAT,
    avg_rhythmic_complexity FLOAT,
    faiss_index_size INT,
    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
);

-- Insert initial statistics
INSERT INTO embedding_statistics (
    total_routes,
    routes_with_embeddings,
    embedding_coverage,
    avg_melodic_complexity,
    avg_harmonic_complexity,
    avg_rhythmic_complexity,
    faiss_index_size
)
SELECT 
    COUNT(*) as total_routes,
    COUNT(route_embedding) as routes_with_embeddings,
    COUNT(route_embedding) / COUNT(*) * 100 as embedding_coverage,
    AVG(melodic_complexity) as avg_melodic_complexity,
    AVG(harmonic_complexity) as avg_harmonic_complexity,
    AVG(rhythmic_complexity) as avg_rhythmic_complexity,
    0 as faiss_index_size
FROM routes;

-- Create stored procedure to update embedding statistics
DELIMITER //

CREATE OR REPLACE PROCEDURE update_embedding_statistics()
BEGIN
    INSERT INTO embedding_statistics (
        total_routes,
        routes_with_embeddings,
        embedding_coverage,
        avg_melodic_complexity,
        avg_harmonic_complexity,
        avg_rhythmic_complexity,
        faiss_index_size
    )
    SELECT 
        COUNT(*) as total_routes,
        COUNT(route_embedding) as routes_with_embeddings,
        COUNT(route_embedding) / COUNT(*) * 100 as embedding_coverage,
        AVG(melodic_complexity) as avg_melodic_complexity,
        AVG(harmonic_complexity) as avg_harmonic_complexity,
        AVG(rhythmic_complexity) as avg_rhythmic_complexity,
        (SELECT faiss_index_size FROM embedding_statistics ORDER BY id DESC LIMIT 1) as faiss_index_size
    FROM routes;
END //

DELIMITER ;

-- Show results
SELECT 'Vector embedding schema created successfully!' as status;
SELECT 
    COUNT(*) as total_routes,
    COUNT(route_embedding) as routes_with_embeddings,
    ROUND(COUNT(route_embedding) / COUNT(*) * 100, 2) as coverage_percent
FROM routes;
