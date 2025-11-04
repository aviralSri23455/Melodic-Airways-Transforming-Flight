-- Fix coordinate precision for airports table
-- This migration increases the precision of latitude and longitude columns

USE melody_aero;

-- Modify airports table to increase coordinate precision
ALTER TABLE airports 
    MODIFY COLUMN latitude DECIMAL(12,10) NOT NULL,
    MODIFY COLUMN longitude DECIMAL(13,10) NOT NULL;

-- Verify the changes
DESCRIBE airports;

SELECT 'Coordinate precision updated successfully' as status;
