-- Fix column sizes for OpenFlights data import
-- The dst column needs to be larger to accommodate all OpenFlights values
-- The latitude/longitude columns also need adjustment

USE melody_aero;

-- Alter the dst column to accommodate longer values
ALTER TABLE airports MODIFY COLUMN dst VARCHAR(10);

-- Verify the change
DESCRIBE airports;
