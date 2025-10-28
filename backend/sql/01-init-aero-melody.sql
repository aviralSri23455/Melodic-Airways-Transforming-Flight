-- Aero Melody Database Initialization Script
-- Simple setup for local MariaDB

-- Use the database
USE aero_melody;

-- Log initialization
SELECT 'Aero Melody database initialized successfully!' AS St

-- Grant privileges to aero_user
GRANT ALL PRIVILEGES ON aero_melody.* TO 'aero_user'@'%';

-- Flush privileges
FLUSH PRIVILEGES;

-- Log initialization
SELECT 'Aero Melody database initialized successfully!' AS Status;
