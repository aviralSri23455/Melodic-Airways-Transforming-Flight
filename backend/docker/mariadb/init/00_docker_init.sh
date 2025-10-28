#!/bin/bash

# Wait for MariaDB to be ready
until mysql -u root -p${MYSQL_ROOT_PASSWORD} -e "SELECT 1" &> /dev/null; do
    echo "Waiting for MariaDB..."
    sleep 2
done

echo "MariaDB is ready. Running initialization..."

# Create database if it doesn't exist
mysql -u root -p${MYSQL_ROOT_PASSWORD} -e "CREATE DATABASE IF NOT EXISTS ${MYSQL_DATABASE};"

# Grant privileges to the user
mysql -u root -p${MYSQL_ROOT_PASSWORD} -e "GRANT ALL PRIVILEGES ON ${MYSQL_DATABASE}.* TO '${MYSQL_USER}'@'%' IDENTIFIED BY '${MYSQL_PASSWORD}';"

# Enable vector extension
mysql -u root -p${MYSQL_ROOT_PASSWORD} ${MYSQL_DATABASE} -e "INSTALL SONAME 'vector';"

# Enable columnstore
mysql -u root -p${MYSQL_ROOT_PASSWORD} ${MYSQL_DATABASE} -e "INSTALL SONAME 'columnstore';"

# Enable galera
mysql -u root -p${MYSQL_ROOT_PASSWORD} ${MYSQL_DATABASE} -e "INSTALL SONAME 'galera';"

# Create vector similarity function
mysql -u root -p${MYSQL_ROOT_PASSWORD} ${MYSQL_DATABASE} -e "
CREATE FUNCTION IF NOT EXISTS vector_cosine_similarity(a JSON, b JSON)
RETURNS FLOAT
DETERMINISTIC
BEGIN
    DECLARE similarity FLOAT DEFAULT 0.0;
    SET similarity = JSON_EXTRACT(a, '$.cosine_similarity', JSON_EXTRACT(b, '$'));
    RETURN similarity;
END;
"

# Create vector distance function
mysql -u root -p${MYSQL_ROOT_PASSWORD} ${MYSQL_DATABASE} -e "
CREATE FUNCTION IF NOT EXISTS vector_euclidean_distance(a VECTOR(512), b VECTOR(512))
RETURNS FLOAT
DETERMINISTIC
BEGIN
    RETURN SQRT(SUM(POW(a - b, 2)));
END;
"

echo "MariaDB initialization completed."
