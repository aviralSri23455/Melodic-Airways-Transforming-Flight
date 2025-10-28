"""
Galera Cluster Configuration for Multi-Master Replication
Provides synchronous multi-master replication for real-time collaboration
"""

import os
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


class GaleraConfig:
    """Galera cluster configuration manager"""

    def __init__(self):
        self.config_template = """
# Galera Cluster Configuration for MariaDB
# This configuration enables synchronous multi-master replication

[mysqld]
# Basic settings
bind-address = 0.0.0.0
default_storage_engine = InnoDB
innodb_autoinc_lock_mode = 2
innodb_doublewrite = 1
innodb_flush_log_at_trx_commit = 2

# Galera settings
wsrep_on = ON
wsrep_provider = /usr/lib/galera/libgalera_smm.so
wsrep_cluster_address = gcomm://node1.example.com:4567,node2.example.com:4567,node3.example.com:4567
wsrep_cluster_name = aero_melody_cluster
wsrep_node_name = {node_name}
wsrep_node_address = {node_address}

# Synchronous replication
wsrep_sync_wait = 7

# Cluster settings
wsrep_sst_method = mariabackup
wsrep_sst_auth = galera_user:galera_password

# Flow control
wsrep_flow_control_mode = QUORUM
wsrep_flow_control_threshold = 100

# Performance
wsrep_max_ws_rows = 131072
wsrep_max_ws_size = 2147483648

# Logging
wsrep_log_conflicts = ON
wsrep_debug = ON

# Auto-increment settings for multi-master
wsrep_auto_increment_control = ON

# SSL/TLS (recommended for production)
# wsrep_ssl_cert = /path/to/cert.pem
# wsrep_ssl_key = /path/to/key.pem
# wsrep_ssl_ca = /path/to/ca.pem

# Node timeout settings
wsrep_node_timeout = PT30S
wsrep_network_timeout = PT30S
wsrep_wait_timeout = PT30S

# IST/RST settings
wsrep_ist_receive_timeout = PT30S
wsrep_ist_progress = 1

# Cache settings for performance
wsrep_provider_options = "gcache.size=1G; gcache.page_size=1G; gcache.recover=yes"

# Debugging (disable in production)
wsrep_debug = 1
"""

    def generate_node_config(self, node_name: str, node_address: str, cluster_addresses: List[str]) -> str:
        """Generate Galera configuration for a specific node"""
        cluster_address = ",".join(cluster_addresses)

        config = self.config_template.format(
            node_name=node_name,
            node_address=node_address
        )

        # Replace cluster address
        config = config.replace(
            "gcomm://node1.example.com:4567,node2.example.com:4567,node3.example.com:4567",
            f"gcomm://{cluster_address}"
        )

        return config

    def get_bootstrap_config(self) -> str:
        """Get configuration for bootstrap node"""
        return self.config_template.format(
            node_name="bootstrap_node",
            node_address="localhost:4567"
        ).replace(
            "wsrep_cluster_address = gcomm://",
            "wsrep_cluster_address = gcomm://"
        )

    def get_cluster_status_script(self) -> str:
        """Generate script to check cluster status"""
        return """
#!/bin/bash
# Galera Cluster Status Check Script

NODE_NAME="$1"
CLUSTER_ADDRESSES="$2"

echo "=== Galera Cluster Status ==="
echo "Node: $NODE_NAME"
echo "Cluster: $CLUSTER_ADDRESSES"
echo "Timestamp: $(date)"

# Check if MariaDB is running
if ! mysqladmin ping -u root -p${MYSQL_ROOT_PASSWORD} 2>/dev/null; then
    echo "❌ MariaDB is not running"
    exit 1
fi

echo "✅ MariaDB is running"

# Check cluster status
CLUSTER_STATUS=$(mysql -u root -p${MYSQL_ROOT_PASSWORD} -e "SHOW STATUS LIKE 'wsrep_cluster_status';" 2>/dev/null | grep wsrep_cluster_status | awk '{print $2}')

if [ "$CLUSTER_STATUS" = "Primary" ]; then
    echo "✅ Cluster status: $CLUSTER_STATUS"
else
    echo "❌ Cluster status: $CLUSTER_STATUS"
fi

# Get cluster size
CLUSTER_SIZE=$(mysql -u root -p${MYSQL_ROOT_PASSWORD} -e "SHOW STATUS LIKE 'wsrep_cluster_size';" 2>/dev/null | grep wsrep_cluster_size | awk '{print $2}')

echo "📊 Cluster size: $CLUSTER_SIZE nodes"

# Get connected nodes
CONNECTED_NODES=$(mysql -u root -p${MYSQL_ROOT_PASSWORD} -e "SHOW STATUS LIKE 'wsrep_connected';" 2>/dev/null | grep wsrep_connected | awk '{print $2}')

echo "🔗 Connected: $CONNECTED_NODES"

# Get local state
LOCAL_STATE=$(mysql -u root -p${MYSQL_ROOT_PASSWORD} -e "SHOW STATUS LIKE 'wsrep_local_state_comment';" 2>/dev/null | grep wsrep_local_state_comment | awk '{print $2}')

echo "🏠 Local state: $LOCAL_STATE"

echo "=== End Status ==="
"""


class GaleraManager:
    """Manager for Galera cluster operations"""

    def __init__(self):
        self.config = GaleraConfig()
        self.cluster_nodes: List[Dict[str, str]] = []
        self.is_cluster_ready = False

    def add_node(self, node_name: str, node_address: str, port: str = "4567"):
        """Add a node to the cluster configuration"""
        node = {
            "name": node_name,
            "address": node_address,
            "port": port,
            "status": "unknown"
        }
        self.cluster_nodes.append(node)
        logger.info(f"Added node {node_name} ({node_address}:{port}) to cluster")

    def generate_cluster_configs(self) -> Dict[str, str]:
        """Generate configurations for all cluster nodes"""
        if not self.cluster_nodes:
            raise ValueError("No nodes configured in cluster")

        cluster_addresses = [f"{node['address']}:{node['port']}" for node in self.cluster_nodes]

        configs = {}
        for node in self.cluster_nodes:
            config = self.config.generate_node_config(
                node_name=node['name'],
                node_address=f"{node['address']}:{node['port']}",
                cluster_addresses=cluster_addresses
            )
            configs[node['name']] = config

        return configs

    def generate_docker_compose(self) -> str:
        """Generate Docker Compose configuration for Galera cluster"""
        if not self.cluster_nodes:
            raise ValueError("No nodes configured in cluster")

        compose_config = """
version: '3.8'

services:
"""

        for i, node in enumerate(self.cluster_nodes):
            node_name = node['name']
            node_address = node['address']

            compose_config += f"""
  mariadb-{node_name}:
    image: mariadb:10.11
    container_name: mariadb-{node_name}
    environment:
      MYSQL_ROOT_PASSWORD: ${MYSQL_ROOT_PASSWORD}
      MYSQL_DATABASE: aero_melody
      MYSQL_USER: aero_user
      MYSQL_PASSWORD: ${MYSQL_PASSWORD}
      GALERA_USER: galera_user
      GALERA_PASSWORD: ${GALERA_PASSWORD}
    volumes:
      - ./data/{node_name}:/var/lib/mysql
      - ./config/{node_name}.cnf:/etc/mysql/conf.d/galera.cnf
      - ./scripts:/docker-entrypoint-initdb.d
    networks:
      - galera-network
    ports:
      - "{3306 + i}:3306"
      - "{4567 + i}:4567"
      - "{4444 + i}:4444"
    healthcheck:
      test: ["CMD", "mysqladmin", "ping", "-u", "root", "-p${MYSQL_ROOT_PASSWORD}"]
      interval: 10s
      timeout: 5s
      retries: 5
      start_period: 30s
    command:
      - mysqld
      - --wsrep-cluster-address=gcomm://{','.join([f'{n["address"]}:{n["port"]}' for n in self.cluster_nodes])}
      - --wsrep-node-name={node_name}
      - --wsrep-node-address={node_address}:{node["port"]}
      - --wsrep-sst-method=mariabackup
      - --wsrep-sst-auth=galera_user:${GALERA_PASSWORD}
"""

        compose_config += """

networks:
  galera-network:
    driver: bridge

volumes:
  mariadb-data:
    driver: local
"""

        return compose_config

    def generate_init_script(self) -> str:
        """Generate initialization script for first node"""
        return """
#!/bin/bash
# Galera Cluster Initialization Script

set -e

echo "Initializing Galera cluster..."

# Wait for MariaDB to start
until mysql -u root -p${MYSQL_ROOT_PASSWORD} -e "SELECT 1" &>/dev/null; do
    echo "Waiting for MariaDB..."
    sleep 2
done

echo "MariaDB started successfully"

# Create galera user for SST
mysql -u root -p${MYSQL_ROOT_PASSWORD} -e "
CREATE USER IF NOT EXISTS 'galera_user'@'%' IDENTIFIED BY '${GALERA_PASSWORD}';
GRANT RELOAD, LOCK TABLES, REPLICATION CLIENT ON *.* TO 'galera_user'@'%';
GRANT REPLICATION SLAVE ON *.* TO 'galera_user'@'%';
FLUSH PRIVILEGES;
"

# Create application database and user
mysql -u root -p${MYSQL_ROOT_PASSWORD} -e "
CREATE DATABASE IF NOT EXISTS aero_melody CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
CREATE USER IF NOT EXISTS 'aero_user'@'%' IDENTIFIED BY '${MYSQL_PASSWORD}';
GRANT ALL PRIVILEGES ON aero_melody.* TO 'aero_user'@'%';
FLUSH PRIVILEGES;
"

echo "Galera cluster initialization completed"
"""

    def get_bootstrap_command(self, node_name: str) -> str:
        """Get command to bootstrap a node"""
        return f"docker exec mariadb-{node_name} mysqld --wsrep-cluster-address=gcomm://"

    def get_join_command(self, node_name: str) -> str:
        """Get command to join a node to cluster"""
        cluster_addresses = ",".join([f"{node['address']}:{node['port']}" for node in self.cluster_nodes])
        return f"docker exec mariadb-{node_name} mysqld --wsrep-cluster-address=gcomm://{cluster_addresses}"

    def validate_cluster_config(self) -> List[str]:
        """Validate cluster configuration and return warnings/errors"""
        issues = []

        if len(self.cluster_nodes) < 3:
            issues.append("Warning: Galera cluster should have at least 3 nodes for fault tolerance")

        if len(self.cluster_nodes) % 2 == 0:
            issues.append("Warning: Even number of nodes may cause split-brain scenarios")

        # Check for duplicate addresses
        addresses = [node['address'] for node in self.cluster_nodes]
        if len(addresses) != len(set(addresses)):
            issues.append("Error: Duplicate node addresses found")

        # Check port conflicts
        ports = [node['port'] for node in self.cluster_nodes]
        if len(ports) != len(set(ports)):
            issues.append("Error: Duplicate node ports found")

        return issues

    @property
    def nodes(self) -> List[Dict[str, str]]:
        """Get list of configured cluster nodes"""
        return self.cluster_nodes

    def get_cluster_status(self) -> Optional[Dict]:
        """Get current cluster status (simplified for testing)"""
        if not self.cluster_nodes:
            return None

        # In a real implementation, this would connect to the cluster
        # For testing, return mock status
        return {
            "status": "Primary" if self.cluster_nodes else "Not Configured",
            "cluster_size": len(self.cluster_nodes),
            "nodes": self.cluster_nodes,
            "ready": self.is_cluster_ready
        }


# Global cluster manager instance
cluster_manager = GaleraManager()


def get_galera_manager() -> GaleraManager:
    """Get the global Galera cluster manager"""
    return cluster_manager


def setup_cluster_nodes(nodes: List[Dict[str, str]]):
    """Setup cluster with predefined nodes"""
    manager = get_galera_manager()

    for node in nodes:
        manager.add_node(
            node_name=node['name'],
            node_address=node['address'],
            port=node.get('port', '4567')
        )

    logger.info(f"Configured Galera cluster with {len(nodes)} nodes")
