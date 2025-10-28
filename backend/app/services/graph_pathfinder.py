"""
Graph-based route pathfinding service using NetworkX
Implements Dijkstra's algorithm for optimal flight route discovery
"""

import networkx as nx
from typing import List, Dict, Tuple, Optional
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
import logging
import math

from app.models.models import Airport, Route

logger = logging.getLogger(__name__)


class FlightNetworkGraph:
    """Builds and manages flight network graph for pathfinding"""

    def __init__(self):
        self.graph = nx.DiGraph()
        self.airport_cache = {}
        self.is_built = False

    async def build_graph(self, db: AsyncSession):
        """Build flight network graph from database - OPTIMIZED VERSION"""
        try:
            # Load all airports at once and create lookup dictionary
            airports_result = await db.execute(select(Airport))
            airports = airports_result.scalars().all()
            
            # Create airport lookup by ID for fast access
            airport_lookup = {}
            valid_airports = 0
            
            for airport in airports:
                # Skip airports without valid IATA codes
                if not airport.iata_code or airport.iata_code.strip() == "":
                    continue
                
                # Skip airports without valid coordinates
                if airport.latitude is None or airport.longitude is None:
                    continue
                
                try:
                    self.graph.add_node(
                        airport.iata_code,
                        id=airport.id,
                        name=airport.name,
                        city=airport.city,
                        country=airport.country,
                        latitude=float(airport.latitude),
                        longitude=float(airport.longitude)
                    )
                    self.airport_cache[airport.iata_code] = airport
                    airport_lookup[airport.id] = airport  # Add to lookup
                    valid_airports += 1
                except (ValueError, TypeError) as e:
                    logger.warning(f"Skipping airport {airport.id} due to invalid data: {e}")
                    continue

            logger.info(f"Loaded {valid_airports} valid airports")

            # Load all routes at once - NO MORE INDIVIDUAL QUERIES
            routes_result = await db.execute(select(Route))
            routes = routes_result.scalars().all()
            
            valid_routes = 0
            for route in routes:
                # Use lookup instead of database query
                origin = airport_lookup.get(route.origin_airport_id)
                destination = airport_lookup.get(route.destination_airport_id)

                # Skip routes with invalid airports or IATA codes
                if not origin or not destination:
                    continue
                
                if not origin.iata_code or not destination.iata_code:
                    continue
                
                # Only add edge if both nodes exist in the graph
                if origin.iata_code in self.graph and destination.iata_code in self.graph:
                    try:
                        self.graph.add_edge(
                            origin.iata_code,
                            destination.iata_code,
                            weight=route.distance_km or 0,
                            duration=route.duration_min or 0,
                            route_id=route.id
                        )
                        valid_routes += 1
                    except Exception as e:
                        logger.warning(f"Skipping route {route.id} due to error: {e}")
                        continue

            self.is_built = True
            logger.info(f"Graph built successfully: {self.graph.number_of_nodes()} nodes, {self.graph.number_of_edges()} edges ({valid_routes} valid routes)")

        except Exception as e:
            logger.error(f"Error building graph: {e}")
            raise

    async def _get_airport_by_id(self, db: AsyncSession, airport_id: int) -> Optional[Airport]:
        """Get airport by ID"""
        result = await db.execute(
            select(Airport).where(Airport.id == airport_id)
        )
        return result.scalar_one_or_none()

    def find_shortest_path(
        self,
        origin_code: str,
        destination_code: str,
        weight_metric: str = "weight"
    ) -> Optional[Dict]:
        """
        Find shortest path using Dijkstra's algorithm
        
        Args:
            origin_code: Origin airport IATA code
            destination_code: Destination airport IATA code
            weight_metric: 'weight' for distance, 'duration' for time
            
        Returns:
            Dict with path, total_distance, total_duration, and waypoints
        """
        if not self.is_built:
            logger.error("Graph not built. Call build_graph() first.")
            return None

        try:
            # Find shortest path using Dijkstra
            path = nx.dijkstra_path(
                self.graph,
                origin_code,
                destination_code,
                weight=weight_metric
            )

            # Calculate path metrics
            total_distance = 0
            total_duration = 0
            waypoints = []

            for i in range(len(path) - 1):
                edge_data = self.graph[path[i]][path[i + 1]]
                total_distance += edge_data.get('weight', 0)
                total_duration += edge_data.get('duration', 0)

                waypoints.append({
                    'from': path[i],
                    'to': path[i + 1],
                    'distance_km': edge_data.get('weight', 0),
                    'duration_min': edge_data.get('duration', 0),
                    'route_id': edge_data.get('route_id')
                })

            return {
                'path': path,
                'total_distance_km': total_distance,
                'total_duration_min': total_duration,
                'waypoints': waypoints,
                'num_stops': len(path) - 2  # Excluding origin and destination
            }

        except nx.NetworkXNoPath:
            logger.warning(f"No path found between {origin_code} and {destination_code}")
            return None
        except Exception as e:
            logger.error(f"Error finding path: {e}")
            return None

    def find_alternative_routes(
        self,
        origin_code: str,
        destination_code: str,
        k: int = 3
    ) -> List[Dict]:
        """
        Find k alternative routes between airports
        
        Args:
            origin_code: Origin airport IATA code
            destination_code: Destination airport IATA code
            k: Number of alternative routes to find
            
        Returns:
            List of route dictionaries
        """
        if not self.is_built:
            return []

        try:
            # Find k shortest paths
            paths = list(nx.shortest_simple_paths(
                self.graph,
                origin_code,
                destination_code,
                weight='weight'
            ))

            alternative_routes = []
            for path in paths[:k]:
                total_distance = 0
                total_duration = 0

                for i in range(len(path) - 1):
                    edge_data = self.graph[path[i]][path[i + 1]]
                    total_distance += edge_data.get('weight', 0)
                    total_duration += edge_data.get('duration', 0)

                alternative_routes.append({
                    'path': path,
                    'total_distance_km': total_distance,
                    'total_duration_min': total_duration,
                    'num_stops': len(path) - 2
                })

            return alternative_routes

        except Exception as e:
            logger.error(f"Error finding alternative routes: {e}")
            return []

    def get_hub_airports(self, top_n: int = 10) -> List[Dict]:
        """
        Identify hub airports based on connectivity
        
        Args:
            top_n: Number of top hubs to return
            
        Returns:
            List of hub airport information
        """
        if not self.is_built:
            return []

        # Calculate degree centrality
        centrality = nx.degree_centrality(self.graph)

        # Sort by centrality
        hubs = sorted(centrality.items(), key=lambda x: x[1], reverse=True)[:top_n]

        hub_info = []
        for code, score in hubs:
            node_data = self.graph.nodes[code]
            hub_info.append({
                'iata_code': code,
                'name': node_data.get('name'),
                'city': node_data.get('city'),
                'country': node_data.get('country'),
                'connectivity_score': score,
                'connections': self.graph.degree(code)
            })

        return hub_info

    def find_nearby_airports(
        self,
        iata_code: str,
        max_distance_km: float = 500
    ) -> List[Dict]:
        """
        Find airports within a certain distance
        
        Args:
            iata_code: Airport IATA code
            max_distance_km: Maximum distance in kilometers
            
        Returns:
            List of nearby airports
        """
        if not self.is_built or iata_code not in self.graph:
            return []

        origin_node = self.graph.nodes[iata_code]
        nearby = []

        for node_code, node_data in self.graph.nodes(data=True):
            if node_code == iata_code:
                continue

            # Calculate haversine distance
            distance = self._haversine_distance(
                origin_node['latitude'],
                origin_node['longitude'],
                node_data['latitude'],
                node_data['longitude']
            )

            if distance <= max_distance_km:
                nearby.append({
                    'iata_code': node_code,
                    'name': node_data.get('name'),
                    'city': node_data.get('city'),
                    'distance_km': distance
                })

        return sorted(nearby, key=lambda x: x['distance_km'])

    def _haversine_distance(
        self,
        lat1: float,
        lon1: float,
        lat2: float,
        lon2: float
    ) -> float:
        """Calculate haversine distance between two coordinates"""
        R = 6371  # Earth radius in kilometers

        lat1_rad = math.radians(lat1)
        lat2_rad = math.radians(lat2)
        delta_lat = math.radians(lat2 - lat1)
        delta_lon = math.radians(lon2 - lon1)

        a = (math.sin(delta_lat / 2) ** 2 +
             math.cos(lat1_rad) * math.cos(lat2_rad) *
             math.sin(delta_lon / 2) ** 2)
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

        return R * c

    def get_graph_statistics(self) -> Dict:
        """Get network graph statistics"""
        if not self.is_built:
            return {}

        return {
            'total_airports': self.graph.number_of_nodes(),
            'total_routes': self.graph.number_of_edges(),
            'average_degree': sum(dict(self.graph.degree()).values()) / self.graph.number_of_nodes(),
            'is_connected': nx.is_weakly_connected(self.graph),
            'number_of_components': nx.number_weakly_connected_components(self.graph)
        }


class RoutePathfindingService:
    """Service for route pathfinding and optimization"""

    def __init__(self):
        self.graph = FlightNetworkGraph()

    async def initialize(self, db: AsyncSession):
        """Initialize the service by building the graph"""
        await self.graph.build_graph(db)

    async def find_optimal_route(
        self,
        db: AsyncSession,
        origin_code: str,
        destination_code: str,
        optimize_for: str = "distance"
    ) -> Optional[Dict]:
        """
        Find optimal route between two airports
        
        Args:
            db: Database session
            origin_code: Origin airport IATA code
            destination_code: Destination airport IATA code
            optimize_for: 'distance' or 'duration'
            
        Returns:
            Optimal route information
        """
        if not self.graph.is_built:
            await self.graph.build_graph(db)

        weight_metric = "weight" if optimize_for == "distance" else "duration"
        return self.graph.find_shortest_path(origin_code, destination_code, weight_metric)

    async def find_multi_stop_routes(
        self,
        db: AsyncSession,
        origin_code: str,
        destination_code: str,
        max_stops: int = 3
    ) -> List[Dict]:
        """Find routes with multiple stops"""
        if not self.graph.is_built:
            await self.graph.build_graph(db)

        return self.graph.find_alternative_routes(origin_code, destination_code, max_stops)

    async def discover_hub_airports(
        self,
        db: AsyncSession,
        top_n: int = 10
    ) -> List[Dict]:
        """Discover major hub airports"""
        if not self.graph.is_built:
            await self.graph.build_graph(db)

        return self.graph.get_hub_airports(top_n)

    async def find_nearby_alternatives(
        self,
        db: AsyncSession,
        iata_code: str,
        max_distance_km: float = 500
    ) -> List[Dict]:
        """Find nearby alternative airports"""
        if not self.graph.is_built:
            await self.graph.build_graph(db)

        return self.graph.find_nearby_airports(iata_code, max_distance_km)

    def get_network_stats(self) -> Dict:
        """Get flight network statistics"""
        return self.graph.get_graph_statistics()
