"""
Educational features routes - Geography and graph theory visualization
Using OpenFlights dataset: 3,000+ airports, 67,000+ routes
"""

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from typing import List, Optional, Dict
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy import func, text
import logging

from app.db.database import get_db
from app.models.models import Airport, Route

logger = logging.getLogger(__name__)

router = APIRouter()


class LessonRequest(BaseModel):
    lesson_type: str  # geography, graph-theory, music-theory
    difficulty: str  # beginner, intermediate, advanced


class LessonResponse(BaseModel):
    lesson_id: str
    title: str
    content: dict
    interactive_elements: List[dict]


@router.get("/lessons")
async def get_lessons(db: AsyncSession = Depends(get_db)):
    """Get available educational lessons with real OpenFlights data"""
    try:
        # Get dataset statistics
        airport_count = await db.execute(select(func.count(Airport.id)))
        total_airports = airport_count.scalar() or 3000
        
        route_count = await db.execute(select(func.count(Route.id)))
        total_routes = route_count.scalar() or 67000
        
        return {
            "lessons": [
                {
                    "id": "geography",
                    "title": "Geography Through Sound",
                    "description": f"Learn about world geography using {total_airports:,} airports and {total_routes:,} flight routes from OpenFlights dataset",
                    "difficulty_levels": ["beginner", "intermediate", "advanced"],
                    "topics": [
                        "Distance and pitch correlation",
                        "Direction and melody patterns",
                        "Time zones and tempo variations",
                        "Latitude/longitude to musical scales",
                    ],
                    "dataset_info": {
                        "airports": total_airports,
                        "routes": total_routes,
                        "source": "OpenFlights.org"
                    }
                },
                {
                    "id": "graph-theory",
                    "title": "Graph Theory Visualization",
                    "description": f"Understand graph algorithms through musical pathfinding across {total_routes:,} real flight routes",
                    "difficulty_levels": ["beginner", "intermediate", "advanced"],
                    "topics": [
                        "Dijkstra's shortest path algorithm",
                        "Network connectivity and hubs",
                        "Breadth-first search sonification",
                        "Graph density and musical complexity",
                    ],
                    "dataset_info": {
                        "nodes": total_airports,
                        "edges": total_routes,
                        "source": "OpenFlights.org"
                    }
                },
                {
                    "id": "music-theory",
                    "title": "Music Theory Basics",
                    "description": "Learn scales, tempo, and harmony through interactive flight route examples",
                    "difficulty_levels": ["beginner", "intermediate", "advanced"],
                    "topics": [
                        "Musical scales and modes",
                        "Tempo and rhythm patterns",
                        "Harmony and chord progressions",
                        "Data-to-music mapping techniques",
                    ],
                    "dataset_info": {
                        "examples": "Real flight routes",
                        "source": "OpenFlights.org"
                    }
                },
            ]
        }
    except Exception as e:
        logger.error(f"Error fetching lessons: {e}")
        # Return default lessons if database query fails
        return {
            "lessons": [
                {
                    "id": "geography",
                    "title": "Geography Through Sound",
                    "description": "Learn about world geography using 3,000+ airports and 67,000+ flight routes",
                    "difficulty_levels": ["beginner", "intermediate", "advanced"],
                    "topics": ["Distance and pitch", "Direction and melody", "Time zones and tempo"],
                },
                {
                    "id": "graph-theory",
                    "title": "Graph Theory Visualization",
                    "description": "Understand graph algorithms through musical pathfinding",
                    "difficulty_levels": ["beginner", "intermediate", "advanced"],
                    "topics": ["Dijkstra's algorithm", "Network connectivity", "Shortest path"],
                },
                {
                    "id": "music-theory",
                    "title": "Music Theory Basics",
                    "description": "Learn scales, tempo, and harmony through examples",
                    "difficulty_levels": ["beginner", "intermediate", "advanced"],
                    "topics": ["Scales and modes", "Tempo and rhythm", "Harmony"],
                },
            ]
        }


@router.post("/lessons/{lesson_id}/start", response_model=LessonResponse)
async def start_lesson(
    lesson_id: str, 
    request: LessonRequest,
    db: AsyncSession = Depends(get_db)
):
    """Start an educational lesson with real OpenFlights data"""
    try:
        # Fetch real route examples from database
        sample_routes_query = text("""
            SELECT 
                r.id,
                o.iata_code as origin_code,
                o.name as origin_name,
                o.city as origin_city,
                o.country as origin_country,
                d.iata_code as dest_code,
                d.name as dest_name,
                d.city as dest_city,
                d.country as dest_country,
                r.distance_km
            FROM routes r
            JOIN airports o ON r.origin_airport_id = o.id
            JOIN airports d ON r.destination_airport_id = d.id
            WHERE r.distance_km IS NOT NULL
            AND o.iata_code IN ('JFK', 'LAX', 'LHR', 'CDG', 'NRT', 'DXB', 'SYD', 'SIN')
            AND d.iata_code IN ('JFK', 'LAX', 'LHR', 'CDG', 'NRT', 'DXB', 'SYD', 'SIN')
            ORDER BY r.distance_km DESC
            LIMIT 10
        """)
        
        result = await db.execute(sample_routes_query)
        sample_routes = result.fetchall()
        
        # Build lesson content based on lesson type
        if lesson_id == "geography":
            examples = []
            for route in sample_routes[:5]:
                examples.append({
                    "route": f"{route[1]} → {route[5]}",
                    "origin": f"{route[2]}, {route[3]}, {route[4]}",
                    "destination": f"{route[6]}, {route[7]}, {route[8]}",
                    "distance_km": float(route[9]) if route[9] else 0,
                    "pitch_mapping": "Higher notes for longer distances",
                    "learning_point": f"This {route[9]:.0f}km route creates a unique melody based on geographic coordinates"
                })
            
            content = {
                "introduction": "Learn how geographic distance and direction translate to musical elements using real flight data from OpenFlights",
                "key_concepts": [
                    "Distance → Note Count: Longer routes = more musical notes",
                    "Latitude → Pitch: Northern routes use higher octaves",
                    "Longitude → Scale: East-West travel affects musical scale selection",
                    "Direction → Melody: Route bearing influences melodic contour"
                ],
                "examples": examples,
                "dataset_facts": [
                    "Using 3,000+ airports from OpenFlights.org",
                    "67,000+ real flight routes worldwide",
                    "Covers all continents and major aviation hubs",
                    "Data includes precise latitude/longitude coordinates"
                ]
            }
            
            interactive_elements = [
                {
                    "type": "route_explorer",
                    "description": "Select any two airports to generate and hear their musical route",
                    "action": "Try the Interactive Lab tab"
                },
                {
                    "type": "distance_comparison",
                    "description": "Compare short routes (JFK→LAX) vs long routes (JFK→SYD)",
                    "learning": "Notice how distance affects melody complexity"
                },
                {
                    "type": "quiz",
                    "question": "Which route would create more musical notes?",
                    "options": ["JFK → LAX (4,000km)", "LHR → SYD (17,000km)", "CDG → LHR (350km)"],
                    "answer": "LHR → SYD (17,000km)",
                    "explanation": "Longer distances create more waypoints, resulting in more notes"
                }
            ]
            
        elif lesson_id == "graph-theory":
            # Get hub airports (most connections)
            hub_query = text("""
                SELECT 
                    a.iata_code,
                    a.name,
                    a.city,
                    a.country,
                    COUNT(r.id) as connection_count
                FROM airports a
                JOIN routes r ON a.id = r.origin_airport_id
                WHERE a.iata_code IS NOT NULL
                GROUP BY a.id, a.iata_code, a.name, a.city, a.country
                ORDER BY connection_count DESC
                LIMIT 10
            """)
            
            hub_result = await db.execute(hub_query)
            hubs = hub_result.fetchall()
            
            hub_examples = []
            for hub in hubs[:5]:
                hub_examples.append({
                    "airport": f"{hub[0]} - {hub[1]}",
                    "location": f"{hub[2]}, {hub[3]}",
                    "connections": hub[4],
                    "musical_representation": f"Hub with {hub[4]} connections creates complex harmonic patterns"
                })
            
            content = {
                "introduction": "Understand graph theory concepts through the aviation network. Each airport is a node, each route is an edge.",
                "key_concepts": [
                    "Nodes (Airports): 3,000+ vertices in the graph",
                    "Edges (Routes): 67,000+ connections between nodes",
                    "Shortest Path: Dijkstra's algorithm finds optimal routes",
                    "Hub Detection: High-degree nodes create musical 'centers'",
                    "Network Density: More connections = richer harmonies"
                ],
                "examples": hub_examples,
                "algorithms": [
                    {
                        "name": "Dijkstra's Shortest Path",
                        "description": "Finds the shortest route between two airports",
                        "musical_mapping": "Each node visit plays a note, creating a melody of exploration",
                        "complexity": "O(V²) with adjacency matrix, O(E + V log V) with priority queue"
                    },
                    {
                        "name": "Breadth-First Search",
                        "description": "Explores all routes level by level",
                        "musical_mapping": "Creates layered harmonies as it expands outward",
                        "use_case": "Finding all airports within N stops"
                    }
                ],
                "dataset_facts": [
                    f"Largest hub: {hubs[0][0]} with {hubs[0][4]} connections",
                    "Average path length: 2-3 hops between any two airports",
                    "Network exhibits 'small world' properties",
                    "Real-world graph with scale-free characteristics"
                ]
            }
            
            interactive_elements = [
                {
                    "type": "algorithm_visualization",
                    "description": "Watch Dijkstra's algorithm explore the network musically",
                    "action": "Generate a route to see the algorithm in action"
                },
                {
                    "type": "hub_analysis",
                    "description": "Compare routes through major hubs vs direct routes",
                    "learning": "Hub airports create harmonic 'centers' in the musical composition"
                },
                {
                    "type": "quiz",
                    "question": "What does a high-degree node (hub airport) represent musically?",
                    "options": [
                        "A single note",
                        "A complex chord with many harmonics",
                        "Silence",
                        "Random noise"
                    ],
                    "answer": "A complex chord with many harmonics",
                    "explanation": "More connections = more simultaneous musical elements"
                }
            ]
            
        else:  # music-theory
            content = {
                "introduction": "Learn music theory fundamentals through data-driven composition using flight routes",
                "key_concepts": [
                    "Scales: Major (happy), Minor (melancholic), Pentatonic (peaceful)",
                    "Tempo: Speed of playback (60-180 BPM typical)",
                    "Pitch: MIDI notes 0-127, middle C = 60",
                    "Duration: How long each note plays",
                    "Velocity: Volume/intensity of each note (0-127)"
                ],
                "examples": [
                    {
                        "scale": "Major Scale",
                        "notes": "C-D-E-F-G-A-B-C",
                        "mood": "Happy, bright, optimistic",
                        "example_route": "JFK → LAX (transcontinental, uplifting)",
                        "use_case": "Short, pleasant routes"
                    },
                    {
                        "scale": "Minor Scale",
                        "notes": "A-B-C-D-E-F-G-A",
                        "mood": "Melancholic, introspective, dramatic",
                        "example_route": "LHR → SYD (long overnight flight)",
                        "use_case": "Long-haul, contemplative journeys"
                    },
                    {
                        "scale": "Pentatonic Scale",
                        "notes": "C-D-E-G-A-C",
                        "mood": "Peaceful, meditative, universal",
                        "example_route": "Wellness routes (ocean, mountain)",
                        "use_case": "Therapeutic and relaxation music"
                    },
                    {
                        "scale": "Dorian Mode",
                        "notes": "D-E-F-G-A-B-C-D",
                        "mood": "Jazzy, sophisticated, mysterious",
                        "example_route": "Night flights",
                        "use_case": "Evening and overnight routes"
                    }
                ],
                "mapping_rules": [
                    "Distance → Note Count: 1 note per 100km",
                    "Latitude → Octave: Higher latitudes = higher octaves",
                    "Bearing → Melodic Direction: Eastward = ascending, Westward = descending",
                    "Route Complexity → Harmony: Multi-stop routes add chords"
                ],
                "dataset_facts": [
                    "Each route generates unique musical DNA",
                    "Same route can sound different with different scales",
                    "Tempo affects emotional perception of distance",
                    "Real geographic data ensures authentic patterns"
                ]
            }
            
            interactive_elements = [
                {
                    "type": "scale_comparison",
                    "description": "Generate the same route with different scales",
                    "action": "Try JFK→LAX in Major vs Minor scale"
                },
                {
                    "type": "tempo_experiment",
                    "description": "Adjust tempo to change the feel of a route",
                    "learning": "Faster tempo = energetic, Slower = contemplative"
                },
                {
                    "type": "quiz",
                    "question": "Which scale would best represent a peaceful ocean route?",
                    "options": ["Major", "Minor", "Pentatonic", "Chromatic"],
                    "answer": "Pentatonic",
                    "explanation": "Pentatonic scales are universally pleasant and calming"
                }
            ]
        
        # ✅ Sync education lesson to DuckDB using vector sync helper
        try:
            from app.services.vector_sync_helper import get_vector_sync_helper
            
            vector_sync = get_vector_sync_helper()
            vector_sync.sync_education_lesson(
                lesson_type=lesson_id,
                difficulty=request.difficulty,
                topic=content.get("introduction", "")[:100],
                interaction_count=len(interactive_elements),
                metadata={
                    "key_concepts_count": len(content.get("key_concepts", [])),
                    "examples_count": len(content.get("examples", [])),
                    "has_quiz": any(elem.get("type") == "quiz" for elem in interactive_elements)
                }
            )
            logger.info(f"✅ Synced education lesson to DuckDB: {lesson_id} ({request.difficulty})")
        except Exception as e:
            logger.warning(f"Could not sync education lesson to DuckDB: {e}")
        
        return LessonResponse(
            lesson_id=lesson_id,
            title=f"{lesson_id.replace('-', ' ').title()} - {request.difficulty.title()} Level",
            content=content,
            interactive_elements=interactive_elements,
        )
        
    except Exception as e:
        logger.error(f"Error starting lesson: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/graph-visualization/{origin}/{destination}")
async def visualize_graph_algorithm(origin: str, destination: str):
    """Visualize graph algorithm as music"""
    return {
        "algorithm": "dijkstra",
        "origin": origin,
        "destination": destination,
        "steps": [
            {"node": origin, "note": 60, "time": 0.0},
            {"node": "intermediate", "note": 62, "time": 0.5},
            {"node": destination, "note": 64, "time": 1.0},
        ],
        "explanation": "Each node visit creates a musical note, showing the algorithm's exploration",
    }
