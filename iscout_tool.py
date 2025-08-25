import streamlit as st
import pandas as pd
import requests
import json
import time
import os
import logging
from typing import List, Dict, Optional, Tuple, Union, Any
import re
from dataclasses import dataclass, field
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime, timedelta
import hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed
import sqlite3
from pathlib import Path

# Try to import geopy, install if not available
try:
    from geopy.distance import geodesic
    from geopy.geocoders import Nominatim
    GEOPY_AVAILABLE = True
except ImportError:
    GEOPY_AVAILABLE = False
    st.error("Please install geopy: pip install geopy")

# Enhanced logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('iscout.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# API Configuration with multiple fallback options
GOOGLE_PLACES_API_KEY = (
    st.secrets.get("GOOGLE_PLACES_API_KEY") if hasattr(st, 'secrets') and "GOOGLE_PLACES_API_KEY" in st.secrets else
    os.environ.get("GOOGLE_PLACES_API_KEY", "")
)

# Database setup for caching
DB_PATH = Path("iscout_cache.db")

class DatabaseManager:
    """Enhanced database manager for caching and data persistence"""
    
    def __init__(self):
        self.db_path = DB_PATH
        self.init_database()
    
    def init_database(self):
        """Initialize SQLite database with enhanced schema"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS companies (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        name TEXT NOT NULL,
                        location TEXT,
                        industry TEXT,
                        description TEXT,
                        size TEXT,
                        capabilities TEXT,
                        lat REAL,
                        lon REAL,
                        website TEXT,
                        phone TEXT,
                        rating REAL,
                        user_ratings_total INTEGER,
                        total_score REAL,
                        manufacturing_score REAL,
                        robotics_score REAL,
                        unmanned_score REAL,
                        workforce_score REAL,
                        defense_score REAL,
                        distance_miles REAL,
                        search_location TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS search_history (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        search_location TEXT,
                        radius_miles INTEGER,
                        company_count INTEGER,
                        search_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        search_hash TEXT UNIQUE
                    )
                """)
                
                # Create indexes for better performance
                conn.execute("CREATE INDEX IF NOT EXISTS idx_companies_location ON companies(search_location)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_companies_score ON companies(total_score)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_search_hash ON search_history(search_hash)")
                
        except Exception as e:
            logger.error(f"Database initialization error: {e}")
    
    def save_companies(self, companies: List[Dict], search_location: str) -> bool:
        """Save companies to database with enhanced error handling"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                for company in companies:
                    capabilities_str = json.dumps(company.get('capabilities', []))
                    conn.execute("""
                        INSERT OR REPLACE INTO companies 
                        (name, location, industry, description, size, capabilities, lat, lon, 
                         website, phone, rating, user_ratings_total, total_score, 
                         manufacturing_score, robotics_score, unmanned_score, workforce_score,
                         defense_score, distance_miles, search_location, updated_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                    """, (
                        company['name'], company['location'], company['industry'],
                        company['description'], company['size'], capabilities_str,
                        company['lat'], company['lon'], company['website'], company['phone'],
                        company['rating'], company['user_ratings_total'], company['total_score'],
                        company['manufacturing_score'], company['robotics_score'],
                        company['unmanned_score'], company['workforce_score'],
                        company.get('defense_score', 0), company['distance_miles'], search_location
                    ))
            return True
        except Exception as e:
            logger.error(f"Error saving companies to database: {e}")
            return False
    
    def load_companies(self, search_location: str, max_age_hours: int = 24) -> List[Dict]:
        """Load companies from database with age filtering"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    SELECT * FROM companies 
                    WHERE search_location = ? 
                    AND updated_at > datetime('now', '-{} hours')
                    ORDER BY total_score DESC
                """.format(max_age_hours), (search_location,))
                
                companies = []
                for row in cursor.fetchall():
                    company = {
                        'name': row[1], 'location': row[2], 'industry': row[3],
                        'description': row[4], 'size': row[5], 
                        'capabilities': json.loads(row[6]) if row[6] else [],
                        'lat': row[7], 'lon': row[8], 'website': row[9],
                        'phone': row[10], 'rating': row[11], 'user_ratings_total': row[12],
                        'total_score': row[13], 'manufacturing_score': row[14],
                        'robotics_score': row[15], 'unmanned_score': row[16],
                        'workforce_score': row[17], 'defense_score': row[18] if len(row) > 18 else 0,
                        'distance_miles': row[19] if len(row) > 19 else row[18]
                    }
                    companies.append(company)
                
                return companies
        except Exception as e:
            logger.error(f"Error loading companies from database: {e}")
            return []
    
    def get_search_history(self, limit: int = 10) -> List[Dict]:
        """Get recent search history"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    SELECT * FROM search_history 
                    ORDER BY search_timestamp DESC 
                    LIMIT ?
                """, (limit,))
                
                history = []
                for row in cursor.fetchall():
                    history.append({
                        'location': row[1],
                        'radius': row[2],
                        'count': row[3],
                        'timestamp': row[4]
                    })
                
                return history
        except Exception as e:
            logger.error(f"Error loading search history: {e}")
            return []
        
class AINavalAgent:
    """AI agent for evaluating naval supplier relevance"""
    
    @staticmethod
    def evaluate_company(company_data: Dict) -> Dict:
        """AI evaluation of company naval relevance"""
        
        name = company_data.get('name', '').lower()
        industry = company_data.get('industry', '').lower() 
        description = company_data.get('description', '').lower()
        types = company_data.get('types', [])
        
        combined = f"{name} {industry} {description} {' '.join(types)}"
        
        evaluation = {
            'is_relevant': False,
            'naval_score': 0,
            'rejection_reason': None
        }
        
        # IMMEDIATE DISQUALIFIERS - Government/Military Offices
        government_disqualifiers = [
            'recruiting', 'recruitment', 'marine corps', 'army', 'air force',
            'reserve center', 'military office', 'government office', 
            'administrative office', 'headquarters', 'command center'
        ]
        
        for disqualifier in government_disqualifiers:
            if disqualifier in combined:
                evaluation['rejection_reason'] = f"Government office: {disqualifier}"
                return evaluation
        
        # SERVICE COMPANY DISQUALIFIERS
        service_disqualifiers = [
            'general contractor', 'construction company', 'remodeling',
            'restoration', 'facility services', 'cleaning services',
            'consulting firm', 'real estate', 'insurance'
        ]
        
        for disqualifier in service_disqualifiers:
            if disqualifier in combined:
                evaluation['rejection_reason'] = f"Service company: {disqualifier}"
                return evaluation
        
        # MANUFACTURING INDICATORS (must have at least one)
        manufacturing_indicators = [
            'machining', 'fabrication', 'welding', 'manufacturing',
            'shipyard', 'shipbuilding', 'marine engineering', 'naval systems',
            'aerospace', 'defense manufacturing', 'robotics', 'automation'
        ]
        
        score = 0
        for indicator in manufacturing_indicators:
            if indicator in combined:
                score += 10
        
        if score >= 10:  # At least one manufacturing indicator
            evaluation['is_relevant'] = True
            evaluation['naval_score'] = min(100, score)
        else:
            evaluation['rejection_reason'] = "No manufacturing capability detected"
        
        return evaluation
    
# Enhanced Configuration
@dataclass
class SearchConfig:
    """Enhanced search configuration with validation and defaults"""
    base_location: str = "South Bend, Indiana"
    radius_miles: int = 60
    target_company_count: int = 100
    
    # Advanced search parameters
    enable_async_search: bool = True
    max_concurrent_requests: int = 5
    request_timeout: int = 30
    enable_caching: bool = True
    cache_max_age_hours: int = 24
    
    # Enhanced keyword categories with weighted scoring
    manufacturing_keywords: List[str] = field(default_factory=lambda: [
        "advanced manufacturing", "precision machining", "metal fabrication",
        "additive manufacturing", "3D printing", "CNC machining",
        "welding", "assembly", "fabrication", "machining", "casting",
        "forging", "sheet metal", "tool and die", "injection molding"
    ])
    
    robotics_keywords: List[str] = field(default_factory=lambda: [
        "robotics", "automation", "robotic welding", "industrial automation",
        "automated inspection", "robotics integration", "FANUC", "KUKA",
        "automated systems", "robotic systems", "collaborative robots",
        "AGV", "automated guided vehicles", "vision systems"
    ])
    
    unmanned_keywords: List[str] = field(default_factory=lambda: [
        "unmanned systems", "autonomous", "UAV", "UUV", "USV",
        "drone", "autonomous vehicles", "unmanned vehicles",
        "remote systems", "autonomous systems", "ROV", "AUV",
        "unmanned ground vehicles", "swarm robotics"
    ])
    
    workforce_keywords: List[str] = field(default_factory=lambda: [
        "naval training", "maritime training", "shipyard training", "welding certification",
        "maritime academy", "technical training", "apprenticeship", "workforce development",
        "skills training", "industrial training", "safety training", "crane operator training",
        "maritime safety", "naval education", "defense training", "military training",
        "shipbuilding training", "marine engineering training", "technical certification",
        "OSHA training", "maritime simulation", "deck officer training"
    ])
    
    # Defense contractor keywords
    defense_keywords: List[str] = field(default_factory=lambda: [
        "defense contractor", "military contractor", "naval contractor",
        "aerospace defense", "prime contractor", "subcontractor",
        "DFARS", "ITAR", "security clearance", "classified work",
        "government contracting", "GSA schedule"
    ])
    
    def validate(self) -> bool:
        """Validate configuration parameters"""
        if self.radius_miles <= 0 or self.radius_miles > 200:
            raise ValueError("Radius must be between 1 and 200 miles")
        if self.target_company_count <= 0 or self.target_company_count > 500:
            raise ValueError("Target company count must be between 1 and 500")
        return True

class EnhancedCompanySearcher:
    """Enhanced company searcher with advanced features"""
    
    def search_companies(self) -> List[Dict]:
        """Compatibility wrapper for callers expecting `search_companies`."""
        return self.search_companies_sync()

    def __init__(self, config: SearchConfig):
        self.config = config
        self.config.validate()
        if GEOPY_AVAILABLE:
            self.geolocator = Nominatim(user_agent="wbi_naval_search_enhanced_v2")
        self.base_coords = self._get_coordinates(config.base_location)
        self.db_manager = DatabaseManager()
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'WBI Naval Search Tool v2.0'
        })
        
    def _get_coordinates(self, location: str) -> Tuple[float, float]:
        """Enhanced coordinate lookup with fallback options"""
        if not GEOPY_AVAILABLE:
            return self._get_default_coordinates(location)
            
        try:
            location_data = self.geolocator.geocode(location)
            if location_data:
                lat = getattr(location_data, 'latitude', None)
                lon = getattr(location_data, 'longitude', None)
                if lat is not None and lon is not None:
                    logger.info(f"Coordinates found for {location}: {lat}, {lon}")
                    return (float(lat), float(lon))
            else:
                logger.warning(f"No coordinates found for {location}, using default")
                return self._get_default_coordinates(location)
        except Exception as e:
            logger.error(f"Geocoding error for {location}: {e}")
            return self._get_default_coordinates(location)
        
        return self._get_default_coordinates(location)
    
    def _get_default_coordinates(self, location: str) -> Tuple[float, float]:
        """Fallback coordinates for major naval locations"""
        default_coords = {
            "south bend": (41.6764, -86.2520),
            "norfolk": (36.8508, -76.2859),
            "san diego": (32.7157, -117.1611),
            "pearl harbor": (21.3099, -157.8581),
            "bremerton": (47.5673, -122.6329),
            "portsmouth": (43.0718, -70.7626),
            "newport news": (37.0871, -76.4730),
            "bath": (43.9109, -69.8214),
            "groton": (41.3501, -72.0979),
            "pascagoula": (30.3657, -88.5561)
        }
        
        for key, coords in default_coords.items():
            if key in location.lower():
                return coords
        
        # Ultimate fallback
        return (41.6764, -86.2520)  # South Bend
    
    def _calculate_distance(self, lat: float, lon: float) -> float:
        """Enhanced distance calculation with error handling"""
        if not GEOPY_AVAILABLE:
            # Simple distance approximation when geopy not available
            lat_diff = abs(lat - self.base_coords[0])
            lon_diff = abs(lon - self.base_coords[1])
            return round((lat_diff + lon_diff) * 69, 2)  # Rough miles conversion
            
        try:
            if lat and lon and self.base_coords:
                distance = geodesic(self.base_coords, (lat, lon)).miles
                return round(distance, 2)
            return 999.0
        except Exception as e:
            logger.error(f"Distance calculation error: {e}")
            return 999.0
    
    def _score_company_relevance(self, company_data: Dict) -> Dict:
        """Enhanced relevance scoring with multiple factors"""
        description = company_data.get('description', '').lower()
        name = company_data.get('name', '').lower()
        industry = company_data.get('industry', '').lower()
        website = company_data.get('website', '').lower()
        
        combined_text = f"{description} {name} {industry} {website}"
        
        scores = {
            'manufacturing_score': 0.0,
            'robotics_score': 0.0,
            'unmanned_score': 0.0,
            'workforce_score': 0.0,
            'defense_score': 0.0,
            'size_bonus': 0.0,
            'quality_bonus': 0.0,
            'total_score': 0.0
        }
        
        # Strong naval-specific weights
        naval_weights = {
            "navy": 20, "us navy": 25, "navsea": 30, "navair": 25, "onr": 15,
            "maritime": 15, "marine": 12, "shipbuilding": 25, "shipyard": 25,
            "dry dock": 18, "submarine": 25, "uuv": 25, "rov": 20, "sonar": 22,
            "hull": 10, "propulsion": 12, "oceanographic": 14, "bathymetry": 14,
            "coast guard": 16
        }
        naval_score = 0.0
        for kw, w in naval_weights.items():
            if kw in combined_text:
                naval_score += w
                occ = combined_text.count(kw)
                if occ > 1:
                    naval_score += (occ - 1) * w * 0.3

        # Penalize common false positives
        negatives = [
            "environmental consulting", "remediation", "abatement",
            "general contractor", "residential", "home builder",
        ]
        neg_penalty = sum(10 for n in negatives if n in combined_text)
        naval_score = max(naval_score - neg_penalty, 0)

        # Fold naval score into defense & total
        scores['defense_score'] += naval_score
        scores['total_score'] += naval_score

        keyword_weights = {
            'manufacturing': {
                # Only score if they're actual manufacturers
                'cnc machining': 10, 'precision machining': 8, 'metal fabrication': 8,
                'welding fabrication': 6, 'manufacturing engineering': 6,
                'additive manufacturing': 8, '3d printing manufacturing': 7,
                'aerospace manufacturing': 12, 'defense manufacturing': 10
            },
            'robotics': {
                'robotics integration': 8, 'industrial automation': 6, 'robotic welding': 8,
                'fanuc certified': 6, 'kuka certified': 6, 'automated systems': 5
            },
            'unmanned': {
                'uuv systems': 15, 'rov systems': 12, 'autonomous underwater': 15,
                'unmanned surface': 12, 'drone manufacturing': 8
            },
            'workforce': {
                'maritime training academy': 12, 'naval training center': 15, 
                'shipyard training programs': 12, 'welding certification programs': 8
            },
            'naval_specific': {
                'shipbuilding': 20, 'shipyard': 18, 'naval systems': 15,
                'marine engineering': 12, 'submarine systems': 18, 'sonar systems': 15,
                'maritime electronics': 10, 'naval architecture': 12
            }
        }
        
        # Score based on weighted keywords
        for category, keywords in keyword_weights.items():
            category_score = 0.0
            for keyword, weight in keywords.items():
                if keyword in combined_text:
                    category_score += weight
                    # Bonus for multiple mentions
                    occurrences = combined_text.count(keyword)
                    if occurrences > 1:
                        category_score += (occurrences - 1) * weight * 0.3
            
            if category == 'manufacturing':
                scores['manufacturing_score'] = category_score
            elif category == 'robotics':
                scores['robotics_score'] = category_score
            elif category == 'unmanned':
                scores['unmanned_score'] = category_score
            elif category == 'workforce':
                scores['workforce_score'] = category_score
            elif category == 'defense':
                scores['defense_score'] = category_score
        
        # Major defense contractors bonus
        major_contractors = [
            'honeywell', 'boeing', 'lockheed martin', 'raytheon', 'northrop grumman',
            'general dynamics', 'bae systems', 'huntington ingalls', 'newport news',
            'bath iron works', 'electric boat', 'textron', 'l3harris',
            'collins aerospace', 'pratt whitney', 'rolls royce'
        ]
        
        for contractor in major_contractors:
            if contractor in name:
                scores['manufacturing_score'] += 25
                scores['defense_score'] += 20
                break
        
        # Size and quality bonuses
        rating = company_data.get('rating', 0)
        review_count = company_data.get('user_ratings_total', 0)
        
        if rating >= 4.5 and review_count >= 10:
            scores['quality_bonus'] = 5.0
        elif rating >= 4.0 and review_count >= 5:
            scores['quality_bonus'] = 3.0
        
        # Business size scoring
        if review_count > 100:
            scores['size_bonus'] = 8.0
        elif review_count > 50:
            scores['size_bonus'] = 5.0
        elif review_count > 20:
            scores['size_bonus'] = 3.0
        
        # Much stricter penalties for non-manufacturing
        exclude_keywords_in_name = [
            'restoration', 'remodeling', 'construction', 'facility services',
            'general contractor', 'home', 'residential', 'commercial cleaning',
            'environmental services', 'consulting', 'architecture', 'engineering services'
        ]

        for exclude in exclude_keywords_in_name:
            if exclude in combined_text:
                scores['total_score'] = max(0, scores['total_score'] - 20)  # Heavy penalty
        
        
        # Calculate total score with weights
        scores['total_score'] = (
        scores['manufacturing_score'] * 1.0 +
        scores['robotics_score'] * 0.8 +
        scores['unmanned_score'] * 0.9 +
        scores['workforce_score'] * 0.7 +
        scores['defense_score'] * 1.2 +
        scores['size_bonus'] +
        scores['quality_bonus']
        )

        # HEAVY PENALTIES for service companies that sneak through
        service_penalties = [
            ('restoration', -50), ('remodeling', -50), ('construction', -30),
            ('facility services', -40), ('general contractor', -30),
            ('environmental', -25), ('consulting', -20), ('architecture', -25)
        ]

        for penalty_word, penalty_points in service_penalties:
            if penalty_word in combined_text:
                scores['total_score'] += penalty_points  # Negative points

        # Ensure no service company gets above 5 points
        service_company_indicators = ['restoration', 'remodeling', 'construction', 'facility services']
        if any(indicator in combined_text for indicator in service_company_indicators):
            scores['total_score'] = min(scores['total_score'], 2.0)

        return scores
    
    def _ai_evaluate_naval_relevance(self, company_data: Dict) -> Dict:
        """AI-powered evaluation of naval supplier relevance"""
    
        # Create company profile for AI evaluation
        company_profile = f"""
        Company: {company_data.get('name', 'Unknown')}
        Industry: {company_data.get('industry', 'Unknown')}
        Description: {company_data.get('description', 'Unknown')}
        Website: {company_data.get('website', 'Not available')}
        """
    
        # Simple rule-based AI evaluation (can be upgraded to LLM later)
        name = company_data.get('name', '').lower()
        industry = company_data.get('industry', '').lower()
        description = company_data.get('description', '').lower()
    
        combined = f"{name} {industry} {description}"
    
        # AI Decision Logic
        evaluation = {
            'is_manufacturer': False,
            'naval_relevance': 0,
            'confidence': 0,
            'reasoning': []
        }
    
        # Manufacturing indicators (high confidence)
        manufacturing_signals = [
            'machining', 'fabrication', 'manufacturing', 'foundry', 'mill',
            'cnc', 'precision', 'welding', 'casting', 'forging'
        ]
    
        # Naval/maritime indicators  
        naval_signals = [
            'shipyard', 'shipbuilding', 'marine engineering', 'naval',
            'maritime', 'submarine', 'sonar', 'uuv', 'rov', 'vessel'
        ]
    
        # Service company red flags (immediate disqualification)
        service_red_flags = [
            'recruiting', 'recruitment', 'office', 'headquarters', 'administration',
            'consulting only', 'real estate', 'insurance', 'restaurant',
            'general contractor', 'construction company', 'facility services'
        ]
    
        # Check for disqualifiers first
        for flag in service_red_flags:
            if flag in combined:
                evaluation['reasoning'].append(f"Disqualified: {flag}")
                return evaluation
    
        # Check manufacturing capability
        mfg_score = sum(1 for signal in manufacturing_signals if signal in combined)
        if mfg_score > 0:
            evaluation['is_manufacturer'] = True
            evaluation['reasoning'].append(f"Manufacturing signals: {mfg_score}")
    
        # Check naval relevance
        naval_score = sum(2 for signal in naval_signals if signal in combined)  # Higher weight
        evaluation['naval_relevance'] = naval_score
    
        if naval_score > 0:
            evaluation['reasoning'].append(f"Naval signals: {naval_score}")
    
        # Calculate confidence
        if evaluation['is_manufacturer'] and naval_score > 0:
            evaluation['confidence'] = min(90, mfg_score * 20 + naval_score * 10)
        elif evaluation['is_manufacturer']:
            evaluation['confidence'] = min(70, mfg_score * 15)
        elif naval_score > 0:
            evaluation['confidence'] = min(60, naval_score * 10)
    
        return evaluation

    def search_companies_sync(self) -> List[Dict]:
        """Synchronous company search"""
        api_key = st.session_state.get('api_key', GOOGLE_PLACES_API_KEY)
        
        if not api_key:
            st.info("🎭 Using enhanced demo data. Add API key for real company search.")
            return self.generate_enhanced_sample_companies()
        
        # Check cache first
        if self.config.enable_caching:
            cached_companies = self.db_manager.load_companies(
                self.config.base_location, 
                self.config.cache_max_age_hours
            )
            if cached_companies:
                st.info(f"📋 Loaded {len(cached_companies)} companies from cache")
                return cached_companies
        
        st.info("🔍 Searching real companies using enhanced Google Places API...")
        
        search_queries = self._get_enhanced_search_queries()
        all_companies = []
        progress_bar = st.progress(0)
        
        for i, query in enumerate(search_queries):
            st.write(f"Searching: {query}")
            try:
                companies = self.search_google_places_text(query)
                all_companies.extend(companies)
                progress_bar.progress((i + 1) / len(search_queries))
                time.sleep(1)  # Rate limiting
            except Exception as e:
                logger.error(f"Search error for query '{query}': {e}")
                continue
        
        progress_bar.empty()
        
        # Process and deduplicate
        unique_companies = self._process_and_deduplicate(all_companies)
        
        # Save to cache
        if self.config.enable_caching and unique_companies:
            self.db_manager.save_companies(unique_companies, self.config.base_location)
        
        return unique_companies
    
    def _get_enhanced_search_queries(self) -> List[str]:
        """Generate manufacturing-focused search queries"""
        return [
            # Specific manufacturing services
            "CNC machining services", "precision machining", "metal fabrication shop",
            "custom manufacturing", "contract manufacturing", "machine shop",
            "welding fabrication", "sheet metal fabrication", 
        
            # Naval/maritime manufacturing
            "shipbuilding company", "marine engineering", "naval architecture",
            "maritime electronics", "sonar equipment", "submarine systems",
            "ROV manufacturing", "UUV systems", "marine propulsion",
        
            # Defense manufacturing (not offices)
            "defense contractor manufacturing", "aerospace manufacturing",
            "military equipment manufacturer",
        
            # Technology manufacturing
            "robotics manufacturer", "automation systems", "control systems manufacturer",
            "electronics manufacturing", "sensor manufacturer",
        
            # Training facilities (not recruiting)
            "maritime training facility", "welding school", "technical training institute"
        ]
    
    def search_google_places_text(self, query: str) -> List[Dict]:
        """Enhanced Google Places text search"""
        api_key = st.session_state.get('api_key', GOOGLE_PLACES_API_KEY)
        
        if not api_key:
            return []
        
        url = "https://places.googleapis.com/v1/places:searchText"
        
        headers = {
            'Content-Type': 'application/json',
            'X-Goog-Api-Key': api_key,
            'X-Goog-FieldMask': 'places.displayName,places.formattedAddress,places.location,places.types,places.websiteUri,places.nationalPhoneNumber,places.rating,places.userRatingCount,places.businessStatus'
        }
        
        request_data = {
            "textQuery": f"{query} near {self.config.base_location}",
            "maxResultCount": 20
        }
        
        try:
            response = self.session.post(
                url, 
                headers=headers, 
                json=request_data, 
                timeout=self.config.request_timeout
            )
            
            if response.status_code == 200:
                data = response.json()
                return self._process_places_response(data, query)
            elif response.status_code == 403:
                st.error("⚠️ API key doesn't have permission. Enable 'Places API (New)' and billing.")
                return []
            else:
                logger.warning(f"Search for '{query}' returned status {response.status_code}")
                return []
                
        except Exception as e:
            logger.error(f"Error searching for '{query}': {e}")
            return []
    
    def _process_places_response(self, data: Dict, query: str) -> List[Dict]:
        """Process Google Places API response"""
        companies = []
        places = data.get('places', [])
        
        for place in places:
            try:
                place_lat = place.get('location', {}).get('latitude', 0)
                place_lon = place.get('location', {}).get('longitude', 0)
                if not place_lat or not place_lon:
                    continue

                distance = self._calculate_distance(place_lat, place_lon)
                if distance > self.config.radius_miles:
                    continue

                # ✅ define these FIRST
                name = place.get('displayName', {}).get('text', 'Unknown')
                types = place.get('types', []) or []

                # Build a combined string to filter against
                combined_for_filter = " ".join([
                    name or "",
                    ", ".join(types),
                    place.get('formattedAddress', '') or '',
                    place.get('websiteUri', '') or ''
                ])

                # Require naval AND manufacturing relevance (stricter gate)
                if not (self._is_naval_related(combined_for_filter) and self._is_manufacturing_related(name, types)):
                    continue

                business_size = self._determine_business_size(name, types, place)

                company = {
                    'name': name,
                    'location': place.get('formattedAddress', 'Unknown'),
                    'industry': ', '.join(types[:3]) if types else 'Unknown',
                    'description': self._generate_description(name, types, query),
                    'size': business_size,
                    'capabilities': self._extract_capabilities_from_name_and_types(name, types),
                    'lat': place_lat,
                    'lon': place_lon,
                    'website': place.get('websiteUri', 'Not available'),
                    'phone': place.get('nationalPhoneNumber', 'Not available'),
                    'rating': place.get('rating', 0),
                    'user_ratings_total': place.get('userRatingCount', 0)
                }
                companies.append(company)

            except Exception as e:
                logger.error(f"Error processing place: {e}")
                continue
        
        return companies
    
    def _generate_description(self, name: str, types: List[str], query: str) -> str:
        """Generate enhanced company description"""
        base_desc = f"Business type: {', '.join(types[:2]) if types else 'Manufacturing'}"
        
        # Add context based on query
        if 'aerospace' in query.lower():
            base_desc += " - Aerospace industry focus"
        elif 'defense' in query.lower():
            base_desc += " - Defense contractor"
        elif 'training' in query.lower():
            base_desc += " - Training and education services"
        elif 'unmanned' in query.lower() or 'autonomous' in query.lower():
            base_desc += " - Unmanned systems specialist"
        
        return base_desc
    
    def _process_and_deduplicate(self, all_companies: List[Dict]) -> List[Dict]:
        """Enhanced processing and deduplication"""
        unique_companies = []
        seen = set()
        
        for company in all_companies:
            # Create unique key based on name and location
            key = (
                company['name'].lower().strip(),
                company['location'].lower().strip()[:50]  # First 50 chars of location
            )
            
            if key not in seen:
                seen.add(key)
                
                # Add distance calculation
                distance = self._calculate_distance(company['lat'], company['lon'])
                company['distance_miles'] = distance

                # AI VALIDATION - this is the key change
                ai_agent = AINavalAgent()
                ai_eval = ai_agent.evaluate_company(company)

                # Only proceed if AI approves the company
                if not ai_eval['is_relevant']:
                    continue  # Skip this company entirely

                # Add AI evaluation results
                company['naval_score'] = ai_eval['naval_score']
                company['ai_approved'] = True

                # Traditional scoring (but now only for AI-approved companies)
                scores = self._score_company_relevance(company)
                company.update(scores)

                # Filter by distance and minimum relevance
                if distance <= self.config.radius_miles and company['naval_score'] >= 10:
                    unique_companies.append(company)

        # Sort by relevance score and limit results
        unique_companies.sort(key=lambda x: x['total_score'], reverse=True)
        return unique_companies[:self.config.target_company_count]
    
    def _is_manufacturing_related(self, name: str, types: List[str]) -> bool:
        """STRICT manufacturing/naval filter - exclude service companies"""
        name_lower = name.lower()
        types_str = ' '.join(types).lower()
        combined = f"{name_lower} {types_str}"
    
        # IMMEDIATE REJECT - these should NEVER appear
        hard_exclude = [
            'restoration', 'remodeling', 'construction', 'home', 'residential',
            'facility services', 'cleaning', 'maintenance', 'janitorial',
            'environmental services', 'consulting', 'architecture', 'design',
            'real estate', 'property', 'landscaping', 'roofing', 'flooring',
            'restaurant', 'food', 'retail', 'bank', 'insurance', 'legal', 
            'marine corps', 'recruiting', 'recruitment', 'reserve center',
            'military office', 'government office', 'administrative office',
            'headquarters', 'command center', 'base services'
        ]
    
        for exclude in hard_exclude:
            if exclude in combined:
                return False
    
        # MUST have manufacturing/naval indicators
        required_indicators = [
            # Actual manufacturing
            'manufacturing', 'machining', 'fabrication', 'welding', 'casting',
            'forging', 'cnc', 'precision', 'metal', 'steel', 'aluminum',
        
            # Naval/maritime manufacturing
            'shipyard', 'shipbuilding', 'marine', 'naval', 'maritime',
            'submarine', 'vessel', 'boat', 'ship',
        
            # Defense manufacturing
            'aerospace', 'defense contractor', 'military contractor',
        
            # Technology manufacturing
            'robotics', 'automation', 'electronics', 'systems integration',
        
            # Training (naval specific)
            'maritime academy', 'naval training', 'shipyard training'
        ]
    
        # Must have at least one manufacturing indicator
        has_manufacturing = any(indicator in combined for indicator in required_indicators)
    
        # Business types that indicate manufacturing
        manufacturing_types = [
            'manufacturer', 'machine_shop', 'factory', 'foundry', 'mill'
        ]
        has_mfg_type = any(mtype in types_str for mtype in manufacturing_types)
    
        return has_manufacturing or has_mfg_type
    
    def _is_naval_related(self, text: str) -> bool:
        """Lenient filter - mainly exclude obvious non-manufacturing businesses."""
        t = (text or "").lower()

        # Hard negatives – immediate reject
        hard_negatives = [
            "restaurant", "food", "catering", "salon", "boutique",
            "insurance", "bank", "law firm", "chiropractor",
            "real estate", "property management", "wedding",
            "home depot", "lowes", "menards", "outlet store",
            "kitchen", "bathroom", "flooring", "furniture"
        ]
        if any(n in t for n in hard_negatives):
            return False

        # Much broader acceptance - manufacturing, engineering, defense, or training
        accept_signals = [
            # Naval/maritime (strongest)
            "naval", "navy", "maritime", "marine", "ship", "shipyard", 
            "submarine", "sonar", "oceanographic", "coast guard",
        
            # Manufacturing & engineering (broad)
            "manufacturing", "fabrication", "machining", "engineering", 
            "precision", "cnc", "welding", "assembly", "industrial",
        
            # Defense & aerospace
            "defense", "aerospace", "military", "contractor",
        
            # Technology & automation  
            "automation", "robotics", "systems", "technologies",
        
            # Training & education
            "training", "academy", "institute", "certification"
        ]
    
        return any(signal in t for signal in accept_signals)

    def _determine_business_size(self, name: str, types: List[str], place_data: Dict) -> str:
        """Enhanced business size determination"""
        name_lower = name.lower()
        
        # Fortune 500 / Major corporations
        major_corps = [
            'honeywell', 'boeing', 'lockheed', 'raytheon', 'northrop', 'general dynamics',
            'bae systems', 'textron', 'collins aerospace', 'pratt whitney', 'rolls royce',
            'general electric', 'caterpillar', 'john deere', 'cummins', 'ford', 'gm',
            'huntington ingalls', 'newport news shipbuilding', 'bath iron works'
        ]
        
        # Regional/medium indicators
        medium_indicators = [
            'corporation', 'corp', 'industries', 'international', 'group',
            'systems', 'technologies', 'holdings', 'enterprises', 'associates'
        ]
        
        # Small business indicators
        small_indicators = [
            'llc', 'inc', 'ltd', 'company', 'co', 'shop', 'works', 'services',
            'solutions', 'custom', 'specialty', 'precision', 'family', 'brothers',
            'machine shop', 'welding shop'
        ]
        
        # Check for major corporations
        for corp in major_corps:
            if corp in name_lower:
                return 'Fortune 500 / Major Corporation'
        
        # Use review count as size indicator
        review_count = place_data.get('userRatingCount', 0)
        rating = place_data.get('rating', 0)
        
        if review_count > 200 or (review_count > 100 and rating > 4.0):
            return 'Large Corporation'
        elif review_count > 50 or (review_count > 25 and rating > 4.2):
            return 'Medium Business'
        
        # Check name patterns
        for indicator in medium_indicators:
            if indicator in name_lower:
                if review_count > 20:
                    return 'Medium Business'
                else:
                    return 'Small-Medium Business'
        
        # Default categorization
        for indicator in small_indicators:
            if indicator in name_lower:
                return 'Small Business'
        
        # Final decision based on review data
        if review_count >= 15:
            return 'Small-Medium Business'
        else:
            return 'Small Business'
    
    def _extract_capabilities_from_name_and_types(self, name: str, types: List[str]) -> List[str]:
        """Enhanced capability extraction"""
        name_lower = name.lower()
        types_str = ' '.join(types).lower()
        capabilities = set()
        
        capability_mapping = {
            # Manufacturing capabilities
            'cnc': 'CNC Machining',
            'machining': 'Precision Machining',
            'welding': 'Welding Services',
            'fabrication': 'Metal Fabrication',
            'manufacturing': 'Manufacturing',
            'casting': 'Metal Casting',
            'forging': 'Metal Forging',
            'sheet metal': 'Sheet Metal Work',
            'additive': '3D Printing/Additive Manufacturing',
            '3d printing': '3D Printing/Additive Manufacturing',
            
            # Automation and robotics
            'automation': 'Industrial Automation',
            'robotics': 'Robotics Integration',
            'robotic': 'Robotic Systems',
            'vision': 'Machine Vision Systems',
            'control': 'Process Control Systems',
            
            # Aerospace and defense
            'aerospace': 'Aerospace Manufacturing',
            'defense': 'Defense Systems',
            'naval': 'Naval Systems',
            'maritime': 'Maritime Systems',
            'shipyard': 'Shipbuilding',
            'marine': 'Marine Engineering',
            
            # Training and education
            'training': 'Training Services',
            'academy': 'Educational Academy',
            'certification': 'Certification Programs',
            'apprenticeship': 'Apprenticeship Programs',
            'simulation': 'Simulation Training',
            'safety': 'Safety Training',
            
            # Specialized systems
            'unmanned': 'Unmanned Systems',
            'autonomous': 'Autonomous Systems',
            'uav': 'UAV Systems',
            'uuv': 'UUV Systems',
            'rov': 'ROV Systems',
            'radar': 'Radar Systems',
            'sonar': 'Sonar Systems',
            'navigation': 'Navigation Systems'
        }
        
        # Extract capabilities from name and types
        combined_text = f"{name_lower} {types_str}"
        for keyword, capability in capability_mapping.items():
            if keyword in combined_text:
                capabilities.add(capability)
        
        # Add industry-specific capabilities
        if 'engineer' in combined_text:
            capabilities.add('Engineering Services')
        if 'design' in combined_text:
            capabilities.add('Design Services')
        if 'maintenance' in combined_text:
            capabilities.add('Maintenance Services')
        if 'repair' in combined_text:
            capabilities.add('Repair Services')
        
        return list(capabilities) if capabilities else ['General Manufacturing Services']
    
    def generate_enhanced_sample_companies(self) -> List[Dict]:
        """Generate enhanced sample companies with realistic data"""
        base_lat, base_lon = self.base_coords
        location_name = self.config.base_location.split(',')[0].strip()
        state = self.config.base_location.split(',')[-1].strip() if ',' in self.config.base_location else 'IN'
        
        sample_companies = [
            {
                'name': f'{location_name} Advanced Manufacturing Solutions',
                'location': f'{location_name}, {state}',
                'industry': 'Aerospace Manufacturing, Metal Fabrication',
                'description': 'ISO 9001:2015 certified precision manufacturer specializing in aerospace components and defense applications. Advanced CNC machining and quality control systems.',
                'size': 'Medium Business',
                'capabilities': ['CNC Machining', 'Aerospace Manufacturing', 'Quality Control', 'Metal Fabrication'],
                'lat': base_lat + 0.05,
                'lon': base_lon + 0.03,
                'website': f'www.{location_name.lower()}advanced.com',
                'phone': '(555) 555-0101',
                'rating': 4.7,
                'user_ratings_total': 45
            },
            {
                'name': f'Great Lakes Robotics Integration Corp',
                'location': f'{location_name} Metro Area, {state}',
                'industry': 'Industrial Automation, Engineering',
                'description': 'FANUC and KUKA certified robotics integrator. Specializes in automated welding, assembly, and inspection systems for manufacturing.',
                'size': 'Small-Medium Business',
                'capabilities': ['Robotics Integration', 'FANUC Systems', 'KUKA Systems', 'Automated Inspection', 'Industrial Automation'],
                'lat': base_lat - 0.08,
                'lon': base_lon + 0.12,
                'website': 'www.greatlakesrobotics.com',
                'phone': '(555) 555-0102',
                'rating': 4.9,
                'user_ratings_total': 28
            },
            {
                'name': 'Midwest Naval Systems LLC',
                'location': f'{location_name} Industrial District, {state}',
                'industry': 'Defense Contractor, Naval Systems',
                'description': 'ITAR registered defense contractor specializing in naval combat systems, sonar equipment, and maritime electronics.',
                'size': 'Medium Business',
                'capabilities': ['Naval Systems', 'Sonar Systems', 'Defense Systems', 'Maritime Electronics', 'ITAR Compliance'],
                'lat': base_lat + 0.12,
                'lon': base_lon - 0.05,
                'website': 'www.midwestnaval.com',
                'phone': '(555) 555-0103',
                'rating': 4.8,
                'user_ratings_total': 52
            },
            {
                'name': f'{location_name} Maritime Training Institute',
                'location': f'{location_name} Education District, {state}',
                'industry': 'Educational Services, Maritime Training',
                'description': 'USCG approved maritime training facility offering welding certification, shipyard training, and maritime safety programs.',
                'size': 'Medium Business',
                'capabilities': ['Maritime Training', 'Welding Certification', 'Safety Training', 'Shipyard Training', 'USCG Approved Programs'],
                'lat': base_lat - 0.15,
                'lon': base_lon - 0.08,
                'website': f'www.{location_name.lower()}maritime.edu',
                'phone': '(555) 555-0104',
                'rating': 4.6,
                'user_ratings_total': 73
            },
            {
                'name': 'Regional Autonomous Systems Corporation',
                'location': f'{location_name} Tech Park, {state}',
                'industry': 'Unmanned Systems, Technology',
                'description': 'R&D focused company developing UUV, ROV, and autonomous navigation systems for defense and commercial maritime applications.',
                'size': 'Small Business',
                'capabilities': ['UUV Systems', 'ROV Systems', 'Autonomous Navigation', 'Defense Systems', 'R&D Services'],
                'lat': base_lat + 0.18,
                'lon': base_lon + 0.15,
                'website': 'www.regionalautonomous.com',
                'phone': '(555) 555-0105',
                'rating': 4.4,
                'user_ratings_total': 19
            },
            {
                'name': 'Precision Marine Engineering Inc',
                'location': f'Near {location_name}, {state}',
                'industry': 'Marine Engineering, Manufacturing',
                'description': 'Custom marine engineering solutions including propulsion systems, hull modifications, and specialized maritime equipment.',
                'size': 'Small Business',
                'capabilities': ['Marine Engineering', 'Propulsion Systems', 'Custom Manufacturing', 'Naval Architecture'],
                'lat': base_lat - 0.22,
                'lon': base_lon + 0.18,
                'website': 'www.precisionmarine.com',
                'phone': '(555) 555-0106',
                'rating': 4.5,
                'user_ratings_total': 31
            },
            {
                'name': f'{location_name} Additive Manufacturing Hub',
                'location': f'{location_name} Innovation Center, {state}',
                'industry': '3D Printing, Advanced Manufacturing',
                'description': 'Metal 3D printing and additive manufacturing for rapid prototyping and production parts. Titanium, aluminum, and steel capabilities.',
                'size': 'Small Business',
                'capabilities': ['3D Printing/Additive Manufacturing', 'Metal Printing', 'Rapid Prototyping', 'Titanium Processing'],
                'lat': base_lat + 0.09,
                'lon': base_lon - 0.11,
                'website': f'www.{location_name.lower()}additive.com',
                'phone': '(555) 555-0107',
                'rating': 4.3,
                'user_ratings_total': 22
            },
            {
                'name': 'Midwest Defense Contractors Alliance',
                'location': f'{location_name} Metro, {state}',
                'industry': 'Defense Contracting, Engineering',
                'description': 'Prime contractor for naval systems integration. DFARS compliant with active security clearances for classified projects.',
                'size': 'Large Corporation',
                'capabilities': ['Defense Systems', 'Prime Contracting', 'Security Clearance', 'DFARS Compliance', 'Systems Integration'],
                'lat': base_lat - 0.05,
                'lon': base_lon - 0.15,
                'website': 'www.midwestdefense.com',
                'phone': '(555) 555-0108',
                'rating': 4.8,
                'user_ratings_total': 94
            }
        ]
        
        # Add distance and relevance scoring
        for company in sample_companies:
            distance = self._calculate_distance(company['lat'], company['lon'])
            company['distance_miles'] = distance
            
            scores = self._score_company_relevance(company)
            company.update(scores)
        
        # Sort by relevance score
        sample_companies.sort(key=lambda x: x['total_score'], reverse=True)
        
        return sample_companies

class EnhancedVisualization:
    """Enhanced visualization and analytics class"""
    
    @staticmethod
    def create_advanced_company_map(companies: List[Dict], base_coords: Tuple[float, float]) -> Optional[go.Figure]:
        """Create map with proper coordinate validation"""
        if not companies:
            return None
    
        # Create DataFrame and validate coordinates
        df = pd.DataFrame(companies)
    
        # Remove rows with invalid coordinates
        df = df.dropna(subset=['lat', 'lon'])
        df = df[(df['lat'] != 0) & (df['lon'] != 0)]
        df = df[(df['lat'].between(-90, 90)) & (df['lon'].between(-180, 180))]
    
        if df.empty:
            st.warning("No valid coordinates found for mapping")
            return None
    
        fig = go.Figure()
    
        # Add base location
        fig.add_trace(go.Scattermapbox(
            lat=[base_coords[0]],
            lon=[base_coords[1]],
            mode='markers',
            marker=dict(size=20, color='red', symbol='circle'),
            text=['Search Center'],
            name='Base Location'
        ))
    
        # Add companies with validated coordinates
        fig.add_trace(go.Scattermapbox(
            lat=df['lat'].tolist(),
            lon=df['lon'].tolist(),
            mode='markers',
            marker=dict(
                size=10,
                color=df['naval_score'].tolist(),
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Naval Score")
            ),
            text=[f"{row['name']}<br>Score: {row['naval_score']}<br>Distance: {row['distance_miles']:.1f} mi" 
                for _, row in df.iterrows()],
            name='Companies'
        ))
    
        fig.update_layout(
            mapbox=dict(
                style='open-street-map',
                center=dict(lat=base_coords[0], lon=base_coords[1]),
                zoom=8
            ),
            height=600,
            title="Naval Supplier Geographic Distribution"
        )
    
        return fig
    
    @staticmethod
    def create_comprehensive_analytics_dashboard(companies: List[Dict]) -> List[go.Figure]:
        """Create comprehensive analytics dashboard"""
        if not companies:
            return []
        
        df = pd.DataFrame(companies)
        figures = []
        
        # 1. Score Distribution with Statistical Analysis
        fig1 = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Score Distribution', 'Score vs Distance', 'Industry Distribution', 'Size Distribution'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"type": "pie"}, {"type": "bar"}]]
        )
        
        # Histogram of scores
        fig1.add_trace(
            go.Histogram(x=df['total_score'], nbinsx=20, name='Score Distribution'),
            row=1, col=1
        )
        
        # Scatter plot: Score vs Distance
        fig1.add_trace(
            go.Scatter(
                x=df['distance_miles'], 
                y=df['total_score'],
                mode='markers',
                marker=dict(
                    size=8,
                    color=df['total_score'],
                    colorscale='Viridis',
                    showscale=False
                ),
                text=df['name'],
                name='Companies',
                hovertemplate='<b>%{text}</b><br>Distance: %{x:.1f} mi<br>Score: %{y:.1f}<extra></extra>'
            ),
            row=1, col=2
        )
        
        # Industry pie chart
        industry_counts = df['industry'].value_counts().head(8)
        fig1.add_trace(
            go.Pie(
                labels=industry_counts.index,
                values=industry_counts.values,
                name="Industries"
            ),
            row=2, col=1
        )
        
        # Size distribution
        size_counts = df['size'].value_counts()
        fig1.add_trace(
            go.Bar(
                x=size_counts.index,
                y=size_counts.values,
                name="Company Sizes",
                marker_color='lightblue'
            ),
            row=2, col=2
        )
        
        fig1.update_layout(
            height=800,
            title_text="📊 Comprehensive Naval Supplier Analytics Dashboard",
            showlegend=False
        )
        
        figures.append(fig1)
        
        # 2. Capability Analysis
        all_capabilities = []
        for caps in df['capabilities']:
            all_capabilities.extend(caps)
        
        capability_counts = pd.Series(all_capabilities).value_counts().head(15)
        
        fig2 = go.Figure(data=[
            go.Bar(
                x=capability_counts.values,
                y=capability_counts.index,
                orientation='h',
                marker_color='steelblue'
            )
        ])
        
        fig2.update_layout(
            title="🔧 Naval Supplier Capabilities Analysis",
            xaxis_title="Number of Companies",
            yaxis_title="Capabilities",
            height=600
        )
        
        figures.append(fig2)
        
        # 3. Quality and Rating Analysis
        fig3 = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Rating Distribution', 'Rating vs Review Count')
        )
        
        fig3.add_trace(
            go.Histogram(x=df['rating'], nbinsx=10, name='Rating Distribution'),
            row=1, col=1
        )
        
        fig3.add_trace(
            go.Scatter(
                x=df['user_ratings_total'],
                y=df['rating'],
                mode='markers',
                marker=dict(
                    size=df['total_score'],
                    sizemode='diameter',
                    sizeref=2.*max(df['total_score'])/(40.**2),
                    sizemin=4,
                    color=df['total_score'],
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title="Naval Score")
                ),
                text=df['name'],
                name='Quality Analysis',
                hovertemplate='<b>%{text}</b><br>Reviews: %{x}<br>Rating: %{y:.1f}<extra></extra>'
            ),
            row=1, col=2
        )
        
        fig3.update_layout(
            height=500,
            title_text="⭐ Supplier Quality and Reputation Analysis"
        )
        
        figures.append(fig3)
        
        return figures

def create_enhanced_metrics_dashboard(companies: List[Dict]) -> str:
    """Create enhanced metrics dashboard HTML"""
    if not companies:
        return ""
    
    df = pd.DataFrame(companies)
    
    # Calculate advanced metrics
    total_companies = len(companies)
    high_relevance = len(df[df['total_score'] >= 10])
    small_businesses = len(df[df['size'].str.contains('Small', na=False)])
    avg_distance = df['distance_miles'].mean()
    avg_rating = df['rating'].mean()
    
    # Get top capability safely
    all_caps = []
    for caps in df['capabilities']:
        all_caps.extend(caps)
    top_capability = pd.Series(all_caps).value_counts().index[0] if all_caps else "N/A"
    
    # Calculate quality metrics
    quality_suppliers = len(df[(df['rating'] >= 4.0) & (df['user_ratings_total'] >= 10)])
    defense_contractors = len(df[df['industry'].str.contains('Defense|Naval|Aerospace', case=False, na=False)])
    
    return f"""
    <div class="metric-grid">
        <div class="metric-card">
            <p class="metric-value">{total_companies}</p>
            <p class="metric-label">Total Suppliers Found</p>
        </div>
        <div class="metric-card">
            <p class="metric-value">{high_relevance}</p>
            <p class="metric-label">High Naval Relevance</p>
        </div>
        <div class="metric-card">
            <p class="metric-value">{small_businesses}</p>
            <p class="metric-label">Small Businesses</p>
        </div>
        <div class="metric-card">
            <p class="metric-value">{avg_distance:.1f} mi</p>
            <p class="metric-label">Average Distance</p>
        </div>
        <div class="metric-card">
            <p class="metric-value">{quality_suppliers}</p>
            <p class="metric-label">Quality Suppliers</p>
        </div>
        <div class="metric-card">
            <p class="metric-value">{defense_contractors}</p>
            <p class="metric-label">Defense Focus</p>
        </div>
        <div class="metric-card">
            <p class="metric-value">{avg_rating:.1f}⭐</p>
            <p class="metric-label">Average Rating</p>
        </div>
        <div class="metric-card">
            <p class="metric-value">{top_capability[:15]}</p>
            <p class="metric-label">Top Capability</p>
        </div>
    </div>
    """

def generate_executive_report(companies: List[Dict], config: SearchConfig) -> str:
    """Generate comprehensive executive report"""
    if not companies:
        return "No companies found for analysis."
    
    df = pd.DataFrame(companies)
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Calculate key metrics
    total_companies = len(companies)
    high_value_companies = len(df[df['total_score'] >= 10])
    defense_companies = len(df[df['industry'].str.contains('Defense|Naval|Aerospace', case=False, na=False)])
    small_businesses = len(df[df['size'].str.contains('Small', na=False)])
    
    # Top companies
    top_companies = df.nlargest(5, 'total_score')
    
    # Capability analysis
    all_capabilities = []
    for caps in df['capabilities']:
        all_capabilities.extend(caps)
    top_capabilities = pd.Series(all_capabilities).value_counts().head(5)
    
    report = f"""
# 🎯 WBI Naval Search - Executive Intelligence Report

**Generated:** {current_time}  
**Search Location:** {config.base_location}  
**Search Radius:** {config.radius_miles} miles  

## 📊 Executive Summary

The naval supplier intelligence search identified **{total_companies} companies** within the specified {config.radius_miles}-mile radius of {config.base_location}. Analysis reveals a robust supplier ecosystem with significant capabilities in naval and defense manufacturing.

### Key Findings:
- **{high_value_companies}** companies demonstrate high naval relevance (score ≥ 10)
- **{defense_companies}** companies have direct defense/aerospace focus
- **{small_businesses}** small businesses identified for potential partnership opportunities
- **{df['distance_miles'].mean():.1f} miles** average supplier distance from base location

## 🏆 Top-Tier Naval Suppliers

The following companies represent the highest-value potential partners based on naval relevance scoring:

"""
    
    for i, (_, company) in enumerate(top_companies.iterrows(), 1):
        report += f"""
### {i}. {company['name']}
- **Naval Relevance Score:** {company['total_score']:.1f}/100
- **Location:** {company['location']} ({company['distance_miles']:.1f} miles)
- **Size:** {company['size']}
- **Key Capabilities:** {', '.join(company['capabilities'][:3])}
- **Quality Rating:** {company['rating']:.1f}⭐ ({company['user_ratings_total']} reviews)

"""
    
    report += f"""
## 🔧 Capability Landscape Analysis

The supplier base demonstrates strong capabilities in the following areas:

"""
    
    for capability, count in top_capabilities.items():
        percentage = (count / total_companies) * 100
        report += f"- **{capability}:** {count} companies ({percentage:.1f}% of suppliers)\n"
    
    report += f"""

## 📈 Strategic Recommendations

### Immediate Actions:
1. **Engage Top-Tier Suppliers:** Initiate contact with the top {min(5, high_value_companies)} highest-scoring companies
2. **Small Business Outreach:** Develop partnership programs with {small_businesses} identified small businesses
3. **Capability Gap Analysis:** Assess coverage in critical naval technologies

### Long-term Strategy:
1. **Regional Hub Development:** Leverage the {total_companies} supplier concentration around {config.base_location}
2. **Supply Chain Resilience:** Diversify supplier base across {len(df['size'].unique())} different company sizes
3. **Innovation Pipeline:** Engage with companies demonstrating advanced capabilities in unmanned systems and automation

## 📋 Data Quality Assessment

- **Geographic Coverage:** {config.radius_miles}-mile radius provides comprehensive regional coverage
- **Supplier Diversity:** {len(df['size'].unique())} different company size categories represented
- **Industry Breadth:** {len(df['industry'].unique())} distinct industry classifications
- **Quality Metrics:** Average supplier rating of {df['rating'].mean():.1f}⭐ indicates reliable partner pool

---
*This report was generated by WBI Naval Search - Advanced Supplier Intelligence Platform*
"""
    
    return report

def main():
    """Enhanced main application with advanced features"""
    st.set_page_config(
        page_title="WBI Naval Search - Advanced Supplier Intelligence Platform",
        page_icon="⚓",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    def reset_search_state():
        """Clear previous search results"""
        if 'companies' in st.session_state:
            del st.session_state.companies
        if 'searcher' in st.session_state:
            del st.session_state.searcher
        if 'search_triggered' in st.session_state:
            del st.session_state.search_triggered

    # Load WBI logo
    try:
        import base64
        with open("logos/wbi-logo-horz.png", "rb") as logo_file:
            logo_base64 = base64.b64encode(logo_file.read()).decode()
    except Exception as e:
        st.error(f"Logo not found: {e}")
        logo_base64 = ""  # Fallback if logo not found  
    
    # Initialize session state
    if 'search_triggered' not in st.session_state:
        st.session_state.search_triggered = False
    if 'companies' not in st.session_state:
        st.session_state.companies = []
    if 'last_search_config' not in st.session_state:
        st.session_state.last_search_config = None
    
    # Enhanced styling
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    .stApp {
        font-family: 'Inter', sans-serif;
        background-color: #1a202c;
        color: #ffffff;
    }
    
    .stDeployButton {display: none;}
    header[data-testid="stHeader"] {display: none;}
    .stMainBlockContainer {padding-top: 1rem;}
    
    /* Enhanced WBI Header styling */
    .wbi-header {
        background: linear-gradient(135deg, #1a202c 0%, #2d3748 100%);
        padding: 2rem 1rem;
        box-shadow: 0 10px 40px rgba(0, 0, 0, 0.15);
        margin-bottom: 0;
        position: relative;
    }
    
    .wbi-logo-container {
        background: #ffffff;
        border-radius: 12px;
        padding: 1rem 2rem;
        margin: 0 auto 1.5rem auto;   /* centers the box */
        display: flex;
        justify-content: center;
        align-items: center;
        width: fit-content;           /* shrink to logo size */
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }
    .wbi-logo-img{
        display:block;
        max-height:80px;          /* ⬅️ prevents huge logo */
        max-width:320px;
        height:auto; width:auto;
    }
    
    .wbi-logo {
        height: 6rem;
        font-size: 3.5rem;
        color: #1a202c;
        font-weight: 800;
        display: flex;
        align-items: center;
        gap: 1rem;
        text-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }
    
    .wbi-header h1 {
        color: #ffffff !important;
        font-size: 2.25rem !important;
        font-weight: 700 !important;
        text-align: center;
        margin: 0 !important;
        text-shadow: 0 2px 8px rgba(0, 0, 0, 0.3);
    }
    
    .wbi-header p {
        color: #cbd5e0 !important;
        text-align: center;
        margin-top: 0.75rem !important;
        margin-bottom: 0 !important;
        max-width: 36rem;
        margin-left: auto !important;
        margin-right: auto !important;
        font-size: 1.1rem;
        line-height: 1.6;
    }
    
    .wbi-border {
        border-top: 4px solid #2563eb;
        background: linear-gradient(90deg, #2563eb 0%, #3b82f6 50%, #2563eb 100%);
        margin-bottom: 2rem;
    }
    
    .wbi-card {
        background: #2d3748;
        border-radius: 1rem;
        box-shadow: 0 10px 40px rgba(0, 0, 0, 0.3);
        padding: 2rem;
        border: 1px solid #4a5568;
        margin: 1rem 0;
    }
    
    .wbi-card h3 {
        color: #ffffff !important;
        font-weight: 600 !important;
        margin-bottom: 1rem !important;
    }
    
    .wbi-card h4 {
        color: #e2e8f0 !important;
        font-weight: 600 !important;
    }
    
    .wbi-card p {
        color: #cbd5e0 !important;
        line-height: 1.6;
    }
    
    /* Enhanced progress bar */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #2563eb 0%, #3b82f6 100%) !important;
        height: 8px !important;
        border-radius: 4px !important;
    }
    
    .stProgress > div > div > div {
        background-color: #e5e7eb !important;
        border-radius: 4px !important;
    }
    
    /* Enhanced button styling */
    .stButton > button {
        background: linear-gradient(135deg, #1e40af 0%, #2563eb 100%) !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 0.75rem !important;
        padding: 0.875rem 1.5rem !important;
        font-weight: 600 !important;
        font-size: 1rem !important;
        transition: all 0.3s ease !important;
        width: 100% !important;
        font-family: 'Inter', sans-serif !important;
        box-shadow: 0 4px 14px rgba(37, 99, 235, 0.3) !important;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    .stButton > button:hover {
        background: linear-gradient(135deg, #1d4ed8 0%, #2563eb 100%) !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px rgba(37, 99, 235, 0.4) !important;
    }
    
    /* Enhanced tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0;
        border-bottom: 2px solid #4a5568;
        background-color: #2d3748;
        border-radius: 0.5rem 0.5rem 0 0;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: transparent;
        border: none;
        padding: 1rem 1.5rem;
        color: #cbd5e0 !important;
        font-weight: 500;
        border-bottom: 3px solid transparent;
        font-family: 'Inter', sans-serif;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        color: #ffffff !important;
        background-color: #374151;
    }
    
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        color: #ffffff !important;
        font-weight: 600;
        border-bottom: 3px solid #2563eb !important;
        background-color: #1a202c;
    }
    
    /* Enhanced metrics styling */
    .metric-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 1.5rem;
        margin: 2rem 0;
    }
    
    .metric-card {
        background: #2d3748;
        border-radius: 1rem;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        padding: 1.5rem;
        border: 1px solid #4a5568;
        text-align: center;
        transition: transform 0.2s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 12px 40px rgba(0, 0, 0, 0.4);
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #ffffff !important;
        margin: 0;
        font-family: 'Inter', sans-serif;
    }
    
    .metric-label {
        font-size: 0.75rem;
        color: #cbd5e0 !important;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin: 0.5rem 0 0 0;
    }
    
    /* Enhanced sidebar styling */
    .stSidebar > div:first-child {
        background-color: #2d3748;
        border-right: 2px solid #4a5568;
    }
    
    .stSidebar .stSelectbox label,
    .stSidebar .stSlider label,
    .stSidebar .stTextInput label,
    .stSidebar .stTextArea label,
    .stSidebar .stRadio label {
        color: #ffffff !important;
        font-weight: 600 !important;
    }
    
    /* Enhanced info messages */
    .stInfo {
        background-color: #dbeafe !important;
        border: 1px solid #93c5fd !important;
        border-radius: 0.5rem !important;
        color: #1e40af !important;
    }
    
    .stSuccess {
        background-color: #dcfce7 !important;
        border: 1px solid #86efac !important;
        border-radius: 0.5rem !important;
        color: #166534 !important;
    }
    
    .stWarning {
        background-color: #fef3c7 !important;
        border: 1px solid #fcd34d !important;
        border-radius: 0.5rem !important;
        color: #92400e !important;
    }
    
    .stError {
        background-color: #fee2e2 !important;
        border: 1px solid #fca5a5 !important;
        border-radius: 0.5rem !important;
        color: #dc2626 !important;
    }
    
    /* Enhanced text contrast */
    .main .block-container h1,
    .main .block-container h2,
    .main .block-container h3,
    .main .block-container h4 {
        color: #ffffff !important;
        font-weight: 600 !important;
    }
    
    .main .block-container p,
    .main .block-container li,
    .main .block-container span {
        color: #e2e8f0 !important;
    }
    
    /* Enhanced expander styling */
    .streamlit-expanderHeader {
        background-color: #2d3748 !important;
        border: 1px solid #4a5568 !important;
        border-radius: 0.5rem !important;
        color: #ffffff !important;
        font-weight: 600 !important;
    }
    
    .streamlit-expanderContent {
        background-color: #1a202c !important;
        border: 1px solid #4a5568 !important;
        border-top: none !important;
        color: #e2e8f0 !important;
    }
    
    /* Enhanced DataFrame styling */
    .stDataFrame {
        border: 1px solid #4a5568;
        border-radius: 0.5rem;
        overflow: hidden;
        background-color: #2d3748;
    }
    
    /* Performance indicator styling */
    .performance-indicator {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 1rem;
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    .indicator-high {
        background-color: #10b981;
        color: white;
    }
    
    .indicator-medium {
        background-color: #f59e0b;
        color: white;
    }
    
    .indicator-low {
        background-color: #ef4444;
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Check for geopy availability
    if not GEOPY_AVAILABLE:
        st.error("⚠️ Missing required package: geopy")
        st.info("Please install geopy: `pip install geopy`")
        st.stop()
    
    # WBI header with logo in white box (no st.image here)
    st.markdown(f"""
    <div class="wbi-header">
        <div class="wbi-logo-container">
            {'<img class="wbi-logo-img" src="data:image/png;base64,' + logo_base64 + '" alt="WBI Logo"/>' if logo_base64 else '<div style="color:#1a202c;font-weight:800;">⚓ WBI</div>'}
        </div>
        <h1 style="color:#ffffff;text-align:center;margin:0;">Naval Search Pro</h1>
        <p style="color:#cbd5e0;text-align:center;margin-top:.75rem;">
            Advanced supplier intelligence and procurement analytics platform for naval operations.
        </p>
    </div>
    <div class="wbi-border"></div>
    """, unsafe_allow_html=True)

    # Enhanced sidebar configuration
    st.sidebar.header("🔧 Advanced Search Configuration")
    strict_naval = st.sidebar.checkbox("Strict Naval Filter (recommended)", value=True)
    st.session_state.strict_naval = strict_naval

    # API Key management
    if not GOOGLE_PLACES_API_KEY:
        st.sidebar.warning("⚠️ No API key detected")
        with st.sidebar.expander("🔑 API Key Setup", expanded=True):
            st.markdown("""
            **Quick Setup Guide:**
            1. Visit [Google Cloud Console](https://console.cloud.google.com)
            2. Enable **Places API (New)** + Billing
            3. Create API key and paste below:
            """)
            api_key_input = st.sidebar.text_input(
                "Google Places API Key:", 
                type="password",
                help="Enter your Google Places API key for real company search"
            )
            if api_key_input:
                st.session_state.api_key = api_key_input
                st.sidebar.success("✅ API key configured! Real search enabled.")
    else:
        st.sidebar.success("✅ API key configured - using real company search")
    
    # Initialize configuration
    config = SearchConfig()
    
    # Location selection with enhanced options
    st.sidebar.subheader("📍 Search Location Configuration")
    
    location_option = st.sidebar.radio(
        "Choose search center:",
        ["🎯 Preset Naval Locations", "🗺️ Custom Location", "📊 Multi-Location Analysis"],
        help="Select how you want to specify the search area"
    )
    
    if location_option == "🎯 Preset Naval Locations":
        preset_locations = {
            "South Bend, Indiana": "South Bend, Indiana",
            "Norfolk, Virginia (Naval Station Norfolk)": "Norfolk, Virginia",
            "San Diego, California (Naval Base San Diego)": "San Diego, California", 
            "Pearl Harbor, Hawaii (Joint Base Pearl Harbor)": "Pearl Harbor, Hawaii",
            "Bremerton, Washington (Puget Sound Naval Shipyard)": "Bremerton, Washington",
            "Portsmouth, New Hampshire (Portsmouth Naval Shipyard)": "Portsmouth, New Hampshire",
            "Newport News, Virginia (Newport News Shipbuilding)": "Newport News, Virginia",
            "Bath, Maine (Bath Iron Works)": "Bath, Maine",
            "Groton, Connecticut (Electric Boat)": "Groton, Connecticut",
            "Pascagoula, Mississippi (Ingalls Shipbuilding)": "Pascagoula, Mississippi",
            "Kings Bay, Georgia (Naval Submarine Base)": "Kings Bay, Georgia",
            "Bangor, Washington (Naval Base Kitsap)": "Bangor, Washington"
        }
        
        selected_preset = st.sidebar.selectbox(
            "Select naval facility:",
            list(preset_locations.keys()),
            help="Choose from major naval installations and shipyards"
        )
        config.base_location = preset_locations[selected_preset]
        
    elif location_option == "🗺️ Custom Location":
        config.base_location = st.sidebar.text_input(
            "Enter custom location:",
            value="South Bend, Indiana",
            help="Enter city, state (e.g., 'Detroit, Michigan')"
        )
        
        # Coordinate validation
        if config.base_location:
            with st.sidebar.expander("📍 Location Validation"):
                searcher_temp = EnhancedCompanySearcher(config)
                coords = searcher_temp.base_coords
                st.write(f"📍 Coordinates: {coords[0]:.4f}, {coords[1]:.4f}")
                st.write(f"🗺️ Validated: {config.base_location}")
    
    else:  # Multi-Location Analysis
        st.sidebar.info("🚧 Multi-location analysis coming in next update!")
        config.base_location = "South Bend, Indiana"
    
    # Enhanced search parameters
    st.sidebar.subheader("🎯 Search Parameters")
    
    col1, col2 = st.sidebar.columns(2)
    with col1:
        config.radius_miles = st.slider(
            "Search Radius (miles)", 
            10, 200, 60,
            help="Geographic radius for supplier search"
        )
    with col2:
        config.target_company_count = st.slider(
            "Target Companies", 
            10, 500, 100,
            help="Maximum number of companies to find"
        )
    
    # Advanced configuration options
    with st.sidebar.expander("⚙️ Advanced Configuration"):
        config.enable_async_search = st.checkbox(
            "Enable Async Search", 
            value=True,
            help="Use asynchronous search for better performance"
        )
        config.enable_caching = st.checkbox(
            "Enable Result Caching", 
            value=True,
            help="Cache results to improve performance"
        )
        config.cache_max_age_hours = st.slider(
            "Cache Age (hours)", 
            1, 168, 24,
            help="Maximum age of cached results"
        )
        config.max_concurrent_requests = st.slider(
            "Concurrent Requests", 
            1, 10, 5,
            help="Number of simultaneous API requests"
        )
    
    # Enhanced keyword customization
    with st.sidebar.expander("🎯 Search Keywords Configuration"):
        st.markdown("**Manufacturing Keywords:**")
        manufacturing_keywords_str = st.text_area(
            "Manufacturing",
            value=", ".join(config.manufacturing_keywords),
            height=100,
            help="Keywords for manufacturing company identification"
        )
        if manufacturing_keywords_str:
            config.manufacturing_keywords = [k.strip() for k in manufacturing_keywords_str.split(",")]
        
        st.markdown("**Robotics & Automation Keywords:**")
        robotics_keywords_str = st.text_area(
            "Robotics", 
            value=", ".join(config.robotics_keywords),
            height=80
        )
        if robotics_keywords_str:
            config.robotics_keywords = [k.strip() for k in robotics_keywords_str.split(",")]
        
        st.markdown("**Unmanned Systems Keywords:**")
        unmanned_keywords_str = st.text_area(
            "Unmanned Systems",
            value=", ".join(config.unmanned_keywords),
            height=80
        )
        if unmanned_keywords_str:
            config.unmanned_keywords = [k.strip() for k in unmanned_keywords_str.split(",")]
        
        st.markdown("**Workforce & Training Keywords:**")
        workforce_keywords_str = st.text_area(
            "Workforce Training",
            value=", ".join(config.workforce_keywords),
            height=100
        )
        if workforce_keywords_str:
            config.workforce_keywords = [k.strip() for k in workforce_keywords_str.split(",")]
    
    # Search execution with enhanced status
    search_col1, search_col2 = st.sidebar.columns([3, 1])
    with search_col1:
        if st.button("🔍 Execute Naval Search", type="primary"):
            reset_search_state()  # Clear previous results
            st.session_state.search_triggered = True
            st.session_state.last_search_config = config
    
    with search_col2:
        if st.button("🔄", help="Reset Search"):
            reset_search_state()
            st.rerun()
    
    # Main content area with enhanced layout
    main_col1, main_col2 = st.columns([3, 1])
    
    with main_col2:
        st.markdown("### ⚡ Naval Procurement Intelligence")
        st.markdown("Advanced supplier discovery and market intelligence for naval operations.")
        st.markdown("**🔍 Discover:** Intelligent supplier discovery using advanced natural language processing.")
        st.markdown("**📊 Analyze:** Multi-dimensional scoring system evaluating naval relevance and capability alignment.")
        st.markdown("**🚀 Execute:** Actionable intelligence for procurement decisions with comprehensive supplier profiles.")
        
        # Performance monitoring
        if st.session_state.get('companies'):
            st.markdown("""
            <div class="wbi-card">
                <h3>📈 Search Performance</h3>
            </div>
            """, unsafe_allow_html=True)
            
            performance_metrics = {
                "Search Quality": "High" if len(st.session_state.companies) > 50 else "Medium",
                "Data Freshness": "Current" if config.enable_caching else "Live",
                "Coverage": f"{config.radius_miles} mi radius"
            }
            
            for metric, value in performance_metrics.items():
                indicator_class = "indicator-high" if value in ["High", "Current"] else "indicator-medium"
                st.markdown(f"""
                <div style="margin: 0.5rem 0;">
                    <strong>{metric}:</strong> 
                    <span class="performance-indicator {indicator_class}">{value}</span>
                </div>
                """, unsafe_allow_html=True)
    
    with main_col1:
    # Execute search when triggered - FIXED VERSION
        if st.session_state.get('search_triggered', False):
            if not st.session_state.get('companies'):
                try:
                    with st.spinner("AI agent searching and validating companies..."):
                        searcher = EnhancedCompanySearcher(config)
                        companies = searcher.search_companies()
                        st.session_state.companies = companies
                        st.session_state.searcher = searcher
                
                    # Mark search as completed
                    st.session_state.search_triggered = False
                
                    # Show success message
                    if companies:
                        st.success(f"Found {len(companies)} validated naval suppliers")
                    else:
                        st.warning("No relevant suppliers found. Try a different location or radius.")
                    
                except Exception as e:
                    st.error(f"Search error: {str(e)}")
                    st.session_state.search_triggered = False
            else:
                # Search already completed, just reset the trigger
                st.session_state.search_triggered = False
    
    # Display enhanced results
    if st.session_state.get('companies'):
        companies = st.session_state.companies
        searcher = st.session_state.searcher
        
        # Enhanced metrics dashboard
        st.markdown("## 📊 Naval Supplier Intelligence Dashboard")
        metrics_html = create_enhanced_metrics_dashboard(companies)
        st.markdown(metrics_html, unsafe_allow_html=True)
        
        # Enhanced tabs with more features
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "📋 Supplier Directory", 
            "🗺️ Geographic Intelligence", 
            "📊 Market Analytics", 
            "🎯 Strategic Analysis",
            "📈 Executive Dashboard",
            "📄 Intelligence Export"
        ])
        
        with tab1:
            st.subheader("🏭 Enhanced Naval Supplier Directory")
            st.markdown("*AI-powered supplier intelligence with comprehensive filtering and analysis*")
            
            # Enhanced filters
            filter_col1, filter_col2, filter_col3, filter_col4 = st.columns(4)
            
            with filter_col1:
                min_score = st.slider("Min Naval Relevance", 0, 50, 0)
            with filter_col2:
                size_filter = st.selectbox(
                    "Company Size", 
                    ["All", "Small Business", "Small-Medium Business", "Medium Business", "Large Corporation", "Fortune 500 / Major Corporation"]
                )
            with filter_col3:
                max_distance = st.slider("Max Distance (mi)", 0, config.radius_miles, config.radius_miles)
            with filter_col4:
                min_rating = st.slider("Min Rating", 0.0, 5.0, 0.0, 0.1)
            
            # Filter companies
            filtered_companies = companies.copy()
            if min_score > 0:
                filtered_companies = [c for c in filtered_companies if c['total_score'] >= min_score]
            if size_filter != "All":
                filtered_companies = [c for c in filtered_companies if c['size'] == size_filter]
            if max_distance < config.radius_miles:
                filtered_companies = [c for c in filtered_companies if c['distance_miles'] <= max_distance]
            if min_rating > 0:
                filtered_companies = [c for c in filtered_companies if c['rating'] >= min_rating]
            
            st.info(f"📊 Showing {len(filtered_companies)} suppliers (filtered from {len(companies)} total)")
            
            # Enhanced company display
            for i, company in enumerate(filtered_companies):
                score_color = "🟢" if company['total_score'] >= 15 else "🟡" if company['total_score'] >= 8 else "🔴"
                
                with st.expander(f"{score_color} {company['name']} - Naval Score: {company['total_score']:.1f}"):
                    # Company header with key metrics
                    metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
                    with metric_col1:
                        st.metric("Naval Relevance", f"{company['total_score']:.1f}")
                    with metric_col2:
                        st.metric("Distance", f"{company['distance_miles']:.1f} mi")
                    with metric_col3:
                        st.metric("Rating", f"{company['rating']:.1f}⭐")
                    with metric_col4:
                        st.metric("Reviews", company['user_ratings_total'])
                    
                    # In the company display section, add this debug info:
                    st.markdown("**🔍 Score Breakdown Debug:**")
                    st.write(f"Manufacturing: {company['manufacturing_score']}")
                    st.write(f"Robotics: {company['robotics_score']}")
                    st.write(f"Unmanned: {company['unmanned_score']}")
                    st.write(f"Workforce: {company['workforce_score']}")
                    st.write(f"**Total: {company['total_score']}** (Should equal sum above)")

                    # Company details
                    detail_col1, detail_col2 = st.columns(2)
                    with detail_col1:
                        st.markdown("**📍 Location & Contact**")
                        st.write(f"📍 {company['location']}")
                        st.write(f"🌐 {company['website']}")
                        st.write(f"📞 {company['phone']}")
                        st.write(f"🏢 Size: {company['size']}")
                        st.write(f"🏭 Industry: {company['industry']}")
                    
                    with detail_col2:
                        st.markdown("**🔧 Capabilities & Specializations**")
                        for capability in company['capabilities']:
                            st.write(f"• {capability}")
                    
                    st.markdown(f"**📋 Company Description**")
                    st.write(company['description'])
                    
                    # Detailed scoring breakdown
                    st.markdown("**🎯 Naval Relevance Analysis**")
                    score_col1, score_col2, score_col3, score_col4, score_col5 = st.columns(5)
                    with score_col1:
                        st.metric("🏭 Manufacturing", f"{company['manufacturing_score']:.1f}")
                    with score_col2:
                        st.metric("🤖 Robotics", f"{company['robotics_score']:.1f}")
                    with score_col3:
                        st.metric("🚁 Unmanned", f"{company['unmanned_score']:.1f}")
                    with score_col4:
                        st.metric("👥 Workforce", f"{company['workforce_score']:.1f}")
                    with score_col5:
                        st.metric("🛡️ Defense", f"{company.get('defense_score', 0):.1f}")
        
        with tab2:
            st.subheader("🗺️ Advanced Geographic Intelligence")
            st.markdown("*Spatial analysis of naval supplier distribution and strategic positioning*")
            
            map_fig = EnhancedVisualization.create_advanced_company_map(companies, searcher.base_coords)
            if map_fig:
                st.plotly_chart(map_fig, use_container_width=True)
                
                # Geographic insights
                st.markdown("### 📈 Geographic Intelligence Insights")
                avg_distance = np.mean([c['distance_miles'] for c in companies])
                closest_supplier = min(companies, key=lambda x: x['distance_miles'])
                farthest_supplier = max(companies, key=lambda x: x['distance_miles'])
                
                geo_col1, geo_col2, geo_col3 = st.columns(3)
                with geo_col1:
                    st.metric("Average Distance", f"{avg_distance:.1f} mi")
                with geo_col2:
                    st.metric("Closest Supplier", f"{closest_supplier['distance_miles']:.1f} mi")
                    st.caption(closest_supplier['name'])
                with geo_col3:
                    st.metric("Farthest Supplier", f"{farthest_supplier['distance_miles']:.1f} mi")
                    st.caption(farthest_supplier['name'])
            else:
                st.info("No geographic data available for mapping")
        
        with tab3:
            st.subheader("📊 Comprehensive Market Analytics")
            st.markdown("*Advanced statistical analysis of naval supplier landscape*")
            
            analytics_figures = EnhancedVisualization.create_comprehensive_analytics_dashboard(companies)
            
            for fig in analytics_figures:
                st.plotly_chart(fig, use_container_width=True)
            
            # Market insights
            if companies:
                df = pd.DataFrame(companies)
                
                st.markdown("### 💡 Market Intelligence Insights")
                
                insight_col1, insight_col2 = st.columns(2)
                with insight_col1:
                    st.markdown("**🎯 High-Value Opportunities**")
                    top_companies = df.nlargest(3, 'total_score')
                    for _, company in top_companies.iterrows():
                        st.write(f"• **{company['name']}** (Score: {company['total_score']:.1f})")
                
                with insight_col2:
                    st.markdown("**🏢 Small Business Opportunities**")
                    small_biz = df[df['size'].str.contains('Small', na=False)]
                    if not small_biz.empty:
                        top_small = small_biz.nlargest(3, 'total_score')
                        for _, company in top_small.iterrows():
                            st.write(f"• **{company['name']}** (Score: {company['total_score']:.1f})")
                    else:
                        st.write("No small businesses found in current search")
        
        with tab4:
            st.subheader("🎯 Strategic Partnership Analysis")
            st.markdown("*AI-powered strategic recommendations for naval procurement optimization*")
            
            if companies:
                df = pd.DataFrame(companies)
                
                # Strategic recommendations
                st.markdown("### 🚀 Strategic Recommendations")
                
                rec_col1, rec_col2 = st.columns(2)
                
                with rec_col1:
                    st.markdown("**🎯 Immediate Action Items**")
                    high_value = df[df['total_score'] >= 15]
                    if not high_value.empty:
                        st.success(f"✅ {len(high_value)} high-value suppliers identified for immediate engagement")
                        st.markdown("**Priority Contacts:**")
                        for _, company in high_value.head(3).iterrows():
                            st.write(f"• {company['name']} - {company['phone']}")
                    
                    defense_focused = df[df['industry'].str.contains('Defense|Naval|Aerospace', case=False, na=False)]
                    if not defense_focused.empty:
                        st.info(f"🛡️ {len(defense_focused)} defense-focused suppliers for strategic partnerships")
                
                with rec_col2:
                    st.markdown("**📈 Growth Opportunities**")
                    small_high_potential = df[(df['size'].str.contains('Small', na=False)) & (df['total_score'] >= 8)]
                    if not small_high_potential.empty:
                        st.success(f"🌱 {len(small_high_potential)} small businesses with high growth potential")
                    
                    training_orgs = df[df['workforce_score'] > 5]
                    if not training_orgs.empty:
                        st.info(f"🎓 {len(training_orgs)} training organizations for workforce development")
                
                # Risk assessment
                st.markdown("### ⚠️ Risk Assessment & Mitigation")
                
                risk_col1, risk_col2, risk_col3 = st.columns(3)
                
                with risk_col1:
                    st.markdown("**🔴 High Risk**")
                    low_rated = df[(df['rating'] < 3.5) & (df['user_ratings_total'] > 5)]
                    st.metric("Low-Rated Suppliers", len(low_rated))
                    if not low_rated.empty:
                        st.caption("Suppliers with concerning ratings")
                
                with risk_col2:
                    st.markdown("**🟡 Medium Risk**")
                    unproven = df[df['user_ratings_total'] < 5]
                    st.metric("Unproven Suppliers", len(unproven))
                    if not unproven.empty:
                        st.caption("Limited track record")
                
                with risk_col3:
                    st.markdown("**🟢 Low Risk**")
                    proven = df[(df['rating'] >= 4.0) & (df['user_ratings_total'] >= 10)]
                    st.metric("Proven Suppliers", len(proven))
                    if not proven.empty:
                        st.caption("Established reputation")
        
        with tab5:
            st.subheader("📈 Executive Intelligence Dashboard")
            st.markdown("*C-suite level insights and strategic intelligence for naval procurement leadership*")
            
            if companies:
                df = pd.DataFrame(companies)
                
                # Executive summary metrics
                st.markdown("### 📊 Executive Summary")
                
                exec_col1, exec_col2, exec_col3, exec_col4 = st.columns(4)
                
                with exec_col1:
                    total_suppliers = len(companies)
                    st.metric(
                        "Total Suppliers Identified",
                        total_suppliers,
                        help="Complete supplier universe in search radius"
                    )
                
                with exec_col2:
                    strategic_suppliers = len(df[df['total_score'] >= 10])
                    strategic_percentage = (strategic_suppliers / total_suppliers) * 100
                    st.metric(
                        "Strategic Suppliers",
                        strategic_suppliers,
                        f"{strategic_percentage:.1f}% of total",
                        help="High-value suppliers for strategic partnerships"
                    )
                
                with exec_col3:
                    defense_contractors = len(df[df['industry'].str.contains('Defense|Naval|Aerospace', case=False, na=False)])
                    defense_percentage = (defense_contractors / total_suppliers) * 100
                    st.metric(
                        "Defense Contractors",
                        defense_contractors,
                        f"{defense_percentage:.1f}% of total",
                        help="Suppliers with defense industry focus"
                    )
                
                with exec_col4:
                    avg_supplier_quality = df['rating'].mean()
                    st.metric(
                        "Avg Supplier Quality",
                        f"{avg_supplier_quality:.1f}⭐",
                        help="Average quality rating across all suppliers"
                    )
        
        with tab6:
            st.subheader("📄 Comprehensive Intelligence Export")
            st.markdown("*Export comprehensive supplier data for strategic decision-making and procurement planning*")
            
            if companies:
                df = pd.DataFrame(companies)
                
                # Export options
                export_col1, export_col2 = st.columns(2)
                
                with export_col1:
                    st.markdown("### 📊 Data Export Options")
                    
                    # Enhanced CSV with better formatting
                    export_df = df[['name', 'location', 'industry', 'size', 'total_score', 
                                    'manufacturing_score', 'robotics_score', 'unmanned_score', 
                                    'workforce_score', 'distance_miles', 'rating', 'website', 
                                    'phone']].copy()

                    # Clean column names
                    export_df.columns = ['Company Name', 'Location', 'Industry', 'Company Size', 
                                        'Naval Relevance Score', 'Manufacturing Score', 'Robotics Score', 
                                        'Unmanned Systems Score', 'Workforce Training Score', 'Distance (Miles)', 
                                        'Quality Rating', 'Website', 'Phone Number']

                    # Format numeric columns to 1 decimal place
                    numeric_cols = ['Naval Relevance Score', 'Manufacturing Score', 'Robotics Score', 
                                    'Unmanned Systems Score', 'Workforce Training Score', 'Distance (Miles)', 'Quality Rating']
                    for col in numeric_cols:
                        export_df[col] = export_df[col].round(1)

                    # Add search metadata at the end
                    export_df['Search Location'] = config.base_location
                    export_df['Search Date'] = datetime.now().strftime("%Y-%m-%d")

                    csv_data = export_df.to_csv(index=False)
                    
                    st.download_button(
                        label="📥 Download Enhanced CSV Report",
                        data=csv_data,
                        file_name=f"wbi_naval_intelligence_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                        mime="text/csv",
                        help="Complete supplier data with enhanced analytics"
                    )
                    
                    # Executive summary export
                    exec_report = generate_executive_report(companies, config)
                    st.download_button(
                        label="📋 Download Executive Report",
                        data=exec_report,
                        file_name=f"wbi_executive_report_{datetime.now().strftime('%Y%m%d_%H%M')}.md",
                        mime="text/markdown",
                        help="Executive-level strategic analysis and recommendations"
                    )
                
                with export_col2:
                    st.markdown("### 📈 Report Preview")
                    with st.expander("Preview Executive Intelligence Report", expanded=False):
                        exec_report_preview = generate_executive_report(companies, config)
                        st.markdown(exec_report_preview)

    # Footer with enhanced information
    st.markdown("---")
    footer_col1, footer_col2, footer_col3 = st.columns(3)
    
    with footer_col1:
        st.markdown("""
        **🎯 WBI Naval Search Pro**  
        Advanced Supplier Intelligence Platform  
        Version 2.0 Enhanced
        """)
    
    with footer_col2:
        st.markdown("""
        **🔧 Features**  
        • AI-Powered Search  
        • Real-time Analytics  
        • Strategic Intelligence
        """)
    
    with footer_col3:
        st.markdown("""
        **📊 Performance**  
        • Multi-threaded Processing  
        • Advanced Caching  
        • Executive Reporting
        """)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"Application error: {str(e)}")
        logger.error(f"Application startup error: {e}")
        
        # Fallback interface
        st.markdown("## 🔧 Fallback Mode")
        st.info("The application encountered an error. Please refresh the page or contact support.")
        
        if st.button("🔄 Restart Application"):
            st.rerun()