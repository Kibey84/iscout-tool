import streamlit as st
import pandas as pd
import requests
import json
import time
import os
import logging
from typing import List, Dict, Optional, Tuple, Union
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

# API Configuration
GOOGLE_PLACES_API_KEY = (
    st.secrets.get("GOOGLE_PLACES_API_KEY") if hasattr(st, 'secrets') and "GOOGLE_PLACES_API_KEY" in st.secrets else
    os.environ.get("GOOGLE_PLACES_API_KEY", "")
)

# Database setup
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
                
        except Exception as e:
            logger.error(f"Database initialization error: {e}")
    
    def save_companies(self, companies: List[Dict], search_location: str) -> bool:
        """Save companies to database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                for company in companies:
                    capabilities_str = json.dumps(company.get('capabilities', []))
                    conn.execute("""
                        INSERT OR REPLACE INTO companies 
                        (name, location, industry, description, size, capabilities, lat, lon, 
                         website, phone, rating, user_ratings_total, total_score, 
                         manufacturing_score, robotics_score, unmanned_score, workforce_score,
                         distance_miles, search_location, updated_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                    """, (
                        company['name'], company['location'], company['industry'],
                        company['description'], company['size'], capabilities_str,
                        company['lat'], company['lon'], company['website'], company['phone'],
                        company['rating'], company['user_ratings_total'], company['total_score'],
                        company['manufacturing_score'], company['robotics_score'],
                        company['unmanned_score'], company['workforce_score'],
                        company['distance_miles'], search_location
                    ))
            return True
        except Exception as e:
            logger.error(f"Error saving companies: {e}")
            return False
    
    def load_companies(self, search_location: str, max_age_hours: int = 24) -> List[Dict]:
        """Load companies from database"""
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
                        'workforce_score': row[17], 'distance_miles': row[18]
                    }
                    companies.append(company)
                
                return companies
        except Exception as e:
            logger.error(f"Error loading companies: {e}")
            return []

@dataclass
class SearchConfig:
    """Enhanced search configuration"""
    base_location: str = "South Bend, Indiana"
    radius_miles: int = 60
    target_company_count: int = 100
    enable_caching: bool = True
    cache_max_age_hours: int = 24
    
    manufacturing_keywords: List[str] = field(default_factory=lambda: [
        "advanced manufacturing", "precision machining", "metal fabrication",
        "additive manufacturing", "3D printing", "CNC machining",
        "welding", "assembly", "fabrication", "machining"
    ])
    
    robotics_keywords: List[str] = field(default_factory=lambda: [
        "robotics", "automation", "robotic welding", "industrial automation",
        "automated inspection", "robotics integration", "FANUC", "KUKA"
    ])
    
    unmanned_keywords: List[str] = field(default_factory=lambda: [
        "unmanned systems", "autonomous", "UAV", "UUV", "USV",
        "drone", "autonomous vehicles", "unmanned vehicles"
    ])
    
    workforce_keywords: List[str] = field(default_factory=lambda: [
        "naval training", "maritime training", "shipyard training",
        "welding certification", "maritime academy", "technical training"
    ])
    
    def validate(self) -> bool:
        """Validate configuration"""
        if self.radius_miles <= 0 or self.radius_miles > 200:
            raise ValueError("Radius must be between 1 and 200 miles")
        if self.target_company_count <= 0 or self.target_company_count > 500:
            raise ValueError("Target company count must be between 1 and 500")
        return True

class EnhancedCompanySearcher:
    """Enhanced company searcher with advanced features"""
    
    def __init__(self, config: SearchConfig):
        self.config = config
        self.config.validate()
        if GEOPY_AVAILABLE:
            self.geolocator = Nominatim(user_agent="wbi_naval_search_enhanced_v2")
        self.base_coords = self._get_coordinates(config.base_location)
        self.db_manager = DatabaseManager()
        self.session = requests.Session()
        
    def _get_coordinates(self, location: str) -> Tuple[float, float]:
        """Get coordinates for location"""
        if not GEOPY_AVAILABLE:
            return self._get_default_coordinates(location)
            
        try:
            location_data = self.geolocator.geocode(location, timeout=10)
            if location_data:
                return (location_data.latitude, location_data.longitude)
            else:
                return self._get_default_coordinates(location)
        except Exception as e:
            logger.error(f"Geocoding error: {e}")
            return self._get_default_coordinates(location)
    
    def _get_default_coordinates(self, location: str) -> Tuple[float, float]:
        """Fallback coordinates"""
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
        
        return (41.6764, -86.2520)  # South Bend default
    
    def _calculate_distance(self, lat: float, lon: float) -> float:
        """Calculate distance from base location"""
        if not GEOPY_AVAILABLE:
            # Simple distance approximation
            lat_diff = abs(lat - self.base_coords[0])
            lon_diff = abs(lon - self.base_coords[1])
            return (lat_diff + lon_diff) * 69  # Rough miles conversion
            
        try:
            if lat and lon and self.base_coords:
                distance = geodesic(self.base_coords, (lat, lon)).miles
                return round(distance, 2)
            return 999.0
        except Exception as e:
            logger.error(f"Distance calculation error: {e}")
            return 999.0
    
    def _score_company_relevance(self, company_data: Dict) -> Dict:
        """Score company relevance"""
        description = company_data.get('description', '').lower()
        name = company_data.get('name', '').lower()
        industry = company_data.get('industry', '').lower()
        website = company_data.get('website', '').lower()
        
        combined_text = f"{description} {name} {industry} {website}"
        
        scores = {
            'manufacturing_score': 0,
            'robotics_score': 0,
            'unmanned_score': 0,
            'workforce_score': 0,
            'defense_score': 0,
            'total_score': 0
        }
        
        # Manufacturing keywords
        manufacturing_keywords = {
            'aerospace': 10, 'defense': 10, 'naval': 12, 'military': 9,
            'shipbuilding': 15, 'precision': 8, 'cnc': 6, 'machining': 7,
            'fabrication': 6, 'welding': 5, 'manufacturing': 5
        }
        
        # Score based on keywords
        for keyword, weight in manufacturing_keywords.items():
            if keyword in combined_text:
                scores['manufacturing_score'] += weight
        
        # Major contractors bonus
        major_contractors = [
            'honeywell', 'boeing', 'lockheed', 'raytheon', 'northrop',
            'general dynamics', 'bae systems', 'textron'
        ]
        
        for contractor in major_contractors:
            if contractor in name:
                scores['manufacturing_score'] += 25
                scores['defense_score'] += 20
                break
        
        # Quality bonus
        rating = company_data.get('rating', 0)
        if rating >= 4.5:
            scores['manufacturing_score'] += 5
        elif rating >= 4.0:
            scores['manufacturing_score'] += 3
        
        # Calculate total
        scores['total_score'] = (
            scores['manufacturing_score'] +
            scores['robotics_score'] +
            scores['unmanned_score'] +
            scores['workforce_score'] +
            scores['defense_score']
        )
        
        return scores
    
    def search_companies(self) -> List[Dict]:
        """Main search function"""
        api_key = st.session_state.get('api_key', GOOGLE_PLACES_API_KEY)
        
        if not api_key:
            st.info("🎭 Using demo data. Add API key for real search.")
            return self.generate_sample_companies()
        
        # Check cache
        if self.config.enable_caching:
            cached = self.db_manager.load_companies(
                self.config.base_location, 
                self.config.cache_max_age_hours
            )
            if cached:
                st.info(f"📋 Loaded {len(cached)} companies from cache")
                return cached
        
        st.info("🔍 Searching real companies...")
        
        search_queries = self._get_search_queries()
        all_companies = []
        progress_bar = st.progress(0)
        
        for i, query in enumerate(search_queries):
            st.write(f"Searching: {query}")
            try:
                companies = self.search_google_places_text(query)
                all_companies.extend(companies)
                progress_bar.progress((i + 1) / len(search_queries))
                time.sleep(1)
            except Exception as e:
                logger.error(f"Search error for '{query}': {e}")
                continue
        
        progress_bar.empty()
        
        # Process results
        unique_companies = self._process_and_deduplicate(all_companies)
        
        # Save to cache
        if self.config.enable_caching and unique_companies:
            self.db_manager.save_companies(unique_companies, self.config.base_location)
        
        return unique_companies
    
    def _get_search_queries(self) -> List[str]:
        """Generate search queries"""
        return [
            "Honeywell Aerospace", "Boeing manufacturing", "Lockheed Martin",
            "defense contractors", "aerospace manufacturing", "naval contractors",
            "precision machining companies", "CNC machining services",
            "metal fabrication", "shipyard", "maritime training",
            "robotics integration", "unmanned systems"
        ]
    
    def search_google_places_text(self, query: str) -> List[Dict]:
        """Search Google Places"""
        api_key = st.session_state.get('api_key', GOOGLE_PLACES_API_KEY)
        
        if not api_key:
            return []
        
        url = "https://places.googleapis.com/v1/places:searchText"
        
        headers = {
            'Content-Type': 'application/json',
            'X-Goog-Api-Key': api_key,
            'X-Goog-FieldMask': 'places.displayName,places.formattedAddress,places.location,places.types,places.websiteUri,places.nationalPhoneNumber,places.rating,places.userRatingCount'
        }
        
        request_data = {
            "textQuery": f"{query} near {self.config.base_location}",
            "maxResultCount": 20
        }
        
        try:
            response = self.session.post(url, headers=headers, json=request_data, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                return self._process_places_response(data, query)
            elif response.status_code == 403:
                st.error("⚠️ API key permission error")
                return []
            else:
                logger.warning(f"Search returned status {response.status_code}")
                return []
                
        except Exception as e:
            logger.error(f"Error searching '{query}': {e}")
            return []
    
    def _process_places_response(self, data: Dict, query: str) -> List[Dict]:
        """Process API response"""
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
                
                name = place.get('displayName', {}).get('text', 'Unknown')
                types = place.get('types', [])
                
                if self._is_manufacturing_related(name, types):
                    company = {
                        'name': name,
                        'location': place.get('formattedAddress', 'Unknown'),
                        'industry': ', '.join(types[:3]) if types else 'Unknown',
                        'description': self._generate_description(name, types),
                        'size': self._determine_business_size(name, place),
                        'capabilities': self._extract_capabilities(name, types),
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
    
    def _is_manufacturing_related(self, name: str, types: List[str]) -> bool:
        """Check if business is manufacturing related"""
        name_lower = name.lower()
        types_str = ' '.join(types).lower()
        
        # Exclusions
        exclude_keywords = [
            'restaurant', 'food', 'retail', 'bank', 'insurance',
            'gas station', 'automotive repair'
        ]
        
        for exclude in exclude_keywords:
            if exclude in name_lower:
                return False
        
        # Include keywords
        include_keywords = [
            'manufacturing', 'fabrication', 'machining', 'aerospace',
            'defense', 'precision', 'engineering', 'systems',
            'training', 'academy', 'shipyard', 'maritime'
        ]
        
        for keyword in include_keywords:
            if keyword in name_lower or keyword in types_str:
                return True
        
        return False
    
    def _determine_business_size(self, name: str, place_data: Dict) -> str:
        """Determine business size"""
        name_lower = name.lower()
        review_count = place_data.get('userRatingCount', 0)
        
        major_corps = ['honeywell', 'boeing', 'lockheed', 'raytheon']
        for corp in major_corps:
            if corp in name_lower:
                return 'Large Corporation'
        
        if review_count > 100:
            return 'Large Corporation'
        elif review_count > 50:
            return 'Medium Business'
        else:
            return 'Small Business'
    
    def _extract_capabilities(self, name: str, types: List[str]) -> List[str]:
        """Extract capabilities"""
        name_lower = name.lower()
        capabilities = []
        
        capability_map = {
            'cnc': 'CNC Machining',
            'machining': 'Precision Machining',
            'welding': 'Welding Services',
            'fabrication': 'Metal Fabrication',
            'aerospace': 'Aerospace Manufacturing',
            'training': 'Training Services'
        }
        
        for keyword, capability in capability_map.items():
            if keyword in name_lower:
                capabilities.append(capability)
        
        return capabilities if capabilities else ['General Manufacturing']
    
    def _generate_description(self, name: str, types: List[str]) -> str:
        """Generate description"""
        return f"Business type: {', '.join(types[:2]) if types else 'Manufacturing'}"
    
    def _process_and_deduplicate(self, all_companies: List[Dict]) -> List[Dict]:
        """Process and deduplicate companies"""
        unique_companies = []
        seen = set()
        
        for company in all_companies:
            key = (
                company['name'].lower().strip(),
                company['location'].lower().strip()[:50]
            )
            
            if key not in seen:
                seen.add(key)
                
                distance = self._calculate_distance(company['lat'], company['lon'])
                company['distance_miles'] = distance
                
                scores = self._score_company_relevance(company)
                company.update(scores)
                
                if distance <= self.config.radius_miles and company['total_score'] >= 1:
                    unique_companies.append(company)
        
        unique_companies.sort(key=lambda x: x['total_score'], reverse=True)
        return unique_companies[:self.config.target_company_count]
    
    def generate_sample_companies(self) -> List[Dict]:
        """Generate sample companies"""
        base_lat, base_lon = self.base_coords
        location_name = self.config.base_location.split(',')[0].strip()
        state = self.config.base_location.split(',')[-1].strip() if ',' in self.config.base_location else 'IN'
        
        sample_companies = [
            {
                'name': f'{location_name} Advanced Manufacturing Solutions',
                'location': f'{location_name}, {state}',
                'industry': 'Aerospace Manufacturing, Metal Fabrication',
                'description': 'ISO 9001:2015 certified precision manufacturer specializing in aerospace components.',
                'size': 'Medium Business',
                'capabilities': ['CNC Machining', 'Aerospace Manufacturing', 'Quality Control'],
                'lat': base_lat + 0.05,
                'lon': base_lon + 0.03,
                'website': f'www.{location_name.lower()}advanced.com',
                'phone': '(555) 555-0101',
                'rating': 4.7,
                'user_ratings_total': 45
            },
            {
                'name': 'Great Lakes Robotics Integration Corp',
                'location': f'{location_name} Metro Area, {state}',
                'industry': 'Industrial Automation, Engineering',
                'description': 'FANUC and KUKA certified robotics integrator.',
                'size': 'Small Business',
                'capabilities': ['Robotics Integration', 'Industrial Automation'],
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
                'description': 'ITAR registered defense contractor specializing in naval systems.',
                'size': 'Medium Business',
                'capabilities': ['Naval Systems', 'Defense Systems'],
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
                'description': 'USCG approved maritime training facility.',
                'size': 'Medium Business',
                'capabilities': ['Maritime Training', 'Safety Training'],
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
                'description': 'R&D company developing UUV and ROV systems.',
                'size': 'Small Business',
                'capabilities': ['UUV Systems', 'ROV Systems'],
                'lat': base_lat + 0.18,
                'lon': base_lon + 0.15,
                'website': 'www.regionalautonomous.com',
                'phone': '(555) 555-0105',
                'rating': 4.4,
                'user_ratings_total': 19
            }
        ]
        
        # Add scoring
        for company in sample_companies:
            distance = self._calculate_distance(company['lat'], company['lon'])
            company['distance_miles'] = distance
            
            scores = self._score_company_relevance(company)
            company.update(scores)
        
        sample_companies.sort(key=lambda x: x['total_score'], reverse=True)
        return sample_companies

class EnhancedVisualization:
    """Enhanced visualization class"""
    
    @staticmethod
    def create_advanced_company_map(companies: List[Dict], base_coords: Tuple[float, float]) -> Optional[go.Figure]:
        """Create interactive map"""
        if not companies:
            return None
        
        df = pd.DataFrame(companies)
        fig = go.Figure()
        
        # Add base location
        fig.add_trace(go.Scattermapbox(
            lat=[base_coords[0]],
            lon=[base_coords[1]],
            mode='markers',
            marker=dict(size=25, color='red', symbol='star'),
            text=['🎯 Search Center'],
            name='Search Center'
        ))
        
        # Add companies
        colors = df['total_score']
        sizes = np.where(df['total_score'] > 10, 15, 
                np.where(df['total_score'] > 5, 12, 8))
        
        hover_text = []
        for _, row in df.iterrows():
            hover_text.append(
                f"<b>{row['name']}</b><br>"
                f"Score: {row['total_score']:.1f}<br>"
                f"Distance: {row['distance_miles']:.1f} mi<br>"
                f"Size: {row['size']}"
            )
        
        fig.add_trace(go.Scattermapbox(
            lat=df['lat'],
            lon=df['lon'],
            mode='markers',
            marker=dict(
                size=sizes,
                color=colors,
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Naval Relevance Score")
            ),
            text=hover_text,
            name='Companies',
            hovertemplate='%{text}<extra></extra>'
        ))
        
        fig.update_layout(
            mapbox=dict(
                style='open-street-map',
                center=dict(lat=base_coords[0], lon=base_coords[1]),
                zoom=8
            ),
            height=700,
            title="🗺️ Naval Supplier Geographic Intelligence"
        )
        
        return fig

def create_enhanced_metrics_dashboard(companies: List[Dict]) -> str:
    """Create metrics dashboard HTML"""
    if not companies:
        return ""
    
    df = pd.DataFrame(companies)
    
    total_companies = len(companies)
    high_relevance = len(df[df['total_score'] >= 10])
    small_businesses = len(df[df['size'].str.contains('Small', na=False)])
    avg_distance = df['distance_miles'].mean()
    avg_rating = df['rating'].mean()
    
    # Get top capability
    all_caps = []
    for caps in df['capabilities']:
        all_caps.extend(caps)
    top_capability = pd.Series(all_caps).value_counts().index[0] if all_caps else "N/A"
    
    quality_suppliers = len(df[(df['rating'] >= 4.0) & (df['user_ratings_total'] >= 10)])
    defense_contractors = len(df[df['industry'].str.contains('Defense|Naval|Aerospace', case=False, na=False)])
    
    return f"""
    <div class="metric-grid">
        <div class="metric-card">
            <p class="metric-value">{total_companies}</p>
            <p class="metric-label">Total Suppliers</p>
        </div>
        <div class="metric-card">
            <p class="metric-value">{high_relevance}</p>
            <p class="metric-label">High Relevance</p>
        </div>
        <div class="metric-card">
            <p class="metric-value">{small_businesses}</p>
            <p class="metric-label">Small Businesses</p>
        </div>
        <div class="metric-card">
            <p class="metric-value">{avg_distance:.1f} mi</p>
            <p class="metric-label">Avg Distance</p>
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
            <p class="metric-label">Avg Rating</p>
        </div>
        <div class="metric-card">
            <p class="metric-value">{top_capability[:15]}</p>
            <p class="metric-label">Top Capability</p>
        </div>
    </div>
    """

def generate_executive_report(companies: List[Dict], config: SearchConfig) -> str:
    """Generate executive report"""
    if not companies:
        return "No companies found for analysis."
    
    df = pd.DataFrame(companies)
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    total_companies = len(companies)
    high_value_companies = len(df[df['total_score'] >= 10])
    defense_companies = len(df[df['industry'].str.contains('Defense|Naval|Aerospace', case=False, na=False)])
    small_businesses = len(df[df['size'].str.contains('Small', na=False)])
    
    top_companies = df.nlargest(5, 'total_score')
    
    report = f"""
# 🎯 WBI Naval Search - Executive Intelligence Report

**Generated:** {current_time}  
**Search Location:** {config.base_location}  
**Search Radius:** {config.radius_miles} miles  

## 📊 Executive Summary

The naval supplier intelligence search identified **{total_companies} companies** within the specified {config.radius_miles}-mile radius of {config.base_location}.

### Key Findings:
- **{high_value_companies}** companies demonstrate high naval relevance (score ≥ 10)
- **{defense_companies}** companies have direct defense/aerospace focus
- **{small_businesses}** small businesses identified for potential partnerships
- **{df['distance_miles'].mean():.1f} miles** average supplier distance

## 🏆 Top-Tier Naval Suppliers

"""
    
    for i, (_, company) in enumerate(top_companies.iterrows(), 1):
        report += f"""
### {i}. {company['name']}
- **Naval Relevance Score:** {company['total_score']:.1f}
- **Location:** {company['location']} ({company['distance_miles']:.1f} miles)
- **Size:** {company['size']}
- **Key Capabilities:** {', '.join(company['capabilities'][:3])}
- **Rating:** {company['rating']:.1f}⭐ ({company['user_ratings_total']} reviews)

"""
    
    return report

def main():
    """Main application function"""
    st.set_page_config(
        page_title="WBI Naval Search - Advanced Supplier Intelligence Platform",
        page_icon="⚓",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Initialize session state
    if 'search_triggered' not in st.session_state:
        st.session_state.search_triggered = False
    if 'companies' not in st.session_state:
        st.session_state.companies = []
    
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
    
    .wbi-header {
        background: linear-gradient(135deg, #1a202c 0%, #2d3748 100%);
        padding: 2rem 1rem;
        box-shadow: 0 10px 40px rgba(0, 0, 0, 0.15);
        margin-bottom: 0;
    }
    
    .wbi-logo-container {
        background: linear-gradient(135deg, #ffffff 0%, #f7fafc 100%);
        border-radius: 1rem;
        padding: 1rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.2);
        display: flex;
        justify-content: center;
        align-items: center;
        width: fit-content;
        margin-left: auto;
        margin-right: auto;
    }
    
    .wbi-logo {
        font-size: 3.5rem;
        color: #1a202c;
        font-weight: 800;
        display: flex;
        align-items: center;
        gap: 1rem;
    }
    
    .wbi-header h1 {
        color: #ffffff !important;
        font-size: 2.25rem !important;
        font-weight: 700 !important;
        text-align: center;
        margin: 0 !important;
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
    
    .wbi-card p {
        color: #cbd5e0 !important;
        line-height: 1.6;
    }
    
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
    
    .stSidebar > div:first-child {
        background-color: #2d3748;
        border-right: 2px solid #4a5568;
    }
    
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
    
    .main .block-container h1,
    .main .block-container h2,
    .main .block-container h3 {
        color: #ffffff !important;
        font-weight: 600 !important;
    }
    
    .main .block-container p {
        color: #e2e8f0 !important;
    }
    
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
    </style>
    """, unsafe_allow_html=True)
    
    # Check for geopy availability
    if not GEOPY_AVAILABLE:
        st.error("⚠️ Missing required package: geopy")
        st.info("Please install geopy: `pip install geopy`")
        st.stop()
    
    # WBI header
    st.markdown("""
    <div class="wbi-header">
        <div class="wbi-logo-container">
            <div class="wbi-logo">
                ⚓ WBI
            </div>
        </div>
        <h1>Naval Search Pro</h1>
        <p>Advanced supplier intelligence and procurement analytics platform for naval operations. Discover, analyze, and connect with defense contractors and maritime suppliers.</p>
    </div>
    <div class="wbi-border"></div>
    """, unsafe_allow_html=True)
    
    # Sidebar configuration
    st.sidebar.header("🔧 Advanced Search Configuration")
    
    # API Key management
    if not GOOGLE_PLACES_API_KEY:
        st.sidebar.warning("⚠️ No API key detected")
        with st.sidebar.expander("🔑 API Key Setup", expanded=True):
            st.markdown("Enter your Google Places API key:")
            api_key_input = st.text_input(
                "Google Places API Key:", 
                type="password",
                help="Enter your API key for real company search"
            )
            if api_key_input:
                st.session_state.api_key = api_key_input
                st.success("✅ API key configured!")
    else:
        st.sidebar.success("✅ API key configured")
    
    # Configuration
    config = SearchConfig()
    
    # Location selection
    st.sidebar.subheader("📍 Search Location")
    
    location_option = st.sidebar.radio(
        "Choose search center:",
        ["🎯 Preset Naval Locations", "🗺️ Custom Location"]
    )
    
    if location_option == "🎯 Preset Naval Locations":
        preset_locations = {
            "South Bend, Indiana (WBI Headquarters)": "South Bend, Indiana",
            "Norfolk, Virginia (Naval Station Norfolk)": "Norfolk, Virginia",
            "San Diego, California (Naval Base San Diego)": "San Diego, California", 
            "Pearl Harbor, Hawaii": "Pearl Harbor, Hawaii",
            "Bremerton, Washington (Puget Sound Naval Shipyard)": "Bremerton, Washington",
            "Portsmouth, New Hampshire": "Portsmouth, New Hampshire",
            "Newport News, Virginia": "Newport News, Virginia",
            "Bath, Maine (Bath Iron Works)": "Bath, Maine",
            "Groton, Connecticut (Electric Boat)": "Groton, Connecticut",
            "Pascagoula, Mississippi": "Pascagoula, Mississippi"
        }
        
        selected_preset = st.sidebar.selectbox(
            "Select naval facility:",
            list(preset_locations.keys())
        )
        config.base_location = preset_locations[selected_preset]
        
    else:
        config.base_location = st.sidebar.text_input(
            "Enter custom location:",
            value="South Bend, Indiana",
            help="Enter city, state (e.g., 'Detroit, Michigan')"
        )
    
    # Search parameters
    st.sidebar.subheader("🎯 Search Parameters")
    
    col1, col2 = st.sidebar.columns(2)
    with col1:
        config.radius_miles = st.slider("Search Radius (miles)", 10, 200, 60)
    with col2:
        config.target_company_count = st.slider("Target Companies", 10, 500, 100)
    
    # Search execution
    if st.sidebar.button("🔍 Execute Naval Search", type="primary"):
        st.session_state.search_triggered = True
    
    # Main content area
    col1, col2 = st.columns([3, 1])
    
    with col2:
        st.markdown("""
        <div class="wbi-card">
            <h3>⚡ Naval Procurement Intelligence</h3>
            <p>Advanced supplier discovery and market intelligence for naval operations.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col1:
        # Execute search when triggered
        if st.session_state.get('search_triggered', False):
            search_start_time = time.time()
            
            with st.spinner("🔍 Executing naval supplier search..."):
                try:
                    searcher = EnhancedCompanySearcher(config)
                    companies = searcher.search_companies()
                    
                    search_duration = time.time() - search_start_time
                    
                    st.session_state.companies = companies
                    st.session_state.searcher = searcher
                    st.session_state.search_duration = search_duration
                    
                    if companies:
                        st.success(f"✅ Found {len(companies)} naval suppliers in {search_duration:.1f}s")
                    else:
                        st.warning("⚠️ No companies found matching criteria")
                        
                except Exception as e:
                    st.error(f"❌ Search error: {str(e)}")
                    logger.error(f"Search execution error: {e}")
    
    # Display results
    if st.session_state.get('companies'):
        companies = st.session_state.companies
        searcher = st.session_state.searcher
        
        # Metrics dashboard
        st.markdown("## 📊 Naval Supplier Intelligence Dashboard")
        metrics_html = create_enhanced_metrics_dashboard(companies)
        st.markdown(metrics_html, unsafe_allow_html=True)
        
        # Tabs
        tab1, tab2, tab3 = st.tabs([
            "📋 Supplier Directory", 
            "🗺️ Geographic Intelligence", 
            "📄 Intelligence Export"
        ])
        
        with tab1:
            st.subheader("🏭 Naval Supplier Directory")
            
            # Filters
            filter_col1, filter_col2, filter_col3 = st.columns(3)
            
            with filter_col1:
                min_score = st.slider("Min Naval Relevance", 0, 50, 0)
            with filter_col2:
                size_filter = st.selectbox("Company Size", ["All", "Small Business", "Medium Business", "Large Corporation"])
            with filter_col3:
                max_distance = st.slider("Max Distance (mi)", 0, config.radius_miles, config.radius_miles)
            
            # Filter companies
            filtered_companies = companies.copy()
            if min_score > 0:
                filtered_companies = [c for c in filtered_companies if c['total_score'] >= min_score]
            if size_filter != "All":
                filtered_companies = [c for c in filtered_companies if c['size'] == size_filter]
            if max_distance < config.radius_miles:
                filtered_companies = [c for c in filtered_companies if c['distance_miles'] <= max_distance]
            
            st.info(f"📊 Showing {len(filtered_companies)} suppliers (filtered from {len(companies)} total)")
            
            # Company display
            for company in filtered_companies:
                score_color = "🟢" if company['total_score'] >= 15 else "🟡" if company['total_score'] >= 8 else "🔴"
                
                with st.expander(f"{score_color} {company['name']} - Naval Score: {company['total_score']:.1f}"):
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
                        st.markdown("**🔧 Capabilities**")
                        for capability in company['capabilities']:
                            st.write(f"• {capability}")
                    
                    st.markdown(f"**📋 Description:** {company['description']}")
                    
                    # Scoring breakdown
                    score_col1, score_col2, score_col3, score_col4 = st.columns(4)
                    with score_col1:
                        st.metric("🏭 Manufacturing", f"{company['manufacturing_score']:.1f}")
                    with score_col2:
                        st.metric("🤖 Robotics", f"{company['robotics_score']:.1f}")
                    with score_col3:
                        st.metric("🚁 Unmanned", f"{company['unmanned_score']:.1f}")
                    with score_col4:
                        st.metric("👥 Workforce", f"{company['workforce_score']:.1f}")
        
        with tab2:
            st.subheader("🗺️ Geographic Intelligence")
            
            map_fig = EnhancedVisualization.create_advanced_company_map(companies, searcher.base_coords)
            if map_fig:
                st.plotly_chart(map_fig, use_container_width=True)
            else:
                st.info("No geographic data available for mapping")
        
        with tab3:
            st.subheader("📄 Intelligence Export")
            
            if companies:
                df = pd.DataFrame(companies)
                
                # CSV Export
                csv_data = df.to_csv(index=False)
                st.download_button(
                    label="📥 Download CSV Report",
                    data=csv_data,
                    file_name=f"wbi_naval_intelligence_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                    mime="text/csv"
                )
                
                # Executive summary export
                exec_report = generate_executive_report(companies, config)
                st.download_button(
                    label="📋 Download Executive Report",
                    data=exec_report,
                    file_name=f"wbi_executive_report_{datetime.now().strftime('%Y%m%d_%H%M')}.md",
                    mime="text/markdown"
                )
                
                # Report preview
                st.markdown("### 📋 Executive Report Preview")
                with st.expander("Preview Executive Intelligence Report", expanded=False):
                    exec_report_preview = generate_executive_report(companies, config)
                    st.markdown(exec_report_preview)

    # Footer
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
        
        st.markdown("## 🔧 Fallback Mode")
        st.info("The application encountered an error. Please refresh the page.")
        
        if st.button("🔄 Restart Application"):
            st.rerun()