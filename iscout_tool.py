import streamlit as st
import pandas as pd
import requests
import time
import os
import json
from typing import List, Dict, Tuple
from dataclasses import dataclass, field
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import numpy as np

# API Configuration
GOOGLE_PLACES_API_KEY = os.environ.get("GOOGLE_PLACES_API_KEY", "")

@dataclass
class SearchConfig:
    base_location: str = "South Bend, Indiana"
    radius_miles: int = 60
    target_company_count: int = 100
    search_multi_state: bool = False
    enable_association_search: bool = True
    
    # Enhanced keyword categories (restored from original)
    manufacturing_keywords: List[str] = field(default_factory=lambda: [
        "advanced manufacturing", "precision machining", "metal fabrication",
        "additive manufacturing", "3D printing", "CNC machining",
        "welding", "assembly", "fabrication", "machining", "casting",
        "forging", "sheet metal", "tool and die", "injection molding",
        "shipbuilding", "naval shipyard", "marine engineering", "hull fabrication"
    ])
    
    # NEW: Microelectronics keywords (HIGH PRIORITY for Navy microelectronics center)
    microelectronics_keywords: List[str] = field(default_factory=lambda: [
        "microelectronics", "semiconductor", "electronics manufacturing",
        "PCB manufacturing", "printed circuit board", "IC design",
        "integrated circuits", "electronic components", "naval electronics",
        "maritime electronics", "avionics", "radar systems", "sonar electronics",
        "navigation electronics", "communication systems", "electronic warfare",
        "signal processing", "embedded systems", "microprocessors",
        "FPGA", "analog circuits", "digital circuits", "RF electronics"
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
        "unmanned ground vehicles", "swarm robotics",
        "underwater drones", "maritime autonomous", "naval drones"
    ])
    
    workforce_keywords: List[str] = field(default_factory=lambda: [
        "naval training", "maritime training", "shipyard training", "welding certification",
        "maritime academy", "technical training", "apprenticeship", "workforce development",
        "skills training", "industrial training", "safety training", "crane operator training",
        "maritime safety", "naval education", "defense training", "military training",
        "shipbuilding training", "marine engineering training", "technical certification",
        "OSHA training", "maritime simulation", "deck officer training",
        "electronics training", "microelectronics training", "semiconductor training"
    ])
    
    defense_keywords: List[str] = field(default_factory=lambda: [
        "defense contractor", "military contractor", "naval contractor",
        "aerospace defense", "prime contractor", "subcontractor",
        "DFARS", "ITAR", "security clearance", "classified work",
        "government contracting", "GSA schedule"
    ])

class EnhancedNavalSearcher:
    def __init__(self, config: SearchConfig):
        self.config = config
        
    def search_companies(self) -> List[Dict]:
        """Enhanced company search with microelectronics focus"""
        api_key = st.session_state.get('api_key', GOOGLE_PLACES_API_KEY)
        
        if not api_key:
            st.error("🔑 Google Places API key is REQUIRED for company search")
            st.info("Add your API key in the sidebar to search for companies")
            return []
        
        st.info("🔍 Searching for naval/manufacturing companies with enhanced queries...")
        
        # Enhanced search queries with microelectronics priority
        search_queries = self._get_enhanced_search_queries()
        
        all_companies = []
        progress_bar = st.progress(0)
        
        for i, query in enumerate(search_queries):
            st.write(f"Searching: {query}")
            companies = self._search_google_places(query, api_key)
            all_companies.extend(companies)
            progress_bar.progress((i + 1) / len(search_queries))
            time.sleep(1)  # Rate limiting
        
        progress_bar.empty()
        
        # Process and deduplicate with enhanced scoring
        unique_companies = self._process_companies_enhanced(all_companies)
        
        st.success(f"✅ Found {len(unique_companies)} companies with enhanced scoring")
        return unique_companies
    
    def _get_enhanced_search_queries(self) -> List[str]:
        """Generate enhanced search queries with microelectronics focus"""
        base_location = self.config.base_location
        
        return [
            # MICROELECTRONICS (TOP PRIORITY for Navy microelectronics center)
            f"microelectronics manufacturing near {base_location}",
            f"semiconductor companies near {base_location}",
            f"electronics manufacturing near {base_location}",
            f"PCB manufacturing near {base_location}",
            f"naval electronics near {base_location}",
            f"radar systems near {base_location}",
            f"sonar electronics near {base_location}",
            f"electronic components near {base_location}",
            
            # SHIPBUILDING AND NAVAL MANUFACTURING  
            f"shipbuilding company near {base_location}",
            f"marine engineering near {base_location}",
            f"naval architecture near {base_location}",
            f"hull fabrication near {base_location}",
            f"shipyard services near {base_location}",
            f"submarine systems near {base_location}",
            
            # DEFENSE MANUFACTURING
            f"defense contractor manufacturing near {base_location}",
            f"aerospace manufacturing near {base_location}",
            f"military equipment manufacturer near {base_location}",
            f"naval systems manufacturer near {base_location}",
        
            # ADVANCED MANUFACTURING
            f"CNC machining services near {base_location}",
            f"precision machining near {base_location}",
            f"metal fabrication shop near {base_location}",
            f"custom manufacturing near {base_location}",
            f"contract manufacturing near {base_location}",
            f"welding fabrication near {base_location}",
            f"additive manufacturing near {base_location}",
        
            # ROBOTICS AND AUTOMATION
            f"robotics manufacturer near {base_location}",
            f"automation systems near {base_location}",
            f"control systems manufacturer near {base_location}",
        
            # TRAINING FACILITIES
            f"maritime training facility near {base_location}",
            f"welding school near {base_location}",
            f"technical training institute near {base_location}",
            f"electronics training near {base_location}"
        ]
    
    def _search_google_places(self, query: str, api_key: str) -> List[Dict]:
        """Enhanced Google Places search with better filtering"""
        url = "https://places.googleapis.com/v1/places:searchText"
        
        headers = {
            'Content-Type': 'application/json',
            'X-Goog-Api-Key': api_key,
            'X-Goog-FieldMask': 'places.displayName,places.formattedAddress,places.location,places.types,places.websiteUri,places.nationalPhoneNumber,places.rating,places.userRatingCount,places.businessStatus'
        }
        
        request_data = {
            "textQuery": query,
            "maxResultCount": 20
        }
        
        try:
            response = requests.post(url, headers=headers, json=request_data, timeout=15)
            
            if response.status_code == 200:
                data = response.json()
                return self._process_places_response(data.get('places', []), query)
            elif response.status_code == 403:
                st.error("❌ API key doesn't have permission. Enable 'Places API (New)' and billing.")
                return []
            else:
                st.warning(f"API returned status {response.status_code} for query: {query}")
                return []
                
        except Exception as e:
            st.warning(f"Search error for '{query}': {str(e)}")
            return []
    
    def _process_places_response(self, places: List[Dict], query: str) -> List[Dict]:
        """Process Google Places response with enhanced filtering"""
        companies = []
        
        for place in places:
            try:
                name = place.get('displayName', {}).get('text', 'Unknown')
                types = place.get('types', [])
                
                # Enhanced relevance filtering
                if self._is_naval_manufacturing_relevant(name, types):
                    lat = place.get('location', {}).get('latitude', 0)
                    lon = place.get('location', {}).get('longitude', 0)
                    
                    company = {
                        'name': name,
                        'location': place.get('formattedAddress', 'Unknown'),
                        'industry': ', '.join(types[:3]) if types else 'Manufacturing',
                        'description': self._generate_enhanced_description(name, types, query),
                        'size': self._determine_business_size_enhanced(name, types, place),
                        'capabilities': self._extract_capabilities_enhanced(name, types),
                        'lat': lat,
                        'lon': lon,
                        'website': place.get('websiteUri', 'Not available'),
                        'phone': place.get('nationalPhoneNumber', 'Not available'),
                        'rating': place.get('rating', 0),
                        'user_ratings_total': place.get('userRatingCount', 0),
                        'types': types
                    }
                    companies.append(company)
                    
            except Exception as e:
                continue  # Skip problematic entries
        
        return companies
    
    def _is_naval_manufacturing_relevant(self, name: str, types: List[str]) -> bool:
        """Enhanced filtering for naval/manufacturing relevance"""
        name_lower = name.lower()
        types_str = ' '.join(types).lower()
        combined = f"{name_lower} {types_str}"
        
        # STRICT exclusions - these should NEVER appear
        hard_exclude = [
            'restaurant', 'food', 'catering', 'retail', 'store', 'bank', 'insurance',
            'real estate', 'school', 'hospital', 'hotel', 'gas station', 'pharmacy',
            'auto repair', 'automotive repair', 'hair salon', 'nail salon', 'spa',
            'cleaning services', 'janitorial', 'landscaping', 'roofing', 'flooring',
            'residential construction', 'home improvement', 'interior design'
        ]
        
        for exclude in hard_exclude:
            if exclude in combined:
                return False
        
        # MUST have manufacturing/naval/electronics indicators
        required_indicators = [
            # Manufacturing
            'manufacturing', 'machining', 'fabrication', 'welding', 'casting',
            'forging', 'cnc', 'precision', 'metal', 'steel', 'aluminum',
            'assembly', 'production',
            
            # Naval/maritime
            'shipyard', 'shipbuilding', 'marine', 'naval', 'maritime',
            'submarine', 'vessel', 'boat', 'ship', 'hull',
            
            # Electronics/microelectronics
            'electronics', 'microelectronics', 'semiconductor', 'pcb',
            'electronic components', 'circuits', 'radar', 'sonar',
            
            # Defense/aerospace
            'aerospace', 'defense contractor', 'military contractor',
            'avionics', 'defense systems',
            
            # Technology/automation
            'robotics', 'automation', 'systems integration', 'controls',
            
            # Training (naval specific)
            'maritime academy', 'naval training', 'shipyard training',
            'welding certification', 'maritime training'
        ]
        
        # Must have at least one manufacturing/naval indicator
        has_relevant_indicator = any(indicator in combined for indicator in required_indicators)
        
        # Business types that indicate manufacturing
        manufacturing_types = [
            'manufacturer', 'machine_shop', 'factory', 'foundry', 'mill',
            'engineering', 'contractor', 'technology'
        ]
        has_mfg_type = any(mtype in types_str for mtype in manufacturing_types)
        
        return has_relevant_indicator or has_mfg_type
    
    def _generate_enhanced_description(self, name: str, types: List[str], query: str) -> str:
        """Generate enhanced description based on search context"""
        base_desc = f"Business type: {', '.join(types[:2]) if types else 'Manufacturing'}"
        
        name_lower = name.lower()
        query_lower = query.lower()
        
        # Add context based on search query and company name
        if 'microelectronics' in query_lower or 'semiconductor' in query_lower:
            base_desc += " - Microelectronics/Semiconductor focus"
        elif 'electronics' in query_lower:
            base_desc += " - Electronics manufacturing"
        elif 'naval' in query_lower or 'marine' in query_lower:
            base_desc += " - Naval/Maritime systems"
        elif 'aerospace' in query_lower or 'defense' in query_lower:
            base_desc += " - Aerospace/Defense contractor"
        elif 'shipbuilding' in query_lower:
            base_desc += " - Shipbuilding industry"
        elif 'precision' in name_lower:
            base_desc += " - Precision manufacturing"
        elif 'automation' in query_lower or 'robotics' in query_lower:
            base_desc += " - Automation/Robotics systems"
        
        return base_desc
    
    def _determine_business_size_enhanced(self, name: str, types: List[str], place_data: Dict) -> str:
        """Enhanced business size determination"""
        name_lower = name.lower()
        review_count = place_data.get('userRatingCount', 0)
        rating = place_data.get('rating', 0)
        
        # Major defense/aerospace corporations
        major_corps = [
            'honeywell', 'boeing', 'lockheed martin', 'raytheon', 'northrop grumman',
            'general dynamics', 'bae systems', 'textron', 'collins aerospace',
            'general electric', 'pratt whitney', 'rolls royce', 'caterpillar',
            'huntington ingalls', 'newport news shipbuilding', 'bath iron works',
            'electric boat', 'l3harris', 'intel', 'texas instruments'
        ]
        
        for corp in major_corps:
            if corp in name_lower:
                return 'Fortune 500 / Major Corporation'
        
        # Enhanced size determination using multiple factors
        if review_count > 200 or (review_count > 100 and rating > 4.0):
            return 'Large Corporation'
        elif review_count > 50 or (review_count > 25 and rating > 4.2):
            return 'Medium Business'
        elif review_count > 10:
            return 'Small-Medium Business'
        else:
            return 'Small Business'
    
    def _extract_capabilities_enhanced(self, name: str, types: List[str]) -> List[str]:
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
            
            # Microelectronics and electronics
            'microelectronics': 'Microelectronics Manufacturing',
            'semiconductor': 'Semiconductor Design/Fab',
            'pcb': 'PCB Manufacturing',
            'electronics': 'Electronics Manufacturing',
            'electronic components': 'Electronic Components',
            'circuits': 'Circuit Design/Manufacturing',
            'radar': 'Radar Systems',
            'sonar': 'Sonar Systems',
            'avionics': 'Avionics Systems',
            
            # Automation and robotics
            'automation': 'Industrial Automation',
            'robotics': 'Robotics Integration',
            'robotic': 'Robotic Systems',
            'vision': 'Machine Vision Systems',
            'control': 'Process Control Systems',
            
            # Naval and maritime
            'shipyard': 'Shipbuilding',
            'shipbuilding': 'Shipbuilding',
            'marine': 'Marine Engineering',
            'naval': 'Naval Systems',
            'maritime': 'Maritime Systems',
            'submarine': 'Submarine Systems',
            'hull': 'Hull Fabrication',
            
            # Aerospace and defense
            'aerospace': 'Aerospace Manufacturing',
            'defense': 'Defense Systems',
            'military': 'Military Systems',
            
            # Training and education
            'training': 'Training Services',
            'academy': 'Educational Academy',
            'certification': 'Certification Programs',
            'apprenticeship': 'Apprenticeship Programs'
        }
        
        combined_text = f"{name_lower} {types_str}"
        for keyword, capability in capability_mapping.items():
            if keyword in combined_text:
                capabilities.add(capability)
        
        return list(capabilities) if capabilities else ['General Manufacturing']
    
    def _process_companies_enhanced(self, all_companies: List[Dict]) -> List[Dict]:
        """Enhanced processing with advanced scoring"""
        unique_companies = []
        seen = set()
        
        for company in all_companies:
            # Enhanced duplicate detection
            key = (company['name'].lower().strip(), company['location'][:50].lower().strip())
            
            if key not in seen:
                seen.add(key)
                
                # Add distance calculation
                company['distance_miles'] = self._estimate_distance(company['lat'], company['lon'])
                
                # Enhanced scoring with multiple categories
                scores = self._calculate_enhanced_relevance_scores(company)
                company.update(scores)
                
                # Filter by distance and minimum relevance
                if (company['distance_miles'] <= self.config.radius_miles and 
                    company['total_score'] >= 5):  # Minimum threshold
                    unique_companies.append(company)
        
        # Sort by total score
        unique_companies.sort(key=lambda x: x['total_score'], reverse=True)
        
        return unique_companies[:self.config.target_company_count]
    
    def _estimate_distance(self, lat: float, lon: float) -> float:
        """Enhanced distance estimation with better base coordinates"""
        # Enhanced base coordinates for major naval locations
        base_coords = {
            'south bend': (41.6764, -86.2520),
            'norfolk': (36.8508, -76.2859),
            'san diego': (32.7157, -117.1611),
            'pearl harbor': (21.3099, -157.8581),
            'newport news': (37.0871, -76.4730),
            'bath': (43.9109, -69.8214),
            'groton': (41.3501, -72.0979)
        }
        
        # Find base coordinates for location
        base_lat, base_lon = 41.6764, -86.2520  # Default to South Bend
        for location_key, coords in base_coords.items():
            if location_key in self.config.base_location.lower():
                base_lat, base_lon = coords
                break
        
        if lat == 0 or lon == 0:
            return 999.0
        
        # Simple distance approximation
        lat_diff = abs(lat - base_lat)
        lon_diff = abs(lon - base_lon)
        distance = ((lat_diff ** 2 + lon_diff ** 2) ** 0.5) * 69  # Rough miles conversion
        
        return round(distance, 1)
    
    def _calculate_enhanced_relevance_scores(self, company: Dict) -> Dict:
        """Enhanced scoring system with microelectronics emphasis"""
        name = company['name'].lower()
        industry = company['industry'].lower()
        description = company['description'].lower()
        capabilities = ' '.join(company['capabilities']).lower()
        combined = f"{name} {industry} {description} {capabilities}"
        
        scores = {
            'manufacturing_score': 0.0,
            'microelectronics_score': 0.0,  # NEW: High priority
            'robotics_score': 0.0,
            'unmanned_score': 0.0,
            'workforce_score': 0.0,
            'defense_score': 0.0,
            'naval_score': 0.0,  # NEW: Naval specific
            'total_score': 0.0
        }
        
        # MICROELECTRONICS SCORING (HIGHEST PRIORITY)
        microelectronics_weights = {
            "microelectronics": 25, "semiconductor": 20, "pcb manufacturing": 18,
            "electronics manufacturing": 15, "integrated circuits": 18, "ic design": 20,
            "naval electronics": 22, "maritime electronics": 20, "avionics": 15,
            "radar systems": 18, "sonar electronics": 18, "navigation electronics": 16,
            "communication systems": 14, "electronic warfare": 20, "signal processing": 16,
            "embedded systems": 14, "microprocessors": 16, "fpga": 15,
            "electronic components": 12, "circuits": 10
        }
        
        for keyword, weight in microelectronics_weights.items():
            if keyword in combined:
                scores['microelectronics_score'] += weight
        
        # NAVAL/MARITIME SCORING
        naval_weights = {
            "shipbuilding": 25, "naval shipyard": 28, "hull fabrication": 20,
            "marine engineering": 18, "submarine": 25, "naval systems": 22,
            "maritime": 15, "shipyard": 20, "vessel": 12, "marine": 12,
            "navy": 20, "naval": 18, "coast guard": 16
        }
        
        for keyword, weight in naval_weights.items():
            if keyword in combined:
                scores['naval_score'] += weight
        
        # MANUFACTURING SCORING
        manufacturing_weights = {
            'precision machining': 12, 'cnc machining': 10, 'metal fabrication': 8,
            'welding': 6, 'manufacturing': 8, 'aerospace manufacturing': 12,
            'defense manufacturing': 10, 'additive manufacturing': 8
        }
        
        for keyword, weight in manufacturing_weights.items():
            if keyword in combined:
                scores['manufacturing_score'] += weight
        
        # DEFENSE SCORING
        defense_weights = {
            'defense contractor': 15, 'military contractor': 12, 'aerospace': 10,
            'defense systems': 12, 'military systems': 10, 'government contracting': 8
        }
        
        for keyword, weight in defense_weights.items():
            if keyword in combined:
                scores['defense_score'] += weight
        
        # ROBOTICS SCORING
        robotics_weights = {
            'robotics': 10, 'automation': 8, 'robotic systems': 10, 'industrial automation': 8
        }
        
        for keyword, weight in robotics_weights.items():
            if keyword in combined:
                scores['robotics_score'] += weight
        
        # QUALITY BONUSES
        rating = company.get('rating', 0)
        review_count = company.get('user_ratings_total', 0)
        
        quality_bonus = 0
        if rating >= 4.5 and review_count >= 10:
            quality_bonus = 5
        elif rating >= 4.0 and review_count >= 5:
            quality_bonus = 3
        
        # Calculate total score with MICROELECTRONICS EMPHASIS
        scores['total_score'] = (
            scores['manufacturing_score'] * 1.0 +
            scores['microelectronics_score'] * 2.0 +  # HIGHEST WEIGHT
            scores['naval_score'] * 1.5 +
            scores['defense_score'] * 1.2 +
            scores['robotics_score'] * 0.8 +
            scores['unmanned_score'] * 0.9 +
            scores['workforce_score'] * 0.7 +
            quality_bonus
        )
        
        return scores

def create_enhanced_company_map(companies: List[Dict], base_location: str):
    """Create enhanced map with better scoring visualization"""
    if not companies:
        return None
    
    df = pd.DataFrame(companies)
    
    # Filter valid coordinates
    df = df[(df['lat'] != 0) & (df['lon'] != 0)]
    
    if df.empty:
        return None
    
    # Enhanced base coordinates
    base_coords = {
        'south bend': (41.6764, -86.2520),
        'norfolk': (36.8508, -76.2859),
        'san diego': (32.7157, -117.1611),
        'pearl harbor': (21.3099, -157.8581),
        'newport news': (37.0871, -76.4730),
        'bath': (43.9109, -69.8214),
        'groton': (41.3501, -72.0979)
    }
    
    base_lat, base_lon = 41.6764, -86.2520  # Default
    for key, coords in base_coords.items():
        if key in base_location.lower():
            base_lat, base_lon = coords
            break
    
    fig = go.Figure()
    
    # Add base location
    fig.add_trace(go.Scattermapbox(
        lat=[base_lat],
        lon=[base_lon],
        mode='markers',
        marker=dict(size=20, color='blue'),
        text=[f'Search Center: {base_location}'],
        name='Base Location'
    ))
    
    # Add companies with enhanced scoring
    fig.add_trace(go.Scattermapbox(
        lat=df['lat'],
        lon=df['lon'],
        mode='markers',
        marker=dict(
            size=12,
            color=df['total_score'],
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title="Enhanced Score")
        ),
        text=[f"{row['name']}<br>Total Score: {row['total_score']:.1f}<br>Microelectronics: {row['microelectronics_score']:.1f}<br>Naval: {row['naval_score']:.1f}<br>Distance: {row['distance_miles']} mi" 
              for _, row in df.iterrows()],
        name='Companies'
    ))
    
    fig.update_layout(
        mapbox=dict(
            style='open-street-map',
            center=dict(lat=base_lat, lon=base_lon),
            zoom=8
        ),
        height=600,
        title="Enhanced Naval Supplier Intelligence Map"
    )
    
    return fig

def create_enhanced_metrics_dashboard(companies: List[Dict]) -> str:
    """Create enhanced metrics dashboard with new scoring categories"""
    if not companies:
        return ""
    
    df = pd.DataFrame(companies)
    
    total_companies = len(companies)
    high_relevance = len(df[df['total_score'] >= 15])
    microelectronics_companies = len(df[df['microelectronics_score'] >= 10])
    naval_focused = len(df[df['naval_score'] >= 10])
    defense_contractors = len(df[df['defense_score'] >= 8])
    small_businesses = len(df[df['size'].str.contains('Small', na=False)])
    avg_distance = df['distance_miles'].mean()
    quality_suppliers = len(df[(df['rating'] >= 4.0) & (df['user_ratings_total'] >= 10)])
    
    return f"""
    <div class="metric-grid">
        <div class="metric-card">
            <p class="metric-value">{total_companies}</p>
            <p class="metric-label">Total Suppliers</p>
        </div>
        <div class="metric-card">
            <p class="metric-value">{microelectronics_companies}</p>
            <p class="metric-label">Microelectronics Focus</p>
        </div>
        <div class="metric-card">
            <p class="metric-value">{naval_focused}</p>
            <p class="metric-label">Naval/Maritime</p>
        </div>
        <div class="metric-card">
            <p class="metric-value">{defense_contractors}</p>
            <p class="metric-label">Defense Contractors</p>
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
            <p class="metric-value">{quality_suppliers}</p>
            <p class="metric-label">Quality Suppliers</p>
        </div>
        <div class="metric-card">
            <p class="metric-value">{avg_distance:.1f} mi</p>
            <p class="metric-label">Avg Distance</p>
        </div>
    </div>
    """

def generate_enhanced_executive_report(companies: List[Dict], config: SearchConfig) -> str:
    """Generate enhanced executive report with new scoring"""
    if not companies:
        return "No companies found for analysis."
    
    df = pd.DataFrame(companies)
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    total_companies = len(companies)
    microelectronics_companies = len(df[df['microelectronics_score'] >= 10])
    naval_companies = len(df[df['naval_score'] >= 10])
    high_value_companies = len(df[df['total_score'] >= 15])
    
    # Top companies by different categories
    top_overall = df.nlargest(3, 'total_score')
    top_microelectronics = df.nlargest(3, 'microelectronics_score')
    top_naval = df.nlargest(3, 'naval_score')
    
    report = f"""
# 🎯 WBI Naval Search - Enhanced Intelligence Report

**Generated:** {current_time}  
**Search Location:** {config.base_location}  
**Search Radius:** {config.radius_miles} miles  
**Enhanced Scoring:** Microelectronics, Naval, Defense, Manufacturing

## 📊 Executive Summary

Enhanced search identified **{total_companies} validated suppliers** with multi-category scoring.

### Key Findings:
- **{microelectronics_companies}** companies with microelectronics capabilities (score ≥ 10)
- **{naval_companies}** companies with naval/maritime focus (score ≥ 10)
- **{high_value_companies}** high-value suppliers overall (score ≥ 15)
- **{df['distance_miles'].mean():.1f} miles** average distance from base

## 🏆 Top Suppliers by Category

### Overall Excellence
"""
    
    for i, (_, company) in enumerate(top_overall.iterrows(), 1):
        report += f"""
**{i}. {company['name']}**
- Total Score: {company['total_score']:.1f}
- Microelectronics: {company['microelectronics_score']:.1f} | Naval: {company['naval_score']:.1f}
- Location: {company['location']} ({company['distance_miles']:.1f} mi)
"""
    
    if microelectronics_companies > 0:
        report += f"""

### Microelectronics Leaders
"""
        for i, (_, company) in enumerate(top_microelectronics.iterrows(), 1):
            report += f"""
**{i}. {company['name']}**
- Microelectronics Score: {company['microelectronics_score']:.1f}
- Capabilities: {', '.join(company['capabilities'][:3])}
"""
    
    if naval_companies > 0:
        report += f"""

### Naval/Maritime Specialists
"""
        for i, (_, company) in enumerate(top_naval.iterrows(), 1):
            report += f"""
**{i}. {company['name']}**
- Naval Score: {company['naval_score']:.1f}
- Capabilities: {', '.join(company['capabilities'][:3])}
"""
    
    return report

def main():
    st.set_page_config(
        page_title="WBI Naval Search - Enhanced Supplier Intelligence",
        page_icon="⚓",
        layout="wide"
    )
    
    # Enhanced WBI Styling
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
        position: relative;
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
        border: 2px solid #e2e8f0;
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
    
    .stSidebar > div:first-child {
        background-color: #2d3748;
        border-right: 2px solid #4a5568;
    }
    
    .stSidebar .stSelectbox label,
    .stSidebar .stSlider label,
    .stSidebar .stTextInput label {
        color: #ffffff !important;
        font-weight: 600 !important;
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
    
    # WBI Header
    st.markdown("""
    <div class="wbi-header">
        <div class="wbi-logo-container">
            <div class="wbi-logo">
                ⚓ WBI
            </div>
        </div>
        <h1>Naval Search Pro</h1>
        <p>Advanced supplier intelligence and procurement analytics platform for naval operations.</p>
    </div>
    <div class="wbi-border"></div>
    """, unsafe_allow_html=True)
    
    # Initialize session state
    if 'search_triggered' not in st.session_state:
        st.session_state.search_triggered = False
    if 'companies' not in st.session_state:
        st.session_state.companies = []
    
    # Sidebar configuration
    st.sidebar.header("🔧 Enhanced Search Configuration")
    
    # API Key management
    if not GOOGLE_PLACES_API_KEY:
        st.sidebar.warning("⚠️ No API key detected")
        with st.sidebar.expander("🔑 API Key Setup", expanded=True):
            st.markdown("""
            **Required for Search:**
            1. Visit [Google Cloud Console](https://console.cloud.google.com)
            2. Enable **Places API (New)** + Billing
            3. Create API key and paste below:
            """)
            api_key_input = st.sidebar.text_input(
                "Google Places API Key:", 
                type="password",
                help="Required for company search"
            )
            if api_key_input:
                st.session_state.api_key = api_key_input
                st.sidebar.success("✅ API key configured!")
    else:
        st.sidebar.success("✅ API key found - enhanced search enabled")
    
    # Search configuration
    config = SearchConfig()
    
    st.sidebar.subheader("📍 Search Location")
    
    preset_locations = {
        "South Bend, Indiana (Naval Microelectronics Center)": "South Bend, Indiana",
        "Norfolk, Virginia (Naval Station Norfolk)": "Norfolk, Virginia",
        "San Diego, California (Naval Base San Diego)": "San Diego, California", 
        "Pearl Harbor, Hawaii (Joint Base Pearl Harbor)": "Pearl Harbor, Hawaii",
        "Newport News, Virginia (Newport News Shipbuilding)": "Newport News, Virginia",
        "Bath, Maine (Bath Iron Works)": "Bath, Maine",
        "Groton, Connecticut (Electric Boat)": "Groton, Connecticut"
    }
    
    selected_preset = st.sidebar.selectbox(
        "Select naval facility:",
        list(preset_locations.keys())
    )
    config.base_location = preset_locations[selected_preset]
    
    config.radius_miles = st.sidebar.slider("Search Radius (miles)", 10, 150, 60)
    config.target_company_count = st.sidebar.slider("Max Companies", 10, 300, 150)
    
    # Enhanced search options
    st.sidebar.subheader("🎯 Enhanced Search Options")
    config.search_multi_state = st.sidebar.checkbox(
        "Multi-State Search", 
        value=False,
        help="Search across multiple states for broader coverage"
    )
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col2:
        st.markdown("""
        <div class="wbi-card">
            <h3>🔍 Enhanced Search</h3>
            <p>Advanced Google Places API integration with multi-category scoring for microelectronics, naval, and defense suppliers.</p>
            
            <div style="margin: 1.5rem 0;">
                <h4>🎯 Enhanced Scoring</h4>
                <p>Microelectronics, naval, defense, manufacturing, and quality scoring for comprehensive evaluation.</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("🔍 Enhanced Search", type="primary"):
            # Clear previous results
            if 'companies' in st.session_state:
                del st.session_state.companies
            st.session_state.search_triggered = True
    
    with col1:
        if st.session_state.get('search_triggered', False):
            if not st.session_state.get('companies'):
                with st.spinner("Executing enhanced search with microelectronics focus..."):
                    searcher = EnhancedNavalSearcher(config)
                    companies = searcher.search_companies()
                    st.session_state.companies = companies
                    st.session_state.searcher = searcher
                
                st.session_state.search_triggered = False
    
    # Display enhanced results
    if st.session_state.get('companies'):
        companies = st.session_state.companies
        
        if not companies:
            st.warning("No companies found. Try a different location or check your API key.")
        else:
            # Enhanced metrics dashboard
            st.markdown("## 📊 Enhanced Naval Intelligence Dashboard")
            metrics_html = create_enhanced_metrics_dashboard(companies)
            st.markdown(metrics_html, unsafe_allow_html=True)
            
            # Enhanced tabs
            tab1, tab2, tab3, tab4 = st.tabs([
                "📋 Enhanced Directory", 
                "🗺️ Intelligence Map", 
                "📊 Advanced Analytics", 
                "📄 Enhanced Export"
            ])
            
            with tab1:
                st.subheader("🏭 Enhanced Naval Supplier Directory")
                st.markdown("*Multi-category scoring: Microelectronics, Naval, Defense, Manufacturing*")
                
                # Enhanced filters
                filter_col1, filter_col2, filter_col3, filter_col4 = st.columns(4)
                with filter_col1:
                    min_total_score = st.slider("Min Total Score", 0, 50, 10)
                with filter_col2:
                    min_micro_score = st.slider("Min Microelectronics Score", 0, 30, 0)
                with filter_col3:
                    min_naval_score = st.slider("Min Naval Score", 0, 30, 0)
                with filter_col4:
                    size_filter = st.selectbox("Company Size", ["All", "Small Business", "Medium Business", "Large Corporation"])
                
                # Filter companies
                filtered_companies = companies.copy()
                if min_total_score > 0:
                    filtered_companies = [c for c in filtered_companies if c['total_score'] >= min_total_score]
                if min_micro_score > 0:
                    filtered_companies = [c for c in filtered_companies if c['microelectronics_score'] >= min_micro_score]
                if min_naval_score > 0:
                    filtered_companies = [c for c in filtered_companies if c['naval_score'] >= min_naval_score]
                if size_filter != "All":
                    filtered_companies = [c for c in filtered_companies if c['size'] == size_filter]
                
                st.info(f"📊 Showing {len(filtered_companies)} companies (filtered from {len(companies)} total)")
                
                # Display companies with enhanced scoring
                for company in filtered_companies:
                    # Enhanced score indicators
                    if company['total_score'] >= 25:
                        score_color = "🔥"
                        score_desc = "Exceptional"
                    elif company['total_score'] >= 15:
                        score_color = "🟢"
                        score_desc = "High Value"
                    elif company['total_score'] >= 10:
                        score_color = "🟡"
                        score_desc = "Good Match"
                    else:
                        score_color = "🔴"
                        score_desc = "Basic"
                    
                    with st.expander(f"{score_color} {company['name']} - {score_desc} (Score: {company['total_score']:.1f})"):
                        # Enhanced metric display
                        metric_col1, metric_col2, metric_col3, metric_col4, metric_col5 = st.columns(5)
                        with metric_col1:
                            st.metric("Total Score", f"{company['total_score']:.1f}")
                        with metric_col2:
                            st.metric("Microelectronics", f"{company['microelectronics_score']:.1f}")
                        with metric_col3:
                            st.metric("Naval/Maritime", f"{company['naval_score']:.1f}")
                        with metric_col4:
                            st.metric("Defense", f"{company['defense_score']:.1f}")
                        with metric_col5:
                            st.metric("Manufacturing", f"{company['manufacturing_score']:.1f}")
                        
                        # Company details
                        detail_col1, detail_col2 = st.columns(2)
                        with detail_col1:
                            st.markdown("**📍 Company Information**")
                            st.write(f"📍 {company['location']}")
                            st.write(f"🌐 {company['website']}")
                            st.write(f"📞 {company['phone']}")
                            st.write(f"🏢 Size: {company['size']}")
                            st.write(f"📏 Distance: {company['distance_miles']:.1f} miles")
                        
                        with detail_col2:
                            st.markdown("**🔧 Enhanced Capabilities**")
                            for capability in company['capabilities']:
                                st.write(f"• {capability}")
                            
                            st.markdown("**⭐ Quality Metrics**")
                            st.write(f"Rating: {company['rating']:.1f}⭐ ({company['user_ratings_total']} reviews)")
                        
                        st.markdown(f"**📋 Description:** {company['description']}")
            
            with tab2:
                st.subheader("🗺️ Enhanced Intelligence Map")
                st.markdown("*Geographic distribution with enhanced scoring visualization*")
                
                map_fig = create_enhanced_company_map(companies, config.base_location)
                if map_fig:
                    st.plotly_chart(map_fig, use_container_width=True)
                else:
                    st.info("No valid coordinates available for mapping")
            
            with tab3:
                st.subheader("📊 Enhanced Analytics Dashboard")
                
                if companies:
                    df = pd.DataFrame(companies)
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Enhanced score distribution
                        fig_hist = px.histogram(df, x='total_score', title='Enhanced Score Distribution', nbins=20)
                        st.plotly_chart(fig_hist, use_container_width=True)
                        
                        # Microelectronics vs Naval scoring
                        fig_scatter = px.scatter(df, x='microelectronics_score', y='naval_score', 
                                               hover_name='name', title='Microelectronics vs Naval Scoring',
                                               size='total_score')
                        st.plotly_chart(fig_scatter, use_container_width=True)
                    
                    with col2:
                        # Category scoring comparison
                        score_categories = ['manufacturing_score', 'microelectronics_score', 'naval_score', 'defense_score']
                        avg_scores = [df[cat].mean() for cat in score_categories]
                        category_names = ['Manufacturing', 'Microelectronics', 'Naval', 'Defense']
                        
                        fig_bar = px.bar(x=category_names, y=avg_scores, title='Average Scores by Category')
                        st.plotly_chart(fig_bar, use_container_width=True)
                        
                        # Top capabilities
                        all_caps = []
                        for caps in df['capabilities']:
                            all_caps.extend(caps)
                        if all_caps:
                            cap_counts = pd.Series(all_caps).value_counts().head(10)
                            fig_cap = px.bar(x=cap_counts.values, y=cap_counts.index, 
                                           title='Top Capabilities', orientation='h')
                            st.plotly_chart(fig_cap, use_container_width=True)
            
            with tab4:
                st.subheader("📄 Enhanced Data Export")
                
                if companies:
                    df = pd.DataFrame(companies)
                    
                    # Create clean export DataFrame with logical column order
                    export_df = df[['name', 'location', 'distance_miles', 'size', 'industry', 
                                    'total_score', 'microelectronics_score', 'naval_score', 
                                    'defense_score', 'manufacturing_score', 'rating', 
                                    'user_ratings_total', 'phone', 'website']].copy()

                    # Clean column names
                    export_df.columns = ['Company Name', 'Location', 'Distance (Miles)', 'Company Size', 'Industry',
                                        'Total Score', 'Microelectronics Score', 'Naval Score', 
                                        'Defense Score', 'Manufacturing Score', 'Rating', 'Review Count', 
                                        'Phone', 'Website']

                    # Round numeric columns to 1 decimal place for readability
                    numeric_columns = ['Distance (Miles)', 'Total Score', 'Microelectronics Score', 
                                       'Naval Score', 'Defense Score', 'Manufacturing Score', 'Rating']
                    for col in numeric_columns:
                        export_df[col] = export_df[col].round(1)

                    # Sort by Total Score (highest first)
                    export_df = export_df.sort_values('Total Score', ascending=False)

                    # Add search metadata at the end
                    export_df['Search Location'] = config.base_location
                    export_df['Search Date'] = datetime.now().strftime("%Y-%m-%d")
                    
                    # Display enhanced dataframe with formatting
                    st.subheader("📊 Enhanced Company Data Preview")

                    # Configure column display with proper widths and formatting
                    column_config = {
                        "Company Name": st.column_config.TextColumn("🏭 Company Name", width="large"),
                        "Location": st.column_config.TextColumn("📍 Location", width="medium"),
                        "Distance (Miles)": st.column_config.NumberColumn("📏 Distance", format="%.1f mi"),
                        "Company Size": st.column_config.TextColumn("🏢 Size", width="small"),
                        "Industry": st.column_config.TextColumn("🏭 Industry", width="medium"),
                        "Total Score": st.column_config.NumberColumn("🎯 Total Score", format="%.1f", help="Overall relevance score"),
                        "Microelectronics Score": st.column_config.NumberColumn("🔬 Micro Score", format="%.1f"),
                        "Naval Score": st.column_config.NumberColumn("⚓ Naval Score", format="%.1f"),
                        "Defense Score": st.column_config.NumberColumn("🛡️ Defense Score", format="%.1f"),
                        "Manufacturing Score": st.column_config.NumberColumn("🏭 Mfg Score", format="%.1f"),
                        "Rating": st.column_config.NumberColumn("⭐ Rating", format="%.1f"),
                        "Review Count": st.column_config.NumberColumn("📝 Reviews", format="%d"),
                        "Phone": st.column_config.TextColumn("📞 Phone", width="medium"),
                        "Website": st.column_config.LinkColumn("🌐 Website", width="medium")
                    }

                    # Display formatted dataframe
                    st.dataframe(export_df, column_config=column_config, use_container_width=True, height=400)
                    
                    # Create CSV for download
                    csv = export_df.to_csv(index=False)
                    
                    # Download buttons
                    col1, col2 = st.columns(2)
                    with col1:
                        st.download_button(
                            label="📥 Download CSV (Plain Text)",
                            data=csv,
                            file_name=f"enhanced_naval_companies_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                            mime="text/csv"
                        )
                    
                    with col2:
                        st.button("📊 Excel Export", help="Excel export coming soon", disabled=True)
                    
                    # Enhanced executive report
                    with st.expander("📋 Enhanced Executive Report Preview"):
                        exec_report = generate_enhanced_executive_report(companies, config)
                        st.markdown(exec_report)

if __name__ == "__main__":
    main()