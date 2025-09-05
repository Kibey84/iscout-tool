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

class RealNavalSearcher:
    def __init__(self, config: SearchConfig):
        self.config = config
        
    def search_companies(self) -> List[Dict]:
        """Search for real companies using Google Places API"""
        api_key = st.session_state.get('api_key', GOOGLE_PLACES_API_KEY)
        
        if not api_key:
            st.error("🔑 Google Places API key is REQUIRED for real company search")
            st.info("Add your API key in the sidebar to search for actual companies")
            return []
        
        st.info("🔍 Searching for real naval/manufacturing companies...")
        
        # Real search queries
        search_queries = [
            f"manufacturing companies near {self.config.base_location}",
            f"machining companies near {self.config.base_location}",
            f"metal fabrication near {self.config.base_location}",
            f"welding companies near {self.config.base_location}",
            f"precision manufacturing near {self.config.base_location}",
            f"aerospace companies near {self.config.base_location}",
            f"defense contractors near {self.config.base_location}",
            f"shipbuilding near {self.config.base_location}",
            f"marine engineering near {self.config.base_location}",
            f"robotics companies near {self.config.base_location}",
            f"automation companies near {self.config.base_location}",
            f"cnc machining near {self.config.base_location}",
            f"industrial automation near {self.config.base_location}",
            f"electronics manufacturing near {self.config.base_location}",
            f"naval systems near {self.config.base_location}"
        ]
        
        all_companies = []
        progress_bar = st.progress(0)
        
        for i, query in enumerate(search_queries):
            st.write(f"Searching: {query}")
            companies = self._search_google_places(query, api_key)
            all_companies.extend(companies)
            progress_bar.progress((i + 1) / len(search_queries))
            time.sleep(1)  # Rate limiting
        
        progress_bar.empty()
        
        # Process and deduplicate
        unique_companies = self._process_companies(all_companies)
        
        st.success(f"✅ Found {len(unique_companies)} real companies")
        return unique_companies
    
    def _search_google_places(self, query: str, api_key: str) -> List[Dict]:
        """Search Google Places API for real companies"""
        url = "https://places.googleapis.com/v1/places:searchText"
        
        headers = {
            'Content-Type': 'application/json',
            'X-Goog-Api-Key': api_key,
            'X-Goog-FieldMask': 'places.displayName,places.formattedAddress,places.location,places.types,places.websiteUri,places.nationalPhoneNumber,places.rating,places.userRatingCount'
        }
        
        request_data = {
            "textQuery": query,
            "maxResultCount": 20
        }
        
        try:
            response = requests.post(url, headers=headers, json=request_data, timeout=15)
            
            if response.status_code == 200:
                data = response.json()
                return self._process_places_response(data.get('places', []))
            elif response.status_code == 403:
                st.error("❌ API key doesn't have permission. Enable 'Places API (New)' and billing.")
                return []
            else:
                st.warning(f"API returned status {response.status_code} for query: {query}")
                return []
                
        except Exception as e:
            st.warning(f"Search error for '{query}': {str(e)}")
            return []
    
    def _process_places_response(self, places: List[Dict]) -> List[Dict]:
        """Process Google Places response into company data"""
        companies = []
        
        for place in places:
            try:
                name = place.get('displayName', {}).get('text', 'Unknown')
                types = place.get('types', [])
                
                # Filter for relevant businesses
                if self._is_relevant_business(name, types):
                    lat = place.get('location', {}).get('latitude', 0)
                    lon = place.get('location', {}).get('longitude', 0)
                    
                    company = {
                        'name': name,
                        'location': place.get('formattedAddress', 'Unknown'),
                        'industry': ', '.join(types[:3]) if types else 'Manufacturing',
                        'description': self._generate_description(name, types),
                        'size': self._determine_business_size(name, types, place),
                        'capabilities': self._extract_capabilities(name, types),
                        'lat': lat,
                        'lon': lon,
                        'website': place.get('websiteUri', 'Not available'),
                        'phone': place.get('nationalPhoneNumber', 'Not available'),
                        'rating': place.get('rating', 0),
                        'user_ratings_total': place.get('userRatingCount', 0)
                    }
                    companies.append(company)
                    
            except Exception as e:
                continue  # Skip problematic entries
        
        return companies
    
    def _is_relevant_business(self, name: str, types: List[str]) -> bool:
        """Filter for manufacturing/engineering businesses"""
        name_lower = name.lower()
        types_str = ' '.join(types).lower()
        combined = f"{name_lower} {types_str}"
        
        # Exclude obvious non-manufacturers
        exclude_keywords = [
            'restaurant', 'food', 'retail', 'store', 'bank', 'insurance', 
            'real estate', 'school', 'hospital', 'hotel', 'gas station',
            'auto repair', 'automotive repair', 'hair salon', 'nail salon'
        ]
        
        for word in exclude_keywords:
            if word in combined:
                return False
        
        # Include manufacturing/engineering indicators
        include_keywords = [
            'manufacturing', 'machining', 'fabrication', 'welding', 'engineering',
            'aerospace', 'defense', 'precision', 'cnc', 'automation', 'robotics',
            'metal', 'industrial', 'contractor', 'systems', 'technologies',
            'marine', 'naval', 'shipyard', 'electronics', 'semiconductor'
        ]
        
        for word in include_keywords:
            if word in combined:
                return True
        
        # Also include certain business types
        relevant_types = ['manufacturer', 'engineering', 'contractor', 'technology']
        for btype in relevant_types:
            if btype in types_str:
                return True
        
        return False
    
    def _generate_description(self, name: str, types: List[str]) -> str:
        """Generate description based on business name and types"""
        base_desc = f"Business type: {', '.join(types[:2]) if types else 'Manufacturing'}"
        
        name_lower = name.lower()
        if 'aerospace' in name_lower:
            base_desc += " - Aerospace industry focus"
        elif 'defense' in name_lower:
            base_desc += " - Defense contractor"
        elif 'marine' in name_lower or 'naval' in name_lower:
            base_desc += " - Marine/Naval systems"
        elif 'precision' in name_lower:
            base_desc += " - Precision manufacturing"
        
        return base_desc
    
    def _determine_business_size(self, name: str, types: List[str], place_data: Dict) -> str:
        """Determine business size based on indicators"""
        name_lower = name.lower()
        review_count = place_data.get('userRatingCount', 0)
        
        # Major corporations
        major_corps = [
            'honeywell', 'boeing', 'lockheed', 'raytheon', 'northrop', 'general dynamics',
            'bae systems', 'textron', 'collins aerospace', 'general electric'
        ]
        
        for corp in major_corps:
            if corp in name_lower:
                return 'Large Corporation'
        
        # Use review count as size indicator
        if review_count > 100:
            return 'Large Corporation'
        elif review_count > 50:
            return 'Medium Business'
        elif review_count > 10:
            return 'Small-Medium Business'
        else:
            return 'Small Business'
    
    def _extract_capabilities(self, name: str, types: List[str]) -> List[str]:
        """Extract capabilities from name and business types"""
        name_lower = name.lower()
        capabilities = set()
        
        capability_mapping = {
            'cnc': 'CNC Machining',
            'machining': 'Precision Machining',
            'welding': 'Welding Services',
            'fabrication': 'Metal Fabrication',
            'manufacturing': 'Manufacturing',
            'automation': 'Industrial Automation',
            'robotics': 'Robotics Integration',
            'aerospace': 'Aerospace Manufacturing',
            'defense': 'Defense Systems',
            'marine': 'Marine Engineering',
            'electronics': 'Electronics Manufacturing',
            'precision': 'Precision Manufacturing'
        }
        
        combined_text = f"{name_lower} {' '.join(types).lower()}"
        for keyword, capability in capability_mapping.items():
            if keyword in combined_text:
                capabilities.add(capability)
        
        return list(capabilities) if capabilities else ['General Manufacturing']
    
    def _process_companies(self, all_companies: List[Dict]) -> List[Dict]:
        """Remove duplicates and add scoring"""
        unique_companies = []
        seen = set()
        
        for company in all_companies:
            # Duplicate detection
            key = (company['name'].lower(), company['location'][:50].lower())
            
            if key not in seen:
                seen.add(key)
                
                # Add distance calculation (simplified)
                company['distance_miles'] = self._estimate_distance(company['lat'], company['lon'])
                
                # Add scoring
                scores = self._calculate_relevance_scores(company)
                company.update(scores)
                
                # Filter by distance
                if company['distance_miles'] <= self.config.radius_miles:
                    unique_companies.append(company)
        
        # Sort by total score
        unique_companies.sort(key=lambda x: x['total_score'], reverse=True)
        
        return unique_companies[:self.config.target_company_count]
    
    def _estimate_distance(self, lat: float, lon: float) -> float:
        """Simple distance estimation (replace with actual geopy if available)"""
        # Default base coordinates for South Bend
        base_lat, base_lon = 41.6764, -86.2520
        
        if lat == 0 or lon == 0:
            return 999.0
        
        # Simple approximation
        lat_diff = abs(lat - base_lat)
        lon_diff = abs(lon - base_lon)
        distance = ((lat_diff ** 2 + lon_diff ** 2) ** 0.5) * 69  # Rough miles conversion
        
        return round(distance, 1)
    
    def _calculate_relevance_scores(self, company: Dict) -> Dict:
        """Calculate relevance scores for real company data"""
        name = company['name'].lower()
        industry = company['industry'].lower()
        description = company['description'].lower()
        combined = f"{name} {industry} {description}"
        
        scores = {
            'manufacturing_score': 0,
            'robotics_score': 0,
            'unmanned_score': 0,
            'workforce_score': 0,
            'defense_score': 0,
            'total_score': 0
        }
        
        # Manufacturing scoring
        manufacturing_keywords = {
            'manufacturing': 5, 'machining': 4, 'fabrication': 4, 'welding': 3,
            'precision': 3, 'cnc': 4, 'metal': 2, 'aerospace': 6, 'marine': 4
        }
        
        for keyword, points in manufacturing_keywords.items():
            if keyword in combined:
                scores['manufacturing_score'] += points
        
        # Robotics scoring
        robotics_keywords = {
            'robotics': 5, 'automation': 4, 'robotic': 4, 'automated': 3
        }
        
        for keyword, points in robotics_keywords.items():
            if keyword in combined:
                scores['robotics_score'] += points
        
        # Defense scoring
        defense_keywords = {
            'defense': 6, 'naval': 6, 'military': 5, 'aerospace': 4, 'marine': 3
        }
        
        for keyword, points in defense_keywords.items():
            if keyword in combined:
                scores['defense_score'] += points
        
        # Quality bonuses
        rating = company.get('rating', 0)
        review_count = company.get('user_ratings_total', 0)
        
        quality_bonus = 0
        if rating >= 4.5 and review_count >= 10:
            quality_bonus = 3
        elif rating >= 4.0 and review_count >= 5:
            quality_bonus = 2
        
        # Calculate total score
        scores['total_score'] = (
            scores['manufacturing_score'] +
            scores['robotics_score'] +
            scores['unmanned_score'] +
            scores['workforce_score'] +
            scores['defense_score'] +
            quality_bonus
        )
        
        return scores

def create_company_map(companies: List[Dict], base_location: str):
    """Create interactive map of real companies"""
    if not companies:
        return None
    
    df = pd.DataFrame(companies)
    
    # Filter valid coordinates
    df = df[(df['lat'] != 0) & (df['lon'] != 0)]
    
    if df.empty:
        return None
    
    fig = go.Figure()
    
    # Add base location (approximate)
    base_lat, base_lon = 41.6764, -86.2520  # South Bend default
    fig.add_trace(go.Scattermapbox(
        lat=[base_lat],
        lon=[base_lon],
        mode='markers',
        marker=dict(size=20, color='blue'),
        text=[f'Search Center: {base_location}'],
        name='Base Location'
    ))
    
    # Add companies
    fig.add_trace(go.Scattermapbox(
        lat=df['lat'],
        lon=df['lon'],
        mode='markers',
        marker=dict(
            size=10,
            color=df['total_score'],
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title="Relevance Score")
        ),
        text=[f"{row['name']}<br>Score: {row['total_score']}<br>Distance: {row['distance_miles']} mi" 
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
        title="Real Naval Supplier Companies"
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
    defense_contractors = len(df[df['industry'].str.contains('Defense|Naval|Aerospace', case=False, na=False)])
    
    # Get top capability
    all_caps = []
    for caps in df['capabilities']:
        all_caps.extend(caps)
    top_capability = pd.Series(all_caps).value_counts().index[0] if all_caps else "N/A"
    
    quality_suppliers = len(df[(df['rating'] >= 4.0) & (df['user_ratings_total'] >= 10)])
    
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
    """Generate executive report"""
    if not companies:
        return "No companies found for analysis."
    
    df = pd.DataFrame(companies)
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    total_companies = len(companies)
    high_value_companies = len(df[df['total_score'] >= 10])
    defense_companies = len(df[df['industry'].str.contains('Defense|Naval|Aerospace', case=False, na=False)])
    small_businesses = len(df[df['size'].str.contains('Small', na=False)])
    
    # Top companies
    top_companies = df.nlargest(5, 'total_score')
    
    report = f"""
# 🎯 WBI Naval Search - Executive Intelligence Report

**Generated:** {current_time}  
**Search Location:** {config.base_location}  
**Search Radius:** {config.radius_miles} miles  

## 📊 Executive Summary

Real company search identified **{total_companies} verified suppliers** within {config.radius_miles} miles of {config.base_location}.

### Key Findings:
- **{high_value_companies}** companies demonstrate high naval relevance (score ≥ 10)
- **{defense_companies}** companies have direct defense/aerospace focus
- **{small_businesses}** small businesses identified for partnership opportunities
- **{df['distance_miles'].mean():.1f} miles** average supplier distance from base

## 🏆 Top Naval Suppliers (Real Companies)

"""
    
    for i, (_, company) in enumerate(top_companies.iterrows(), 1):
        report += f"""
### {i}. {company['name']}
- **Naval Relevance Score:** {company['total_score']:.1f}/100
- **Location:** {company['location']} ({company['distance_miles']:.1f} miles)
- **Size:** {company['size']}
- **Website:** {company['website']}
- **Phone:** {company['phone']}
- **Rating:** {company['rating']:.1f}⭐ ({company['user_ratings_total']} reviews)

"""
    
    return report

def main():
    st.set_page_config(
        page_title="WBI Naval Search - Real Supplier Intelligence",
        page_icon="⚓",
        layout="wide"
    )
    
    # WBI Styling (same as before)
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
        grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
        gap: 1.5rem;
        margin: 2rem 0;
    }
    
    .metric-card {
        background: #2d3748;
        border-radius: 1rem;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        padding: 2rem;
        border: 1px solid #4a5568;
        text-align: center;
        transition: transform 0.2s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 12px 40px rgba(0, 0, 0, 0.4);
    }
    
    .metric-value {
        font-size: 2.5rem;
        font-weight: 700;
        color: #ffffff !important;
        margin: 0;
        font-family: 'Inter', sans-serif;
    }
    
    .metric-label {
        font-size: 0.875rem;
        color: #cbd5e0 !important;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin: 0.75rem 0 0 0;
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
        <p>Real-time supplier intelligence using live Google Places API data for naval operations.</p>
    </div>
    <div class="wbi-border"></div>
    """, unsafe_allow_html=True)
    
    # Initialize session state
    if 'search_triggered' not in st.session_state:
        st.session_state.search_triggered = False
    if 'companies' not in st.session_state:
        st.session_state.companies = []
    
    # Sidebar configuration
    st.sidebar.header("🔧 Search Configuration")
    
    # API Key management
    if not GOOGLE_PLACES_API_KEY:
        st.sidebar.warning("⚠️ No API key detected")
        with st.sidebar.expander("🔑 API Key Setup", expanded=True):
            st.markdown("""
            **Required for Real Search:**
            1. Visit [Google Cloud Console](https://console.cloud.google.com)
            2. Enable **Places API (New)** + Billing
            3. Create API key and paste below:
            """)
            api_key_input = st.sidebar.text_input(
                "Google Places API Key:", 
                type="password",
                help="Required for real company search"
            )
            if api_key_input:
                st.session_state.api_key = api_key_input
                st.sidebar.success("✅ API key configured!")
    else:
        st.sidebar.success("✅ API key found - real search enabled")
    
    # Search configuration
    config = SearchConfig()
    
    st.sidebar.subheader("📍 Search Location")
    
    preset_locations = {
        "South Bend, Indiana": "South Bend, Indiana",
        "Norfolk, Virginia": "Norfolk, Virginia",
        "San Diego, California": "San Diego, California", 
        "Pearl Harbor, Hawaii": "Pearl Harbor, Hawaii",
        "Newport News, Virginia": "Newport News, Virginia",
        "Bath, Maine": "Bath, Maine",
        "Groton, Connecticut": "Groton, Connecticut"
    }
    
    selected_preset = st.sidebar.selectbox(
        "Select naval location:",
        list(preset_locations.keys())
    )
    config.base_location = preset_locations[selected_preset]
    
    config.radius_miles = st.sidebar.slider("Search Radius (miles)", 10, 100, 60)
    config.target_company_count = st.sidebar.slider("Max Companies", 10, 200, 100)
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col2:
        st.markdown("""
        <div class="wbi-card">
            <h3>🔍 Real Company Search</h3>
            <p>Live Google Places API integration searches for actual manufacturing and naval companies - no fake data.</p>
            
            <div style="margin: 1.5rem 0;">
                <h4>✅ Real Results</h4>
                <p>Every company result comes directly from Google Places API with verified business information.</p>
            </div>
            
            <div style="margin: 1.5rem 0;">
                <h4>📊 Live Analytics</h4>
                <p>Real-time scoring and analytics based on actual business data and customer reviews.</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("🔍 Search Real Companies", type="primary"):
            # Clear previous results
            if 'companies' in st.session_state:
                del st.session_state.companies
            st.session_state.search_triggered = True
    
    with col1:
        if st.session_state.get('search_triggered', False):
            if not st.session_state.get('companies'):
                with st.spinner("Searching for real companies..."):
                    searcher = RealNavalSearcher(config)
                    companies = searcher.search_companies()
                    st.session_state.companies = companies
                    st.session_state.searcher = searcher
                
                st.session_state.search_triggered = False
    
    # Display results
    if st.session_state.get('companies'):
        companies = st.session_state.companies
        
        if not companies:
            st.warning("No companies found. Try a different location or check your API key.")
        else:
            # Enhanced metrics dashboard
            st.markdown("## 📊 Real Supplier Intelligence Dashboard")
            metrics_html = create_enhanced_metrics_dashboard(companies)
            st.markdown(metrics_html, unsafe_allow_html=True)
            
            # Tabs for different views
            tab1, tab2, tab3, tab4 = st.tabs([
                "📋 Real Supplier Directory", 
                "🗺️ Geographic View", 
                "📊 Analytics", 
                "📄 Export Data"
            ])
            
            with tab1:
                st.subheader("🏭 Real Naval Supplier Directory")
                st.markdown("*Live data from Google Places API - all companies verified*")
                
                # Filters
                filter_col1, filter_col2, filter_col3 = st.columns(3)
                with filter_col1:
                    min_score = st.slider("Min Relevance Score", 0, 20, 0)
                with filter_col2:
                    size_filter = st.selectbox("Company Size", ["All", "Small Business", "Medium Business", "Large Corporation"])
                with filter_col3:
                    min_rating = st.slider("Min Rating", 0.0, 5.0, 0.0, 0.1)
                
                # Filter companies
                filtered_companies = companies.copy()
                if min_score > 0:
                    filtered_companies = [c for c in filtered_companies if c['total_score'] >= min_score]
                if size_filter != "All":
                    filtered_companies = [c for c in filtered_companies if c['size'] == size_filter]
                if min_rating > 0:
                    filtered_companies = [c for c in filtered_companies if c['rating'] >= min_rating]
                
                st.info(f"📊 Showing {len(filtered_companies)} real companies (filtered from {len(companies)} total)")
                
                # Display companies
                for company in filtered_companies:
                    score_color = "🟢" if company['total_score'] >= 15 else "🟡" if company['total_score'] >= 8 else "🔴"
                    
                    with st.expander(f"{score_color} {company['name']} - Score: {company['total_score']:.1f}"):
                        col1, col2 = st.columns(2)
                        with col1:
                            st.markdown("**📍 Real Business Information**")
                            st.write(f"📍 {company['location']}")
                            st.write(f"🌐 {company['website']}")
                            st.write(f"📞 {company['phone']}")
                            st.write(f"🏢 Size: {company['size']}")
                            st.write(f"🏭 Industry: {company['industry']}")
                        
                        with col2:
                            st.markdown("**🔧 Capabilities**")
                            for capability in company['capabilities']:
                                st.write(f"• {capability}")
                            
                            st.markdown("**⭐ Customer Reviews**")
                            st.write(f"Rating: {company['rating']:.1f}⭐ ({company['user_ratings_total']} reviews)")
                            st.write(f"Distance: {company['distance_miles']:.1f} miles")
                        
                        st.markdown(f"**📋 Description:** {company['description']}")
                        
                        # Score breakdown
                        st.markdown("**🎯 Relevance Analysis**")
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
                            st.metric("🛡️ Defense", f"{company['defense_score']:.1f}")
            
            with tab2:
                st.subheader("🗺️ Real Company Locations")
                st.markdown("*Geographic distribution of verified companies*")
                
                map_fig = create_company_map(companies, config.base_location)
                if map_fig:
                    st.plotly_chart(map_fig, use_container_width=True)
                else:
                    st.info("No valid coordinates available for mapping")
            
            with tab3:
                st.subheader("📊 Real Data Analytics")
                
                if companies:
                    df = pd.DataFrame(companies)
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Score distribution
                        fig_hist = px.histogram(df, x='total_score', title='Relevance Score Distribution')
                        st.plotly_chart(fig_hist, use_container_width=True)
                        
                        # Size distribution
                        size_counts = df['size'].value_counts()
                        fig_pie = px.pie(values=size_counts.values, names=size_counts.index, 
                                       title='Company Size Distribution')
                        st.plotly_chart(fig_pie, use_container_width=True)
                    
                    with col2:
                        # Distance vs Score
                        fig_scatter = px.scatter(df, x='distance_miles', y='total_score', 
                                               hover_name='name', title='Distance vs Relevance Score')
                        st.plotly_chart(fig_scatter, use_container_width=True)
                        
                        # Rating distribution
                        fig_rating = px.histogram(df, x='rating', title='Customer Rating Distribution')
                        st.plotly_chart(fig_rating, use_container_width=True)
            
            with tab4:
                st.subheader("📄 Export Real Company Data")
                
                if companies:
                    df = pd.DataFrame(companies)
                    
                    # Create export DataFrame
                    export_df = df[['name', 'location', 'industry', 'size', 'total_score', 
                                    'manufacturing_score', 'defense_score', 'distance_miles', 
                                    'rating', 'user_ratings_total', 'website', 'phone']].copy()
                    
                    export_df.columns = ['Company Name', 'Location', 'Industry', 'Size', 
                                        'Total Score', 'Manufacturing Score', 'Defense Score', 
                                        'Distance (Miles)', 'Rating', 'Review Count', 'Website', 'Phone']
                    
                    csv = export_df.to_csv(index=False)
                    
                    st.download_button(
                        label="📥 Download Real Company Data (CSV)",
                        data=csv,
                        file_name=f"real_naval_companies_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                        mime="text/csv"
                    )
                    
                    # Executive report
                    with st.expander("📋 Executive Report Preview"):
                        exec_report = generate_executive_report(companies, config)
                        st.markdown(exec_report)

if __name__ == "__main__":
    main()