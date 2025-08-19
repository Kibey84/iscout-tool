import streamlit as st
import pandas as pd
import requests
import json
import time
import os
from typing import List, Dict, Optional
import re
from dataclasses import dataclass, field
from geopy.distance import geodesic
from geopy.geocoders import Nominatim
import plotly.express as px
import plotly.graph_objects as go

# API Configuration - Multiple ways to get API key
GOOGLE_PLACES_API_KEY = (
    st.secrets.get("GOOGLE_PLACES_API_KEY") if "GOOGLE_PLACES_API_KEY" in st.secrets else
    os.environ.get("GOOGLE_PLACES_API_KEY", "")
)

# If no API key found, show input field in sidebar
if not GOOGLE_PLACES_API_KEY:
    st.sidebar.info("🔑 To enable real company search, add your Google Places API key:")
    api_key_input = st.sidebar.text_input(
        "Google Places API Key:", 
        type="password",
        help="Get your API key from Google Cloud Console → Places API"
    )
    if api_key_input:
        GOOGLE_PLACES_API_KEY = api_key_input

# Configuration
@dataclass
class SearchConfig:
    base_location: str = "South Bend, Indiana"
    radius_miles: int = 60
    target_company_count: int = 100
    
    # Keywords for company search - using field with default_factory
    manufacturing_keywords: List[str] = field(default_factory=lambda: [
        "advanced manufacturing", "precision machining", "metal fabrication",
        "additive manufacturing", "3D printing", "CNC machining",
        "welding", "assembly", "fabrication", "machining"
    ])
    
    robotics_keywords: List[str] = field(default_factory=lambda: [
        "robotics", "automation", "robotic welding", "industrial automation",
        "automated inspection", "robotics integration", "FANUC", "KUKA",
        "automated systems", "robotic systems"
    ])
    
    unmanned_keywords: List[str] = field(default_factory=lambda: [
        "unmanned systems", "autonomous", "UAV", "UUV", "USV",
        "drone", "autonomous vehicles", "unmanned vehicles",
        "remote systems", "autonomous systems"
    ])

class CompanySearcher:
    def __init__(self, config: SearchConfig):
        self.config = config
        self.geolocator = Nominatim(user_agent="iscout_naval_search")
        self.base_coords = self._get_coordinates(config.base_location)
        
    def _get_coordinates(self, location: str) -> tuple:
        """Get latitude and longitude for a location"""
        try:
            location_data = self.geolocator.geocode(location)
            if location_data:
                return (location_data.latitude, location_data.longitude)
            return (41.6764, -86.2520)  # South Bend default
        except:
            return (41.6764, -86.2520)  # South Bend default
    
    def _calculate_distance(self, lat: float, lon: float) -> float:
        """Calculate distance from base location"""
        try:
            return geodesic(self.base_coords, (lat, lon)).miles
        except:
            return 999  # Invalid location
    
    def _score_company_relevance(self, company_data: Dict) -> Dict:
        """Score company based on relevance to naval requirements"""
        description = company_data.get('description', '').lower()
        name = company_data.get('name', '').lower()
        industry = company_data.get('industry', '').lower()
        
        combined_text = f"{description} {name} {industry}"
        
        scores = {
            'manufacturing_score': 0,
            'robotics_score': 0,
            'unmanned_score': 0,
            'total_score': 0
        }
        
        # Score manufacturing relevance
        for keyword in self.config.manufacturing_keywords:
            if keyword.lower() in combined_text:
                scores['manufacturing_score'] += 1
        
        # Score robotics relevance
        for keyword in self.config.robotics_keywords:
            if keyword.lower() in combined_text:
                scores['robotics_score'] += 2  # Higher weight for robotics
        
        # Score unmanned systems relevance
        for keyword in self.config.unmanned_keywords:
            if keyword.lower() in combined_text:
                scores['unmanned_score'] += 3  # Highest weight for unmanned
        
        scores['total_score'] = (scores['manufacturing_score'] + 
                               scores['robotics_score'] + 
                               scores['unmanned_score'])
        
        return scores
    
    def search_google_places(self, query: str, radius_meters: int = 50000) -> List[Dict]:
        """Search Google Places API for real companies"""
        if not GOOGLE_PLACES_API_KEY:
            st.warning("⚠️ Google Places API key not configured. Using demo data.")
            return []
        
        companies = []
        lat, lon = self.base_coords
        
        # Google Places API endpoint
        url = "https://maps.googleapis.com/maps/api/place/nearbysearch/json"
        
        params = {
            'location': f'{lat},{lon}',
            'radius': radius_meters,
            'keyword': query,
            'type': 'establishment',
            'key': GOOGLE_PLACES_API_KEY
        }
        
        try:
            response = requests.get(url, params=params)
            data = response.json()
            
            if data['status'] == 'OK':
                for place in data['results']:
                    # Get detailed place information
                    place_details = self._get_place_details(place['place_id'])
                    
                    company = {
                        'name': place.get('name', 'Unknown'),
                        'location': place.get('vicinity', 'Unknown'),
                        'industry': ', '.join(place.get('types', [])),
                        'description': place_details.get('description', f"Business in {', '.join(place.get('types', []))}"),
                        'size': 'Unknown',
                        'capabilities': self._extract_capabilities(place),
                        'lat': place['geometry']['location']['lat'],
                        'lon': place['geometry']['location']['lng'],
                        'website': place_details.get('website', 'Not available'),
                        'phone': place_details.get('phone', 'Not available'),
                        'rating': place.get('rating', 0),
                        'user_ratings_total': place.get('user_ratings_total', 0)
                    }
                    companies.append(company)
            
        except Exception as e:
            st.error(f"Error searching Google Places: {e}")
        
        return companies
    
    def _get_place_details(self, place_id: str) -> Dict:
        """Get detailed information about a place"""
        if not GOOGLE_PLACES_API_KEY:
            return {}
            
        url = "https://maps.googleapis.com/maps/api/place/details/json"
        params = {
            'place_id': place_id,
            'fields': 'website,formatted_phone_number,business_status,types',
            'key': GOOGLE_PLACES_API_KEY
        }
        
        try:
            response = requests.get(url, params=params)
            data = response.json()
            
            if data['status'] == 'OK':
                result = data['result']
                return {
                    'website': result.get('website', 'Not available'),
                    'phone': result.get('formatted_phone_number', 'Not available'),
                    'business_status': result.get('business_status', 'Unknown')
                }
        except Exception as e:
            st.error(f"Error getting place details: {e}")
        
        return {}
    
    def _extract_capabilities(self, place: Dict) -> List[str]:
        """Extract likely capabilities from place data"""
        types = place.get('types', [])
        name = place.get('name', '').lower()
        
        capabilities = []
        
        # Map place types to capabilities
        if 'establishment' in types:
            if any(word in name for word in ['manufacturing', 'machining', 'fabrication']):
                capabilities.extend(['Manufacturing', 'Fabrication'])
            if any(word in name for word in ['welding', 'metal']):
                capabilities.extend(['Welding', 'Metal Working'])
            if any(word in name for word in ['automation', 'robotics']):
                capabilities.extend(['Automation', 'Robotics'])
            if any(word in name for word in ['precision', 'cnc']):
                capabilities.extend(['Precision Machining', 'CNC'])
        
        return capabilities if capabilities else ['General Manufacturing']
    
    def search_real_companies(self) -> List[Dict]:
        """Search for real companies using multiple queries"""
        all_companies = []
        
        # Define search queries for different types of companies
        search_queries = [
            "manufacturing",
            "metal fabrication",
            "CNC machining",
            "welding services", 
            "automation",
            "robotics",
            "3D printing",
            "precision machining",
            "industrial equipment"
        ]
        
        radius_meters = self.config.radius_miles * 1609  # Convert miles to meters
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, query in enumerate(search_queries):
            status_text.text(f"Searching for {query} companies...")
            companies = self.search_google_places(query, radius_meters)
            all_companies.extend(companies)
            progress_bar.progress((i + 1) / len(search_queries))
            time.sleep(0.5)  # Rate limiting
        
        status_text.text("Processing results...")
        
        # Remove duplicates based on name and location
        unique_companies = []
        seen = set()
        
        for company in all_companies:
            key = (company['name'].lower(), company['location'].lower())
            if key not in seen:
                seen.add(key)
                
                # Add distance and relevance scoring
                distance = self._calculate_distance(company['lat'], company['lon'])
                company['distance_miles'] = round(distance, 1)
                
                # Add relevance scoring
                scores = self._score_company_relevance(company)
                company.update(scores)
                
                # Filter by distance
                if distance <= self.config.radius_miles:
                    company['within_radius'] = True
                    unique_companies.append(company)
        
        # Sort by relevance score
        unique_companies.sort(key=lambda x: x['total_score'], reverse=True)
        
        progress_bar.empty()
        status_text.empty()
        
        return unique_companies[:self.config.target_company_count]
    
    def generate_sample_companies(self) -> List[Dict]:
        """Generate sample companies for demonstration"""
        # Base coordinates for generating realistic sample data
        base_lat, base_lon = self.base_coords
        
        # Enhanced sample companies with location-relevant names
        location_name = self.config.base_location.split(',')[0]  # Get city name
        
        sample_companies = [
            {
                'name': f'{location_name} Precision Manufacturing Inc.',
                'location': f'{location_name}, {self.config.base_location.split(",")[-1].strip()}',
                'industry': 'Metal Fabrication',
                'description': 'Advanced CNC machining and precision manufacturing for aerospace and defense applications. Specializes in complex metal components.',
                'size': 'Small Business',
                'capabilities': ['CNC Machining', 'Metal Fabrication', 'Quality Control'],
                'lat': base_lat + (0.1 * (hash(f'{location_name} Precision Manufacturing Inc.') % 100 - 50) / 50),
                'lon': base_lon + (0.1 * (hash(f'{location_name} Precision Manufacturing Inc.') % 100 - 50) / 50),
                'website': f'www.{location_name.lower()}precisionmfg.com',
                'phone': '(555) 555-0101',
                'rating': 4.5,
                'user_ratings_total': 23
            },
            {
                'name': f'{location_name} RoboTech Solutions',
                'location': f'Near {location_name}',
                'industry': 'Industrial Automation',
                'description': 'Robotics integration and automation solutions. FANUC and KUKA certified. Automated inspection systems.',
                'size': 'Small Business',
                'capabilities': ['Robotics Integration', 'FANUC Systems', 'Automated Inspection'],
                'lat': base_lat + 0.05,
                'lon': base_lon - 0.05,
                'website': f'www.{location_name.lower()}robotech.com',
                'phone': '(555) 555-0102',
                'rating': 4.8,
                'user_ratings_total': 15
            },
            {
                'name': 'Advanced Additive Manufacturing LLC',
                'location': f'{location_name} Metro Area',
                'industry': 'Additive Manufacturing',
                'description': '3D printing and additive manufacturing for rapid prototyping and production parts. Metal and polymer capabilities.',
                'size': 'Small Business',
                'capabilities': ['3D Printing', 'Metal Additive', 'Rapid Prototyping'],
                'lat': base_lat - 0.03,
                'lon': base_lon + 0.08,
                'website': 'www.advancedadditive.com',
                'phone': '(555) 555-0103',
                'rating': 4.2,
                'user_ratings_total': 31
            },
            {
                'name': 'Regional Autonomous Systems Corp',
                'location': f'{location_name} Region',
                'industry': 'Unmanned Systems',
                'description': 'Development of unmanned systems and autonomous vehicles for defense applications. UAV and UUV expertise.',
                'size': 'Medium Business',
                'capabilities': ['UAV Systems', 'Autonomous Navigation', 'Defense Systems'],
                'lat': base_lat + 0.08,
                'lon': base_lon + 0.12,
                'website': 'www.regionalautonomous.com',
                'phone': '(555) 555-0104',
                'rating': 4.7,
                'user_ratings_total': 8
            },
            {
                'name': f'{location_name} Industrial Welding Specialists',
                'location': f'{location_name} Industrial District',
                'industry': 'Welding Services',
                'description': 'Robotic welding and advanced welding techniques for heavy manufacturing. Certified for military specifications.',
                'size': 'Small Business',
                'capabilities': ['Robotic Welding', 'MIL-SPEC Welding', 'Heavy Fabrication'],
                'lat': base_lat - 0.07,
                'lon': base_lon - 0.09,
                'website': f'www.{location_name.lower()}welding.com',
                'phone': '(555) 555-0105',
                'rating': 4.6,
                'user_ratings_total': 19
            }
        ]
        
        # Generate additional sample companies around the base location
        regional_prefixes = [
            'Metro', 'Regional', 'Coastal', 'Valley', 'Industrial', 'Maritime',
            'Advanced', 'Precision', 'Integrated', 'Strategic', 'Elite', 'Premier'
        ]
        
        company_types = [
            'Manufacturing', 'Technologies', 'Solutions', 'Systems', 'Industries',
            'Fabrication', 'Automation', 'Engineering', 'Dynamics', 'Innovations'
        ]
        
        industries = [
            'Metal Fabrication', 'Industrial Automation', 'Precision Machining',
            'Additive Manufacturing', 'Assembly Services', 'Quality Control',
            'Robotics Integration', 'Welding Services', 'Manufacturing Services'
        ]
        
        capabilities_pool = [
            ['CNC Machining', 'Quality Control', 'Assembly'],
            ['Robotics', 'Automation', 'Integration'],
            ['Welding', 'Fabrication', 'Assembly'],
            ['3D Printing', 'Rapid Prototyping', 'Design'],
            ['Inspection', 'Testing', 'Certification'],
            ['Metal Working', 'Precision Parts', 'Tooling']
        ]
        
        for i in range(min(15, len(regional_prefixes))):  # Generate 15 additional companies
            if len(sample_companies) >= 50:  # Limit sample size
                break
                
            prefix = regional_prefixes[i % len(regional_prefixes)]
            company_type = company_types[i % len(company_types)]
            company_name = f'{prefix} {company_type}'
            
            # Generate coordinates within the search radius
            angle = (i * 137.5) % 360  # Golden angle for good distribution
            distance_factor = 0.3 + (i % 5) * 0.1  # Varying distances
            
            lat_offset = distance_factor * 0.5 * (1 if i % 2 == 0 else -1)
            lon_offset = distance_factor * 0.5 * (1 if i % 3 == 0 else -1)
            
            company = {
                'name': f'{company_name} {"Corp" if i % 3 == 0 else "LLC" if i % 3 == 1 else "Inc."}',
                'location': f'{location_name} Region',
                'industry': industries[i % len(industries)],
                'description': f'Specialized {industries[i % len(industries)].lower()} services for defense and commercial applications.',
                'size': 'Small Business' if i % 4 != 0 else 'Medium Business',
                'capabilities': capabilities_pool[i % len(capabilities_pool)],
                'lat': base_lat + lat_offset,
                'lon': base_lon + lon_offset,
                'website': f'www.{prefix.lower()}{company_type.lower()}.com',
                'phone': f'(555) 555-{1000 + i:04d}',
                'rating': round(3.5 + (i % 15) * 0.1, 1),
                'user_ratings_total': 5 + (i % 25)
            }
            sample_companies.append(company)
        
        # Add distance and relevance scoring for sample companies
        for company in sample_companies:
            distance = self._calculate_distance(company['lat'], company['lon'])
            company['distance_miles'] = round(distance, 1)
            
            # Add relevance scoring
            scores = self._score_company_relevance(company)
            company.update(scores)
            
            # Filter by distance
            if distance <= self.config.radius_miles:
                company['within_radius'] = True
            else:
                company['within_radius'] = False
        
        # Filter and sort
        valid_companies = [c for c in sample_companies if c['within_radius']]
        valid_companies.sort(key=lambda x: x['total_score'], reverse=True)
        
        return valid_companies[:self.config.target_company_count]
    
    def search_companies(self) -> List[Dict]:
        """Main search function - can use real API or demo data"""
        
        # Check if we should use real API search
        use_real_search = st.sidebar.checkbox(
            "🔍 Use Real Company Search", 
            value=False,
            help="Enable this to search real companies using Google Places API (requires API key)"
        )
        
        if use_real_search and GOOGLE_PLACES_API_KEY:
            return self.search_real_companies()
        elif use_real_search and not GOOGLE_PLACES_API_KEY:
            st.sidebar.error("❌ Google Places API key required for real search")
            st.sidebar.info("Add your API key at the top of the code file")
            return self.generate_sample_companies()
        else:
            return self.generate_sample_companies()

def create_company_map(companies: List[Dict], base_coords: tuple):
    """Create interactive map of companies"""
    if not companies:
        return None
    
    df = pd.DataFrame(companies)
    
    # Create color coding based on relevance score
    df['color'] = df['total_score'].apply(lambda x: 
        'red' if x >= 5 else 'orange' if x >= 3 else 'yellow' if x >= 1 else 'lightblue')
    
    fig = go.Figure()
    
    # Add base location
    fig.add_trace(go.Scattermapbox(
        lat=[base_coords[0]],
        lon=[base_coords[1]],
        mode='markers',
        marker=dict(size=20, color='blue'),
        text=['South Bend, IN (Base)'],
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
            center=dict(lat=base_coords[0], lon=base_coords[1]),
            zoom=8
        ),
        height=600,
        title="Naval Supplier Companies in Selected Region"
    )
    
    return fig

def main():
    st.set_page_config(
        page_title="iScout Naval Supplier Search",
        page_icon="⚓",
        layout="wide"
    )
    
    st.title("🔍 iScout Naval Supplier Search Tool")
    st.markdown("### NAVSEA Shipbuilding and Maintenance Supplier Identification")
    
    # Sidebar configuration
    st.sidebar.header("Search Configuration")
    
    config = SearchConfig()
    
    # Location selection
    st.sidebar.subheader("📍 Search Location")
    
    location_option = st.sidebar.radio(
        "Choose search center:",
        ["Preset Naval Locations", "Custom Location"]
    )
    
    if location_option == "Preset Naval Locations":
        preset_locations = {
            "South Bend, Indiana": "South Bend, Indiana",
            "Norfolk, Virginia (Naval Station Norfolk)": "Norfolk, Virginia",
            "San Diego, California (Naval Base San Diego)": "San Diego, California", 
            "Pearl Harbor, Hawaii": "Pearl Harbor, Hawaii",
            "Bremerton, Washington (Puget Sound Naval Shipyard)": "Bremerton, Washington",
            "Portsmouth, New Hampshire (Portsmouth Naval Shipyard)": "Portsmouth, New Hampshire",
            "Newport News, Virginia (Newport News Shipbuilding)": "Newport News, Virginia",
            "Bath, Maine (Bath Iron Works)": "Bath, Maine",
            "Groton, Connecticut (Electric Boat)": "Groton, Connecticut",
            "Pascagoula, Mississippi (Ingalls Shipbuilding)": "Pascagoula, Mississippi"
        }
        
        selected_preset = st.sidebar.selectbox(
            "Select naval location:",
            list(preset_locations.keys())
        )
        config.base_location = preset_locations[selected_preset]
        
    else:  # Custom Location
        config.base_location = st.sidebar.text_input(
            "Enter custom location:",
            value="South Bend, Indiana",
            help="Enter city, state (e.g., 'Detroit, Michigan' or 'Seattle, Washington')"
        )
    
    # Search parameters
    config.radius_miles = st.sidebar.slider("Search Radius (miles)", 10, 100, 60)
    config.target_company_count = st.sidebar.slider("Target Company Count", 10, 200, 100)
    
    # Keyword customization
    with st.sidebar.expander("Customize Keywords"):
        manufacturing_keywords_str = st.text_area(
            "Manufacturing Keywords",
            value=", ".join(config.manufacturing_keywords)
        )
        if manufacturing_keywords_str:
            config.manufacturing_keywords = [k.strip() for k in manufacturing_keywords_str.split(",")]
        
        robotics_keywords_str = st.text_area(
            "Robotics Keywords", 
            value=", ".join(config.robotics_keywords)
        )
        if robotics_keywords_str:
            config.robotics_keywords = [k.strip() for k in robotics_keywords_str.split(",")]
        
        unmanned_keywords_str = st.text_area(
            "Unmanned Systems Keywords",
            value=", ".join(config.unmanned_keywords) 
        )
        if unmanned_keywords_str:
            config.unmanned_keywords = [k.strip() for k in unmanned_keywords_str.split(",")]
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col2:
        st.markdown("### Key Focus Areas")
        st.markdown("""
        **Primary Challenges:**
        - Chronic Maintenance Delays
        - Workforce Crisis / Human Capital Deficit
        
        **Technology Solutions:**
        - Advanced Manufacturing
        - Robotics & Automation
        - Unmanned Systems
        - Additive Manufacturing
        """)
        
        if st.button("🔍 Search Companies", type="primary"):
            st.session_state.search_triggered = True
    
    with col1:
        if st.session_state.get('search_triggered', False):
            with st.spinner("Searching for companies..."):
                searcher = CompanySearcher(config)
                companies = searcher.search_companies()
                st.session_state.companies = companies
                st.session_state.searcher = searcher
    
    # Display results
    if st.session_state.get('companies'):
        companies = st.session_state.companies
        searcher = st.session_state.searcher
        
        st.success(f"Found {len(companies)} relevant companies within {config.radius_miles} miles of {config.base_location}")
        
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Companies", len(companies))
        with col2:
            high_relevance = len([c for c in companies if c['total_score'] >= 5])
            st.metric("High Relevance", high_relevance)
        with col3:
            small_businesses = len([c for c in companies if c['size'] == 'Small Business'])
            st.metric("Small Businesses", small_businesses)
        with col4:
            avg_distance = sum(c['distance_miles'] for c in companies) / len(companies)
            st.metric("Avg Distance", f"{avg_distance:.1f} mi")
        
        # Tabs for different views
        tab1, tab2, tab3, tab4 = st.tabs(["📊 Company List", "🗺️ Map View", "📈 Analytics", "📋 Export"])
        
        with tab1:
            # Company list with filtering
            st.subheader("Company Directory")
            
            # Filters
            col1, col2, col3 = st.columns(3)
            with col1:
                min_score = st.slider("Minimum Relevance Score", 0, 10, 0)
            with col2:
                size_filter = st.selectbox("Company Size", ["All", "Small Business", "Medium Business"])
            with col3:
                max_distance = st.slider("Maximum Distance", 0, config.radius_miles, config.radius_miles)
            
            # Filter companies
            filtered_companies = companies.copy()
            if min_score > 0:
                filtered_companies = [c for c in filtered_companies if c['total_score'] >= min_score]
            if size_filter != "All":
                filtered_companies = [c for c in filtered_companies if c['size'] == size_filter]
            if max_distance < config.radius_miles:
                filtered_companies = [c for c in filtered_companies if c['distance_miles'] <= max_distance]
            
            # Display companies
            for i, company in enumerate(filtered_companies):
                with st.expander(f"🏭 {company['name']} (Score: {company['total_score']})"):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write(f"**Location:** {company['location']}")
                        st.write(f"**Industry:** {company['industry']}")
                        st.write(f"**Size:** {company['size']}")
                        st.write(f"**Distance:** {company['distance_miles']} miles")
                    with col2:
                        st.write(f"**Website:** {company['website']}")
                        st.write(f"**Phone:** {company['phone']}")
                        st.write(f"**Capabilities:** {', '.join(company['capabilities'])}")
                    
                    st.write(f"**Description:** {company['description']}")
                    
                    # Relevance breakdown
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Total Score", company['total_score'])
                    with col2:
                        st.metric("Manufacturing", company['manufacturing_score'])
                    with col3:
                        st.metric("Robotics", company['robotics_score'])
                    with col4:
                        st.metric("Unmanned", company['unmanned_score'])
        
        with tab2:
            st.subheader("Geographic Distribution")
            map_fig = create_company_map(companies, searcher.base_coords)
            if map_fig:
                st.plotly_chart(map_fig, use_container_width=True)
            else:
                st.info("Map requires additional setup. Showing basic location data.")
                df = pd.DataFrame(companies)
                st.dataframe(df[['name', 'location', 'distance_miles', 'total_score']])
        
        with tab3:
            st.subheader("Search Analytics")
            
            df = pd.DataFrame(companies)
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Score distribution
                fig_hist = px.histogram(df, x='total_score', title='Relevance Score Distribution')
                st.plotly_chart(fig_hist, use_container_width=True)
                
                # Industry distribution
                industry_counts = df['industry'].value_counts()
                fig_pie = px.pie(values=industry_counts.values, names=industry_counts.index, 
                               title='Industry Distribution')
                st.plotly_chart(fig_pie, use_container_width=True)
            
            with col2:
                # Distance vs Score scatter
                fig_scatter = px.scatter(df, x='distance_miles', y='total_score', 
                                       hover_name='name', title='Distance vs Relevance Score')
                st.plotly_chart(fig_scatter, use_container_width=True)
                
                # Size distribution
                size_counts = df['size'].value_counts()
                fig_bar = px.bar(x=size_counts.index, y=size_counts.values, 
                               title='Company Size Distribution')
                st.plotly_chart(fig_bar, use_container_width=True)
        
        with tab4:
            st.subheader("Export Results")
            
            df = pd.DataFrame(companies)
            
            # Create downloadable CSV
            csv = df.to_csv(index=False)
            st.download_button(
                label="📥 Download CSV",
                data=csv,
                file_name=f"iscout_naval_suppliers_{pd.Timestamp.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
            
            # Summary report
            st.subheader("Summary Report")
            st.markdown(f"""
            **iScout Naval Supplier Search Results**
            
            - **Search Area:** {config.radius_miles} miles from {config.base_location}
            - **Companies Found:** {len(companies)}
            - **Small Businesses:** {len([c for c in companies if c['size'] == 'Small Business'])}
            - **High Relevance (Score ≥ 5):** {len([c for c in companies if c['total_score'] >= 5])}
            
            **Top 5 Companies by Relevance:**
            """)
            
            top_companies = sorted(companies, key=lambda x: x['total_score'], reverse=True)[:5]
            for i, company in enumerate(top_companies, 1):
                st.markdown(f"{i}. **{company['name']}** - Score: {company['total_score']} - {company['industry']}")

if __name__ == "__main__":
    main()