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
        
        # Higher value keywords for serious naval suppliers
        high_value_keywords = {
            'defense': 5, 'naval': 5, 'aerospace': 4, 'military': 4,
            'contractor': 3, 'corporation': 2, 'industries': 2,
            'technologies': 2, 'systems': 2, 'solutions': 2
        }
        
        # Score manufacturing relevance
        for keyword in self.config.manufacturing_keywords:
            if keyword.lower() in combined_text:
                scores['manufacturing_score'] += 1
        
        # Score robotics relevance  
        for keyword in self.config.robotics_keywords:
            if keyword.lower() in combined_text:
                scores['robotics_score'] += 2
        
        # Score unmanned systems relevance
        for keyword in self.config.unmanned_keywords:
            if keyword.lower() in combined_text:
                scores['unmanned_score'] += 3
        
        # Bonus points for high-value keywords
        for keyword, bonus in high_value_keywords.items():
            if keyword in combined_text:
                scores['manufacturing_score'] += bonus
        
        # Penalty for small operations
        small_operation_keywords = ['mobile', 'roadside', 'auto', 'truck', 'trailer']
        for keyword in small_operation_keywords:
            if keyword in combined_text:
                scores['manufacturing_score'] = max(0, scores['manufacturing_score'] - 3)
        
        scores['total_score'] = (scores['manufacturing_score'] + 
                               scores['robotics_score'] + 
                               scores['unmanned_score'])
        
        return scores
    
    def search_companies(self) -> List[Dict]:
        """Main search function - uses real API if available, otherwise demo data"""
        
        # Check for API key in session state or environment
        api_key = st.session_state.get('api_key', GOOGLE_PLACES_API_KEY)
        
        if api_key:
            st.info("🔍 Searching real companies using Google Places API...")
            return self.search_real_companies()
        else:
            st.info("📋 Using demo data. Add API key in sidebar for real company search.")
            return self.generate_sample_companies()
    
    def search_real_companies(self) -> List[Dict]:
        """Search for real companies using Google Places API"""
        all_companies = []
        
        # Check for API key in session state or environment
        api_key = st.session_state.get('api_key', GOOGLE_PLACES_API_KEY)
        
        if not api_key:
            st.error("🔑 Google Places API key is required for real search")
            return []
        
        # More targeted search queries for naval suppliers
        search_queries = [
            "defense contractors manufacturing",
            "aerospace manufacturing companies", 
            "naval shipbuilding suppliers",
            "military manufacturing",
            "precision machining defense",
            "industrial automation companies",
            "metal fabrication defense contractors",
            "robotics manufacturing companies",
            "additive manufacturing 3D printing"
        ]
        
        progress_bar = st.progress(0)
        
        for i, query in enumerate(search_queries):
            st.write(f"Searching: {query}")
            companies = self.search_google_places_text(query)
            all_companies.extend(companies)
            progress_bar.progress((i + 1) / len(search_queries))
            time.sleep(1)  # Rate limiting
        
        progress_bar.empty()
        
        # Remove duplicates and process
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
                
                # Filter by distance and minimum relevance
                if distance <= self.config.radius_miles and company['total_score'] >= 1:
                    unique_companies.append(company)
        
        # Sort by relevance score
        unique_companies.sort(key=lambda x: x['total_score'], reverse=True)
        
        return unique_companies[:self.config.target_company_count]
    
    def search_google_places_text(self, query: str) -> List[Dict]:
        """Search Google Places using text search"""
        # Check for API key in session state or environment
        api_key = st.session_state.get('api_key', GOOGLE_PLACES_API_KEY)
        
        if not api_key:
            return []
        
        companies = []
        lat, lon = self.base_coords
        
        # Use the simpler text search API
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
            response = requests.post(url, headers=headers, json=request_data, timeout=15)
            
            if response.status_code == 200:
                data = response.json()
                places = data.get('places', [])
                
                for place in places:
                    # Check if this place is within our radius
                    place_lat = place.get('location', {}).get('latitude', 0)
                    place_lon = place.get('location', {}).get('longitude', 0)
                    
                    if place_lat and place_lon:
                        distance = geodesic(self.base_coords, (place_lat, place_lon)).miles
                        
                        if distance <= self.config.radius_miles:
                            name = place.get('displayName', {}).get('text', 'Unknown')
                            types = place.get('types', [])
                            
                            # Filter for manufacturing-related businesses
                            if self._is_manufacturing_related(name, types):
                                company = {
                                    'name': name,
                                    'location': place.get('formattedAddress', 'Unknown'),
                                    'industry': ', '.join(types[:2]),
                                    'description': f"Business type: {', '.join(types[:2])}",
                                    'size': 'Unknown',
                                    'capabilities': self._extract_capabilities_from_name_and_types(name, types),
                                    'lat': place_lat,
                                    'lon': place_lon,
                                    'website': place.get('websiteUri', 'Not available'),
                                    'phone': place.get('nationalPhoneNumber', 'Not available'),
                                    'rating': place.get('rating', 0),
                                    'user_ratings_total': place.get('userRatingCount', 0)
                                }
                                companies.append(company)
                
            elif response.status_code == 403:
                st.error("❌ API key doesn't have permission. Make sure 'Places API (New)' is enabled and billing is set up.")
            else:
                st.warning(f"Search for '{query}' returned status {response.status_code}")
            
        except Exception as e:
            st.warning(f"Error searching for '{query}': {str(e)}")
        
        return companies
    
    def _is_manufacturing_related(self, name: str, types: List[str]) -> bool:
        """Check if a business is manufacturing-related"""
        name_lower = name.lower()
        types_str = ' '.join(types).lower()
        
        # Exclude small welding/repair shops
        exclude_keywords = [
            'mobile welding', 'roadside', 'automotive repair', 'auto repair',
            'truck repair', 'trailer repair', 'small engine', 'lawn mower'
        ]
        
        for exclude in exclude_keywords:
            if exclude in name_lower:
                return False
        
        # Look for serious manufacturing companies
        manufacturing_keywords = [
            'manufacturing', 'fabrication', 'machining', 'industrial',
            'aerospace', 'defense', 'precision', 'cnc', 'automation', 
            'robotics', 'engineering', 'systems', 'technologies',
            'corporation', 'industries', 'solutions'
        ]
        
        # Also look for business types that indicate larger operations
        business_types = [
            'manufacturer', 'contractor', 'engineering', 'technology',
            'industrial', 'aerospace', 'defense'
        ]
        
        # Check name and types
        for keyword in manufacturing_keywords:
            if keyword in name_lower:
                return True
                
        for btype in business_types:
            if btype in types_str:
                return True
        
        # Special case: if it has "welding" but also "fabrication" or "manufacturing"
        if 'welding' in name_lower and ('fabrication' in name_lower or 'manufacturing' in name_lower):
            return True
        
        return False
    
    def _extract_capabilities_from_name_and_types(self, name: str, types: List[str]) -> List[str]:
        """Extract capabilities from name and types"""
        name_lower = name.lower()
        capabilities = []
        
        capability_mapping = {
            'cnc': 'CNC Machining',
            'machining': 'Precision Machining',
            'welding': 'Welding Services',
            'fabrication': 'Metal Fabrication',
            'manufacturing': 'Manufacturing',
            'automation': 'Industrial Automation',
            'robotics': 'Robotics Integration',
            'precision': 'Precision Manufacturing',
            'metal': 'Metal Working',
            'assembly': 'Assembly Services'
        }
        
        for keyword, capability in capability_mapping.items():
            if keyword in name_lower:
                capabilities.append(capability)
        
        return capabilities if capabilities else ['General Manufacturing']
    
    def generate_sample_companies(self) -> List[Dict]:
        """Generate sample companies for demonstration"""
        base_lat, base_lon = self.base_coords
        location_name = self.config.base_location.split(',')[0]
        
        sample_companies = [
            {
                'name': f'{location_name} Precision Manufacturing Inc.',
                'location': f'{location_name}, {self.config.base_location.split(",")[-1].strip()}',
                'industry': 'Metal Fabrication',
                'description': 'Advanced CNC machining and precision manufacturing for aerospace and defense applications.',
                'size': 'Small Business',
                'capabilities': ['CNC Machining', 'Metal Fabrication', 'Quality Control'],
                'lat': base_lat + 0.05,
                'lon': base_lon + 0.03,
                'website': f'www.{location_name.lower()}precision.com',
                'phone': '(555) 555-0101',
                'rating': 4.5,
                'user_ratings_total': 23
            },
            {
                'name': f'{location_name} RoboTech Solutions',
                'location': f'Near {location_name}',
                'industry': 'Industrial Automation',
                'description': 'Robotics integration and automation solutions. FANUC and KUKA certified.',
                'size': 'Small Business',
                'capabilities': ['Robotics Integration', 'FANUC Systems', 'Automated Inspection'],
                'lat': base_lat - 0.08,
                'lon': base_lon + 0.12,
                'website': f'www.{location_name.lower()}robotech.com',
                'phone': '(555) 555-0102',
                'rating': 4.8,
                'user_ratings_total': 15
            },
            {
                'name': 'Advanced Additive Manufacturing LLC',
                'location': f'{location_name} Metro Area',
                'industry': 'Additive Manufacturing',
                'description': '3D printing and additive manufacturing for rapid prototyping and production parts.',
                'size': 'Small Business',
                'capabilities': ['3D Printing', 'Metal Additive', 'Rapid Prototyping'],
                'lat': base_lat + 0.12,
                'lon': base_lon - 0.05,
                'website': 'www.advancedadditive.com',
                'phone': '(555) 555-0103',
                'rating': 4.2,
                'user_ratings_total': 31
            },
            {
                'name': 'Regional Autonomous Systems Corp',
                'location': f'{location_name} Region',
                'industry': 'Unmanned Systems',
                'description': 'Development of unmanned systems and autonomous vehicles for defense applications.',
                'size': 'Medium Business',
                'capabilities': ['UAV Systems', 'Autonomous Navigation', 'Defense Systems'],
                'lat': base_lat - 0.15,
                'lon': base_lon - 0.08,
                'website': 'www.regionalautonomous.com',
                'phone': '(555) 555-0104',
                'rating': 4.7,
                'user_ratings_total': 8
            },
            {
                'name': f'{location_name} Industrial Welding',
                'location': f'{location_name} Industrial District',
                'industry': 'Welding Services',
                'description': 'Robotic welding and advanced welding techniques for heavy manufacturing.',
                'size': 'Small Business',
                'capabilities': ['Robotic Welding', 'MIL-SPEC Welding', 'Heavy Fabrication'],
                'lat': base_lat + 0.18,
                'lon': base_lon + 0.15,
                'website': f'www.{location_name.lower()}welding.com',
                'phone': '(555) 555-0105',
                'rating': 4.6,
                'user_ratings_total': 19
            }
        ]
        
        # Add distance and relevance scoring
        for company in sample_companies:
            distance = self._calculate_distance(company['lat'], company['lon'])
            company['distance_miles'] = round(distance, 1)
            
            scores = self._score_company_relevance(company)
            company.update(scores)
        
        # Sort by relevance score
        sample_companies.sort(key=lambda x: x['total_score'], reverse=True)
        
        return sample_companies

def create_company_map(companies: List[Dict], base_coords: tuple):
    """Create interactive map of companies"""
    if not companies:
        return None
    
    df = pd.DataFrame(companies)
    
    fig = go.Figure()
    
    # Add base location
    fig.add_trace(go.Scattermapbox(
        lat=[base_coords[0]],
        lon=[base_coords[1]],
        mode='markers',
        marker=dict(size=20, color='blue'),
        text=['Search Center'],
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
    
    # API Key input if not found
    if not GOOGLE_PLACES_API_KEY:
        st.sidebar.warning("⚠️ No API key detected")
        st.sidebar.info("🔑 Add your Google Places API key for real company search:")
        st.sidebar.markdown("**Quick Setup:**")
        st.sidebar.markdown("1. Go to [Google Cloud Console](https://console.cloud.google.com)")
        st.sidebar.markdown("2. Enable 'Places API (New)' + Billing")
        st.sidebar.markdown("3. Create API key and paste below:")
        api_key_input = st.sidebar.text_input(
            "Google Places API Key:", 
            type="password",
            help="Real companies will be searched automatically when key is provided"
        )
        if api_key_input:
            # Store in session state for this session
            st.session_state.api_key = api_key_input
            st.sidebar.success("✅ API key added! Search will use real companies.")
    else:
        st.sidebar.success("✅ API key configured - using real company search")
    
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
            help="Enter city, state (e.g., 'Detroit, Michigan')"
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
            avg_distance = sum(c['distance_miles'] for c in companies) / len(companies) if companies else 0
            st.metric("Avg Distance", f"{avg_distance:.1f} mi")
        
        # Tabs for different views
        tab1, tab2, tab3, tab4 = st.tabs(["📊 Company List", "🗺️ Map View", "📈 Analytics", "📋 Export"])
        
        with tab1:
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
            for company in filtered_companies:
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
                st.info("No companies to display on map")
        
        with tab3:
            st.subheader("Search Analytics")
            
            if companies:
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
            
            if companies:
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