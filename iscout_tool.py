import streamlit as st
import pandas as pd
import requests
import json
import time
from typing import List, Dict, Optional
import re
from dataclasses import dataclass, field
from geopy.distance import geodesic
from geopy.geocoders import Nominatim
import plotly.express as px
import plotly.graph_objects as go

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
    
    def generate_sample_companies(self) -> List[Dict]:
        """Generate sample companies for demonstration"""
        sample_companies = [
            {
                'name': 'Precision Manufacturing Inc.',
                'location': 'South Bend, IN',
                'industry': 'Metal Fabrication',
                'description': 'Advanced CNC machining and precision manufacturing for aerospace and defense applications. Specializes in complex metal components.',
                'size': 'Small Business',
                'capabilities': ['CNC Machining', 'Metal Fabrication', 'Quality Control'],
                'lat': 41.6764 + (0.1 * (hash('Precision Manufacturing Inc.') % 100 - 50) / 50),
                'lon': -86.2520 + (0.1 * (hash('Precision Manufacturing Inc.') % 100 - 50) / 50),
                'website': 'www.precisionmfg.com',
                'phone': '(574) 555-0101'
            },
            {
                'name': 'RoboTech Solutions',
                'location': 'Elkhart, IN',
                'industry': 'Industrial Automation',
                'description': 'Robotics integration and automation solutions. FANUC and KUKA certified. Automated inspection systems.',
                'size': 'Small Business',
                'capabilities': ['Robotics Integration', 'FANUC Systems', 'Automated Inspection'],
                'lat': 41.6820,
                'lon': -85.9767,
                'website': 'www.robotechsolutions.com',
                'phone': '(574) 555-0102'
            },
            {
                'name': 'Advanced Additive Manufacturing',
                'location': 'Mishawaka, IN',
                'industry': 'Additive Manufacturing',
                'description': '3D printing and additive manufacturing for rapid prototyping and production parts. Metal and polymer capabilities.',
                'size': 'Small Business',
                'capabilities': ['3D Printing', 'Metal Additive', 'Rapid Prototyping'],
                'lat': 41.6620,
                'lon': -86.1586,
                'website': 'www.advancedadditive.com',
                'phone': '(574) 555-0103'
            },
            {
                'name': 'Autonomous Systems Corp',
                'location': 'Fort Wayne, IN',
                'industry': 'Unmanned Systems',
                'description': 'Development of unmanned systems and autonomous vehicles for defense applications. UAV and UUV expertise.',
                'size': 'Medium Business',
                'capabilities': ['UAV Systems', 'Autonomous Navigation', 'Defense Systems'],
                'lat': 41.0793,
                'lon': -85.1394,
                'website': 'www.autonomoussystems.com',
                'phone': '(260) 555-0104'
            },
            {
                'name': 'Industrial Welding Specialists',
                'location': 'Goshen, IN',
                'industry': 'Welding Services',
                'description': 'Robotic welding and advanced welding techniques for heavy manufacturing. Certified for military specifications.',
                'size': 'Small Business',
                'capabilities': ['Robotic Welding', 'MIL-SPEC Welding', 'Heavy Fabrication'],
                'lat': 41.5823,
                'lon': -85.8344,
                'website': 'www.indwelding.com',
                'phone': '(574) 555-0105'
            }
        ]
        
        # Generate additional sample companies
        base_names = [
            'Midwest Precision', 'Great Lakes Manufacturing', 'Hoosier Tech',
            'Northern Indiana Fabrication', 'Lakeland Automation', 'Tri-State Manufacturing',
            'Indiana Advanced Systems', 'Heartland Robotics', 'Michiana Manufacturing',
            'Prairie Manufacturing', 'Crossroads Automation', 'Heritage Manufacturing'
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
        
        for i, base_name in enumerate(base_names):
            if len(sample_companies) >= 50:  # Limit sample size
                break
                
            company = {
                'name': f'{base_name} {"Corp" if i % 3 == 0 else "LLC" if i % 3 == 1 else "Inc."}',
                'location': f'Indiana City {i+1}, IN',
                'industry': industries[i % len(industries)],
                'description': f'Specialized manufacturing and technology services. Expert in {industries[i % len(industries)].lower()} solutions.',
                'size': 'Small Business' if i % 4 != 0 else 'Medium Business',
                'capabilities': capabilities_pool[i % len(capabilities_pool)],
                'lat': 41.6764 + (0.5 * (hash(base_name) % 100 - 50) / 50),
                'lon': -86.2520 + (0.5 * (hash(base_name) % 100 - 50) / 50),
                'website': f'www.{base_name.lower().replace(" ", "")}.com',
                'phone': f'(574) 555-{1000 + i:04d}'
            }
            sample_companies.append(company)
        
        return sample_companies
    
    def search_companies(self) -> List[Dict]:
        """Main search function - returns sample data for demo"""
        companies = self.generate_sample_companies()
        
        # Add distance and relevance scoring
        for company in companies:
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
        valid_companies = [c for c in companies if c['within_radius']]
        valid_companies.sort(key=lambda x: x['total_score'], reverse=True)
        
        return valid_companies[:self.config.target_company_count]

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
        title="Naval Supplier Companies in South Bend Region"
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
        
        st.success(f"Found {len(companies)} relevant companies within {config.radius_miles} miles of South Bend, IN")
        
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
            
            - **Search Area:** {config.radius_miles} miles from South Bend, IN
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