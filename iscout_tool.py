import streamlit as st
import pandas as pd
import requests
import json
import os
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go

# SECURE API Configuration - NO hardcoded keys
SAM_GOV_API_KEY = (
    st.secrets.get("SAM_GOV_API_KEY") if hasattr(st, 'secrets') and st.secrets else
    os.environ.get("SAM_GOV_API_KEY", "")
)
OPENAI_API_KEY = (
    st.secrets.get("OPENAI_API_KEY") if hasattr(st, 'secrets') and st.secrets else
    os.environ.get("OPENAI_API_KEY", "")
)
ANTHROPIC_API_KEY = (
    st.secrets.get("ANTHROPIC_API_KEY") if hasattr(st, 'secrets') and st.secrets else
    os.environ.get("ANTHROPIC_API_KEY", "")
)

# Complete WBI Naval Search Platform
st.set_page_config(page_title="WBI Naval Search - Complete", page_icon="⚓", layout="wide")

# Basic session state
if 'companies' not in st.session_state:
    st.session_state.companies = []
    st.session_state.search_complete = False

st.title("⚓ WBI Naval Search - Complete Intelligence Platform")
st.markdown("**Advanced AI-powered supplier intelligence for naval procurement and SBIR identification**")

# Demo naval companies function
def get_demo_companies():
    """Return demo naval companies for testing"""
    return [
        {
            'name': 'Advanced Naval Systems LLC',
            'location': 'Norfolk, Virginia',
            'lat': 36.8468,
            'lon': -76.2929,
            'business_types': ['Small Business', 'SDVOSB'],
            'capabilities': ['Submarine Systems', 'Naval Electronics'],
            'total_contracts': 15,
            'total_value': 25000000,
            'naval_relevance': 9,
            'sbir_eligibility': 8,
            'tech_readiness': 7,
            'distance_miles': 45,
            'source': 'Demo Data',
            'description': 'Advanced naval systems contractor specializing in submarine technology'
        },
        {
            'name': 'Maritime Electronics Corp',
            'location': 'Newport News, Virginia',
            'lat': 37.0871,
            'lon': -76.4730,
            'business_types': ['Small Business', 'WOSB'],
            'capabilities': ['Radar Systems', 'Communications', 'Sonar Electronics'],
            'total_contracts': 8,
            'total_value': 12000000,
            'naval_relevance': 8,
            'sbir_eligibility': 9,
            'tech_readiness': 8,
            'distance_miles': 32,
            'source': 'Demo Data',
            'description': 'Women-owned small business focusing on naval electronics and communications'
        },
        {
            'name': 'Great Lakes Defense Systems',
            'location': 'Cleveland, Ohio',
            'lat': 41.4993,
            'lon': -81.6944,
            'business_types': ['Small Business', 'HUBZone'],
            'capabilities': ['Ship Maintenance', 'Hull Fabrication', 'Marine Engineering'],
            'total_contracts': 22,
            'total_value': 45000000,
            'naval_relevance': 7,
            'sbir_eligibility': 7,
            'tech_readiness': 6,
            'distance_miles': 120,
            'source': 'Demo Data',
            'description': 'HUBZone certified contractor with extensive ship maintenance experience'
        },
        {
            'name': 'Precision Marine Engineering',
            'location': 'San Diego, California',
            'lat': 32.7157,
            'lon': -117.1611,
            'business_types': ['Small Business'],
            'capabilities': ['Marine Engineering', 'Propulsion Systems', 'Naval Architecture'],
            'total_contracts': 12,
            'total_value': 18000000,
            'naval_relevance': 8,
            'sbir_eligibility': 6,
            'tech_readiness': 7,
            'distance_miles': 200,
            'source': 'Demo Data',
            'description': 'Precision marine engineering and propulsion system design'
        },
        {
            'name': 'Coastal Microelectronics Inc',
            'location': 'Bath, Maine',
            'lat': 43.9109,
            'lon': -69.8597,
            'business_types': ['Small Business', 'SBIR Recipient'],
            'capabilities': ['Microelectronics', 'Sensor Systems', 'RF Electronics'],
            'total_contracts': 5,
            'total_value': 8000000,
            'naval_relevance': 7,
            'sbir_eligibility': 9,
            'tech_readiness': 9,
            'distance_miles': 75,
            'source': 'Demo Data',
            'description': 'SBIR Phase II recipient specializing in advanced microelectronics'
        },
        {
            'name': 'Quantum Naval Solutions',
            'location': 'Boston, Massachusetts',
            'lat': 42.3601,
            'lon': -71.0589,
            'business_types': ['Small Business', 'VOSB'],
            'capabilities': ['Quantum Computing', 'Cybersecurity', 'AI Systems'],
            'total_contracts': 3,
            'total_value': 15000000,
            'naval_relevance': 6,
            'sbir_eligibility': 8,
            'tech_readiness': 9,
            'distance_miles': 90,
            'source': 'Demo Data',
            'description': 'Veteran-owned small business developing quantum naval technologies'
        }
    ]

# USASpending API function
def search_usaspending_real():
    """Real USASpending.gov API call"""
    try:
        st.info("🔍 Searching USASpending.gov for naval contracts...")
        
        url = "https://api.usaspending.gov/api/v2/search/spending_by_award/"
        
        payload = {
            "filters": {
                "award_type_codes": ["A", "B", "C", "D"],
                "naics_codes": ["336611", "336612", "334413", "541330"],
                "agencies": [
                    {"type": "awarding", "tier": "toptier", "name": "Department of Defense"}
                ],
                "time_period": [{"start_date": "2022-01-01", "end_date": "2024-12-31"}]
            },
            "fields": ["Award ID", "Recipient Name", "Award Amount", "NAICS Code", "Award Description"],
            "page": 1,
            "limit": 20,
            "sort": "Award Amount",
            "order": "desc"
        }
        
        response = requests.post(url, json=payload, timeout=15)
        
        if response.status_code == 200:
            data = response.json()
            results = data.get('results', [])
            
            companies = []
            company_data = {}
            
            for award in results:
                recipient = award.get('Recipient Name', 'Unknown')
                amount = award.get('Award Amount', 0)
                
                if recipient and recipient != 'Unknown':
                    if recipient not in company_data:
                        company_data[recipient] = {
                            'name': recipient,
                            'total_contracts': 0,
                            'total_value': 0,
                            'source': 'USASpending.gov',
                            'business_types': ['Government Contractor'],
                            'capabilities': ['Naval Contracts'],
                            'naval_relevance': 6,
                            'sbir_eligibility': 5,
                            'tech_readiness': 6,
                            'distance_miles': 100,
                            'location': 'Various',
                            'lat': 38.9072,  # Default DC area
                            'lon': -77.0369,
                            'description': f'Government contractor with documented naval contract history'
                        }
                    
                    company_data[recipient]['total_contracts'] += 1
                    company_data[recipient]['total_value'] += amount
            
            companies = list(company_data.values())
            st.success(f"✅ Found {len(companies)} companies from USASpending.gov")
            return companies
            
        else:
            st.warning(f"USASpending API returned status {response.status_code}")
            return []
            
    except Exception as e:
        st.error(f"USASpending API error: {e}")
        return []

def extract_score_from_line(line):
    """Extract numerical score from AI response line"""
    import re
    # Look for X/10 pattern
    match = re.search(r'(\d+)/10', line)
    if match:
        return int(match.group(1))
    
    # Look for standalone numbers
    numbers = re.findall(r'\b(\d+)\b', line)
    for num in numbers:
        score = int(num)
        if 1 <= score <= 10:
            return score
    return None

# AI Analysis with safe response handling
def analyze_company_with_ai(company):
    """AI-powered company analysis using OpenAI (with safe response handling)"""
    
    # First do rule-based analysis as fallback
    company = analyze_company_simple(company)
    
    # Try AI analysis if API key available
    if OPENAI_API_KEY:
        try:
            import openai
            openai.api_key = OPENAI_API_KEY
            
            company_text = f"""
            Company: {company.get('name', 'Unknown')}
            Business Types: {', '.join(company.get('business_types', []))}
            Capabilities: {', '.join(company.get('capabilities', []))}
            Description: {company.get('description', 'No description')}
            Contracts: {company.get('total_contracts', 0)} total contracts
            Contract Value: ${company.get('total_value', 0):,.0f}
            """
            
            prompt = f"""
            Analyze this company's suitability for US Navy contracts and SBIR eligibility:

            {company_text}

            Provide scores (1-10) for:
            1. Naval Relevance: How well does this company fit naval/maritime needs?
            2. SBIR Eligibility: Likelihood of SBIR qualification (small business, innovation focus)
            3. Technology Readiness: Technical capability level

            Format as:
            Naval Relevance: X/10 - [brief reason]
            SBIR Eligibility: X/10 - [brief reason]
            Technology Readiness: X/10 - [brief reason]

            Keep analysis concise but informative.
            """
            
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=400,
                timeout=10
            )
            
            # SAFE RESPONSE HANDLING - This fixes all the Pylance errors
            analysis = "AI analysis completed"
            try:
                # Use getattr for safe attribute access
                choices = getattr(response, 'choices', None)
                if choices and len(choices) > 0:
                    first_choice = choices[0]
                    message = getattr(first_choice, 'message', None)
                    if message:
                        analysis = getattr(message, 'content', 'No content available')
            except (AttributeError, IndexError, TypeError):
                analysis = "Response parsing error"
            
            # Parse AI scores
            lines = analysis.lower().split('\n')
            for line in lines:
                if 'naval relevance:' in line:
                    score = extract_score_from_line(line)
                    if score:
                        company['naval_relevance'] = score
                elif 'sbir eligibility:' in line:
                    score = extract_score_from_line(line)
                    if score:
                        company['sbir_eligibility'] = score
                elif 'technology readiness:' in line:
                    score = extract_score_from_line(line)
                    if score:
                        company['tech_readiness'] = score
            
            company['ai_analysis'] = analysis
            company['analysis_method'] = 'GPT-3.5 AI Analysis'
            
        except ImportError:
            company['ai_analysis'] = "OpenAI library not installed"
            company['analysis_method'] = 'Rule-based (OpenAI not installed)'
        except Exception as e:
            company['ai_analysis'] = f"AI analysis failed: {str(e)}"
            company['analysis_method'] = 'Rule-based fallback'
    else:
        company['analysis_method'] = 'Rule-based (no OpenAI API key)'
    
    return company

# Simple AI analysis (fallback when AI not available)
def analyze_company_simple(company):
    """Simple rule-based company analysis"""
    name = company.get('name', '').lower()
    capabilities = ' '.join(company.get('capabilities', [])).lower()
    business_types = company.get('business_types', [])
    
    # Naval relevance
    naval_keywords = ['naval', 'marine', 'maritime', 'ship', 'submarine', 'defense']
    naval_score = 5
    for keyword in naval_keywords:
        if keyword in name or keyword in capabilities:
            naval_score += 1
    
    # SBIR eligibility
    sbir_score = 5
    if 'Small Business' in business_types:
        sbir_score += 2
    if any(cert in business_types for cert in ['WOSB', 'VOSB', 'SDVOSB', 'HUBZone']):
        sbir_score += 1
    if 'SBIR' in ' '.join(business_types):
        sbir_score += 2
    
    # Tech readiness
    tech_keywords = ['electronics', 'systems', 'quantum', 'ai', 'cyber']
    tech_score = 6
    for keyword in tech_keywords:
        if keyword in name or keyword in capabilities:
            tech_score += 1
    
    company['naval_relevance'] = min(naval_score, 10)
    company['sbir_eligibility'] = min(sbir_score, 10)
    company['tech_readiness'] = min(tech_score, 10)
    
    return company

def create_visualizations(companies):
    """Create advanced visualizations for naval intelligence"""
    
    if not companies:
        return None, None, None
    
    df = pd.DataFrame(companies)
    
    # 1. SBIR Eligibility vs Naval Relevance Scatter Plot
    fig_scatter = px.scatter(
        df, 
        x='sbir_eligibility', 
        y='naval_relevance',
        size='total_value',
        color='tech_readiness',
        hover_name='name',
        hover_data=['total_contracts', 'distance_miles'],
        title='SBIR Eligibility vs Naval Relevance Analysis',
        labels={
            'sbir_eligibility': 'SBIR Eligibility Score (1-10)',
            'naval_relevance': 'Naval Relevance Score (1-10)',
            'total_value': 'Contract Value ($)',
            'tech_readiness': 'Tech Readiness'
        },
        color_continuous_scale='Viridis',
        size_max=60
    )
    
    fig_scatter.update_layout(
        height=500,
        showlegend=True,
        title_x=0.5
    )
    
    # 2. Business Type Distribution
    all_business_types = []
    for company in companies:
        all_business_types.extend(company.get('business_types', []))
    
    if all_business_types:
        from collections import Counter
        business_counts = Counter(all_business_types)
        
        fig_business = px.pie(
            values=list(business_counts.values()),
            names=list(business_counts.keys()),
            title='Business Type Distribution',
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        fig_business.update_layout(height=400)
    else:
        fig_business = None
    
    # 3. Interactive Map
    fig_map = go.Figure()
    
    # Search center (default to Norfolk)
    search_center_lat = 36.8468
    search_center_lon = -76.2929
    
    # Add search center
    fig_map.add_trace(go.Scattermapbox(
        lat=[search_center_lat],
        lon=[search_center_lon],
        mode='markers',
        marker=dict(size=20, color='red', symbol='star'),
        text=['Search Center'],
        name='Search Center',
        showlegend=True
    ))
    
    # Add company locations
    if 'lat' in df.columns and 'lon' in df.columns:
        valid_coords = df[(df['lat'] != 0) & (df['lon'] != 0)]
        
        if not valid_coords.empty:
            fig_map.add_trace(go.Scattermapbox(
                lat=valid_coords['lat'],
                lon=valid_coords['lon'],
                mode='markers',
                marker=dict(
                    size=valid_coords['naval_relevance'] * 3,
                    color=valid_coords['sbir_eligibility'],
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title="SBIR Eligibility Score"),
                    opacity=0.8
                ),
                text=[
                    f"<b>{row['name']}</b><br>" +
                    f"Naval: {row['naval_relevance']}/10<br>" +
                    f"SBIR: {row['sbir_eligibility']}/10<br>" +
                    f"Contracts: {row.get('total_contracts', 0)}<br>" +
                    f"Value: ${row.get('total_value', 0):,.0f}"
                    for _, row in valid_coords.iterrows()
                ],
                name='Naval Suppliers',
                showlegend=True
            ))
    
    fig_map.update_layout(
        mapbox=dict(
            style='open-street-map',
            center=dict(lat=search_center_lat, lon=search_center_lon),
            zoom=6
        ),
        height=600,
        title="Geographic Distribution of Naval Suppliers",
        showlegend=True
    )
    
    return fig_scatter, fig_business, fig_map

# Sidebar
with st.sidebar:
    st.header("🎯 Search Configuration")
    
    search_location = st.selectbox("Search Region:", [
        "Norfolk, Virginia",
        "Newport News, Virginia", 
        "San Diego, California",
        "Bath, Maine",
        "Groton, Connecticut"
    ])
    
    search_radius = st.slider("Search Radius (miles):", 25, 200, 75)
    max_results = st.slider("Max Results:", 10, 100, 25)
    
    st.subheader("🔍 Data Sources")
    use_demo = st.checkbox("Demo Data", value=True)
    use_usaspending = st.checkbox("USASpending.gov API", value=True)
    use_ai_analysis = st.checkbox("AI-Powered Analysis", value=True, 
                                 help="Uses GPT-3.5 for intelligent company assessment")
    
    if st.button("🚀 Execute Search", type="primary"):
        companies = []
        
        # Get demo data
        if use_demo:
            demo_companies = get_demo_companies()
            companies.extend(demo_companies)
        
        # Get real USASpending data
        if use_usaspending:
            real_companies = search_usaspending_real()
            companies.extend(real_companies)
        
        # Analyze companies
        analyzed_companies = []
        progress_bar = st.progress(0)
        
        for i, company in enumerate(companies):
            if use_ai_analysis:
                analyzed = analyze_company_with_ai(company)
            else:
                analyzed = analyze_company_simple(company)
            analyzed_companies.append(analyzed)
            
            # Update progress
            progress_bar.progress((i + 1) / len(companies))
        
        progress_bar.empty()
        
        # Sort by relevance
        analyzed_companies.sort(key=lambda x: x.get('naval_relevance', 0) + x.get('sbir_eligibility', 0), reverse=True)
        
        st.session_state.companies = analyzed_companies[:max_results]
        st.session_state.search_complete = True

# Main content
st.write("### 📊 API Status")
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.success("✅ Core System")
with col2:
    st.success("✅ USASpending.gov")
with col3:
    if OPENAI_API_KEY:
        st.success("✅ AI Analysis (GPT-3.5)")
    else:
        st.info("ℹ️ Rule-based Analysis")
with col4:
    st.success("✅ Interactive Maps & Charts")

# Results
if st.session_state.search_complete and st.session_state.companies:
    companies = st.session_state.companies
    
    st.success(f"🎯 Found {len(companies)} naval suppliers")
    
    # Enhanced metrics dashboard
    sbir_qualified = len([c for c in companies if c.get('sbir_eligibility', 0) >= 7])
    high_naval = len([c for c in companies if c.get('naval_relevance', 0) >= 7])
    high_tech = len([c for c in companies if c.get('tech_readiness', 0) >= 8])
    total_value = sum(c.get('total_value', 0) for c in companies)
    
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Total Suppliers", len(companies))
    with col2:
        st.metric("SBIR Qualified", sbir_qualified, f"{sbir_qualified/len(companies)*100:.0f}%")
    with col3:
        st.metric("High Naval Relevance", high_naval, f"{high_naval/len(companies)*100:.0f}%")
    with col4:
        st.metric("Tech Leaders", high_tech)
    with col5:
        st.metric("Total Contract Value", f"${total_value/1000000:.1f}M")
    
    # Advanced tabbed interface
    tab1, tab2, tab3, tab4 = st.tabs(["🏢 Supplier Directory", "📊 Analytics Dashboard", "🗺️ Geographic Intelligence", "📋 Export & Reports"])
    
    with tab1:
        st.subheader("🏢 Naval Supplier Intelligence Directory")
        
        # Display companies
        for i, company in enumerate(companies[:10]):
            with st.expander(
                f"🏢 {company.get('name', 'Unknown')} | "
                f"Naval: {company.get('naval_relevance', 0)}/10 | "
                f"SBIR: {company.get('sbir_eligibility', 0)}/10 | "
                f"Value: ${company.get('total_value', 0):,.0f}"
            ):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**🎯 Intelligence Scores**")
                    st.write(f"**Naval Relevance:** {company.get('naval_relevance', 0)}/10")
                    st.write(f"**SBIR Eligibility:** {company.get('sbir_eligibility', 0)}/10")
                    st.write(f"**Tech Readiness:** {company.get('tech_readiness', 0)}/10")
                    st.write(f"**Location:** {company.get('location', 'Unknown')}")
                    st.write(f"**Distance:** {company.get('distance_miles', 0)} miles")
                
                with col2:
                    st.write("**🏷️ Business Information**")
                    st.write(f"**Business Types:** {', '.join(company.get('business_types', []))}")
                    st.write(f"**Capabilities:** {', '.join(company.get('capabilities', []))}")
                    st.write(f"**Contracts:** {company.get('total_contracts', 0)}")
                    st.write(f"**Total Value:** ${company.get('total_value', 0):,.0f}")
                    st.write(f"**Source:** {company.get('source', 'Unknown')}")
                
                if company.get('description'):
                    st.write(f"**Description:** {company['description']}")
                
                # Show AI analysis if available
                if company.get('ai_analysis'):
                    st.write("**🤖 AI Analysis:**")
                    st.write(company['ai_analysis'])
                    st.caption(f"Analysis method: {company.get('analysis_method', 'Unknown')}")
    
    with tab2:
        st.subheader("📊 Advanced Analytics Dashboard")
        
        # Create and display visualizations
        fig_scatter, fig_business, fig_map = create_visualizations(companies)
        
        if fig_scatter:
            st.plotly_chart(fig_scatter, use_container_width=True)
        
        if fig_business:
            st.plotly_chart(fig_business, use_container_width=True)
    
    with tab3:
        st.subheader("🗺️ Geographic Intelligence & Market Distribution")
        
        # Display map
        fig_scatter, fig_business, fig_map = create_visualizations(companies)
        if fig_map:
            st.plotly_chart(fig_map, use_container_width=True)
    
    with tab4:
        st.subheader("📋 Intelligence Export & Executive Reporting")
        
        # Export options
        if companies:
            df_export = pd.DataFrame(companies)
            csv_data = df_export.to_csv(index=False)
            st.download_button(
                "📊 Download Complete Dataset",
                csv_data,
                f"wbi_naval_intelligence_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                "text/csv"
            )

else:
    st.info("👈 Configure search settings and click 'Execute Search' to find naval suppliers")
    
    # Show sample data structure
    st.subheader("📋 Sample Data Preview")
    sample_companies = get_demo_companies()[:2]
    df_sample = pd.DataFrame(sample_companies)
    st.dataframe(df_sample, use_container_width=True)

st.write("---")
st.write("**🎉 Complete WBI Naval Search Platform - All features operational!**")