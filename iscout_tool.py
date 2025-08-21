import streamlit as st
import pandas as pd
import requests
from datetime import datetime

# ULTRA MINIMAL - Just the essentials that we know work
st.set_page_config(page_title="WBI Naval Search", page_icon="⚓")

# Simple session state
if 'companies' not in st.session_state:
    st.session_state.companies = []

# API Keys from environment only
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY", "") if hasattr(st, 'secrets') else ""

st.title("⚓ WBI Naval Search - Minimal Working Version")
st.write("**Boot-safe naval supplier intelligence platform**")

# Simple demo data
def get_demo_companies():
    return [
        {
            'name': 'Advanced Naval Systems LLC',
            'location': 'Norfolk, Virginia',
            'business_types': ['Small Business', 'SDVOSB'],
            'naval_relevance': 9,
            'sbir_eligibility': 8,
            'total_contracts': 15,
            'total_value': 25000000,
            'source': 'Demo Data'
        },
        {
            'name': 'Maritime Electronics Corp',
            'location': 'Newport News, Virginia',
            'business_types': ['Small Business', 'WOSB'],
            'naval_relevance': 8,
            'sbir_eligibility': 9,
            'total_contracts': 8,
            'total_value': 12000000,
            'source': 'Demo Data'
        },
        {
            'name': 'Coastal Defense Systems',
            'location': 'Bath, Maine',
            'business_types': ['Small Business', 'SBIR Recipient'],
            'naval_relevance': 7,
            'sbir_eligibility': 9,
            'total_contracts': 5,
            'total_value': 8000000,
            'source': 'Demo Data'
        }
    ]

# Simple analysis
def analyze_company_simple(company):
    # Keep existing scores or add simple defaults
    if 'naval_relevance' not in company:
        company['naval_relevance'] = 6
    if 'sbir_eligibility' not in company:
        company['sbir_eligibility'] = 7
    if 'tech_readiness' not in company:
        company['tech_readiness'] = 6
    return company

# Sidebar
with st.sidebar:
    st.header("🎯 Search Settings")
    
    location = st.selectbox("Region:", [
        "Norfolk, Virginia",
        "Newport News, Virginia",
        "San Diego, California"
    ])
    
    max_results = st.slider("Max Results:", 5, 20, 10)
    
    if st.button("🚀 Search", type="primary"):
        # Get demo companies
        companies = get_demo_companies()
        
        # Simple analysis
        for company in companies:
            analyze_company_simple(company)
        
        # Sort by naval relevance
        companies.sort(key=lambda x: x.get('naval_relevance', 0), reverse=True)
        
        st.session_state.companies = companies[:max_results]

# Results
companies = st.session_state.companies

if companies:
    st.success(f"🎯 Found {len(companies)} naval suppliers")
    
    # Simple metrics
    sbir_count = len([c for c in companies if c.get('sbir_eligibility', 0) >= 7])
    naval_count = len([c for c in companies if c.get('naval_relevance', 0) >= 7])
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total", len(companies))
    with col2:
        st.metric("SBIR Eligible", sbir_count)
    with col3:
        st.metric("High Naval Score", naval_count)
    
    # Company list
    st.subheader("🏢 Supplier Directory")
    
    for company in companies:
        with st.expander(f"🏢 {company['name']} | Naval: {company['naval_relevance']}/10"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.write(f"**Naval Score:** {company['naval_relevance']}/10")
                st.write(f"**SBIR Score:** {company['sbir_eligibility']}/10")
                st.write(f"**Location:** {company['location']}")
            
            with col2:
                st.write(f"**Business Types:** {', '.join(company['business_types'])}")
                st.write(f"**Contracts:** {company['total_contracts']}")
                st.write(f"**Value:** ${company['total_value']:,.0f}")
    
    # Simple download
    if st.button("📥 Download CSV"):
        df = pd.DataFrame(companies)
        csv = df.to_csv(index=False)
        st.download_button(
            "Download",
            csv,
            f"naval_suppliers_{datetime.now().strftime('%Y%m%d')}.csv",
            "text/csv"
        )

else:
    st.info("👈 Click 'Search' to find naval suppliers")
    st.write("**Demo Features:**")
    st.write("- SBIR-eligible small businesses")
    st.write("- Naval relevance scoring")
    st.write("- Government contract history")

st.write("---")
st.write("**Status:** ✅ Minimal version working - ready to add advanced features")