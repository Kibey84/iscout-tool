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

# Enhanced prime contractor database with POC information
PRIME_CONTRACTORS = {
    "BAE Systems": {
        "divisions": [
            {
                "name": "BAE Systems Norfolk Ship Repair",
                "location": "Norfolk, VA",
                "lat": 36.8485, "lon": -76.2859,
                "capabilities": ["Ship Repair", "Maintenance", "Modernization"],
                "poc_info": {
                    "business_development": "norfolk.shiprepair@baesystems.com",
                    "phone": "(757) 494-4000",
                    "procurement": "procurement.norfolk@baesystems.com"
                },
                "specialties": ["Surface Ship Repair", "Submarine Support", "Fleet Maintenance"]
            },
            {
                "name": "BAE Systems San Diego Ship Repair",
                "location": "San Diego, CA", 
                "lat": 32.7157, "lon": -117.1611,
                "capabilities": ["Ship Repair", "Dry Dock Services", "Combat Systems"],
                "poc_info": {
                    "business_development": "sandiego.shiprepair@baesystems.com",
                    "phone": "(619) 544-2000",
                    "procurement": "procurement.sandiego@baesystems.com"
                },
                "specialties": ["Pacific Fleet Support", "Advanced Combat Systems"]
            }
        ]
    },
    "Lockheed Martin": {
        "divisions": [
            {
                "name": "Lockheed Martin Rotary and Mission Systems",
                "location": "Moorestown, NJ",
                "lat": 39.9689, "lon": -74.9437,
                "capabilities": ["Combat Systems", "Sensors", "C4ISR"],
                "poc_info": {
                    "business_development": "rms.business@lockheedmartin.com",
                    "phone": "(856) 866-2000",
                    "procurement": "supplier.diversity@lockheedmartin.com"
                },
                "specialties": ["AEGIS Combat System", "Radar Systems", "Naval Electronics"]
            },
            {
                "name": "Lockheed Martin Maritime Systems & Sensors",
                "location": "Syracuse, NY",
                "lat": 43.0481, "lon": -76.1474,
                "capabilities": ["Sonar Systems", "Undersea Warfare", "Sensors"],
                "poc_info": {
                    "business_development": "maritime.systems@lockheedmartin.com", 
                    "phone": "(315) 456-2000",
                    "procurement": "maritime.procurement@lockheedmartin.com"
                },
                "specialties": ["Submarine Sonar", "ASW Systems", "Maritime Sensors"]
            }
        ]
    },
    "Northrop Grumman": {
        "divisions": [
            {
                "name": "Northrop Grumman Shipbuilding",
                "location": "Newport News, VA",
                "lat": 37.0871, "lon": -76.4730,
                "capabilities": ["Aircraft Carriers", "Submarines", "Nuclear Propulsion"],
                "poc_info": {
                    "business_development": "newport.news@ngc.com",
                    "phone": "(757) 380-2000", 
                    "procurement": "shipbuilding.procurement@ngc.com"
                },
                "specialties": ["Nuclear Aircraft Carriers", "Virginia Class Submarines"]
            },
            {
                "name": "Northrop Grumman Mission Systems",
                "location": "San Diego, CA",
                "lat": 32.7157, "lon": -117.1611,
                "capabilities": ["C4ISR", "Electronic Warfare", "Cyber"],
                "poc_info": {
                    "business_development": "mission.systems@ngc.com",
                    "phone": "(858) 618-4000",
                    "procurement": "ms.procurement@ngc.com" 
                },
                "specialties": ["Naval Radar", "Electronic Warfare", "Command & Control"]
            }
        ]
    },
    "General Dynamics": {
        "divisions": [
            {
                "name": "General Dynamics Electric Boat",
                "location": "Groton, CT",
                "lat": 41.3501, "lon": -72.0796,
                "capabilities": ["Submarine Design", "Nuclear Submarines", "Submarine Systems"],
                "poc_info": {
                    "business_development": "electric.boat@gd.com",
                    "phone": "(860) 433-3000",
                    "procurement": "eb.procurement@gd.com"
                },
                "specialties": ["Virginia Class", "Columbia Class", "Submarine Design"]
            },
            {
                "name": "General Dynamics Bath Iron Works", 
                "location": "Bath, ME",
                "lat": 43.9109, "lon": -69.8223,
                "capabilities": ["Destroyers", "Surface Combatants", "Ship Design"],
                "poc_info": {
                    "business_development": "bath.ironworks@gd.com",
                    "phone": "(207) 443-3311",
                    "procurement": "biw.procurement@gd.com"
                },
                "specialties": ["DDG Destroyers", "Surface Ship Design", "Combat Systems Integration"]
            }
        ]
    },
    "Huntington Ingalls Industries": {
        "divisions": [
            {
                "name": "Newport News Shipbuilding",
                "location": "Newport News, VA", 
                "lat": 37.0871, "lon": -76.4730,
                "capabilities": ["Aircraft Carriers", "Nuclear Refueling", "Ship Repair"],
                "poc_info": {
                    "business_development": "newport.news.bd@hii-co.com",
                    "phone": "(757) 380-2000",
                    "procurement": "nns.procurement@hii-co.com"
                },
                "specialties": ["Ford Class Carriers", "Nuclear Refueling", "Carrier Maintenance"]
            },
            {
                "name": "Ingalls Shipbuilding",
                "location": "Pascagoula, MS",
                "lat": 30.3658, "lon": -88.5564, 
                "capabilities": ["Destroyers", "Amphibious Ships", "Coast Guard Cutters"],
                "poc_info": {
                    "business_development": "ingalls.bd@hii-co.com",
                    "phone": "(228) 935-3000",
                    "procurement": "ingalls.procurement@hii-co.com"
                },
                "specialties": ["DDG Destroyers", "LPD Amphibious Ships", "National Security Cutters"]
            }
        ]
    },
    "Raytheon Technologies": {
        "divisions": [
            {
                "name": "Raytheon Missiles & Defense",
                "location": "Tucson, AZ",
                "lat": 32.2226, "lon": -110.9747,
                "capabilities": ["Missiles", "Defense Systems", "Radar"],
                "poc_info": {
                    "business_development": "rmd.business@rtx.com",
                    "phone": "(520) 794-1800",
                    "procurement": "rmd.procurement@rtx.com"
                },
                "specialties": ["Standard Missile", "ESSM", "Naval Radar Systems"]
            },
            {
                "name": "Raytheon Intelligence & Space",
                "location": "Aurora, CO",
                "lat": 39.7294, "lon": -104.8319,
                "capabilities": ["Intelligence Systems", "Space Systems", "Cyber"],
                "poc_info": {
                    "business_development": "ris.business@rtx.com", 
                    "phone": "(303) 344-2000",
                    "procurement": "ris.procurement@rtx.com"
                },
                "specialties": ["Naval Intelligence", "Satellite Communications", "Electronic Warfare"]
            }
        ]
    },
    "L3Harris Technologies": {
        "divisions": [
            {
                "name": "L3Harris Ocean Systems",
                "location": "Portsmouth, RI",
                "lat": 41.5976, "lon": -71.2675,
                "capabilities": ["Underwater Systems", "Sonar", "ASW"],
                "poc_info": {
                    "business_development": "ocean.systems@l3harris.com",
                    "phone": "(401) 847-8000", 
                    "procurement": "ocean.procurement@l3harris.com"
                },
                "specialties": ["Towed Array Sonar", "Submarine Detection", "ASW Systems"]
            }
        ]
    }
}

# API Configuration
GOOGLE_PLACES_API_KEY = (
    st.secrets.get("GOOGLE_PLACES_API_KEY") if "GOOGLE_PLACES_API_KEY" in st.secrets else
    os.environ.get("GOOGLE_PLACES_API_KEY", "")
)

@dataclass
class EnhancedSearchConfig:
    base_location: str = "South Bend, Indiana"
    radius_miles: int = 60
    target_company_count: int = 100
    include_prime_contractors: bool = True
    prime_contractor_radius: int = 500  # Larger radius for prime contractors
    
    # Keywords for company search
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
    
    workforce_keywords: List[str] = field(default_factory=lambda: [
        "naval training", "maritime training", "shipyard training", "welding certification",
        "maritime academy", "technical training", "apprenticeship", "workforce development",
        "skills training", "industrial training", "safety training", "crane operator training",
        "maritime safety", "naval education", "defense training", "military training",
        "shipbuilding training", "marine engineering training", "technical certification"
    ])

class EnhancedCompanySearcher:
    def __init__(self, config: EnhancedSearchConfig):
        self.config = config
        self.geolocator = Nominatim(user_agent="wbi_naval_search_enhanced")
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
        """Enhanced scoring including prime contractor bonuses"""
        description = company_data.get('description', '').lower()
        name = company_data.get('name', '').lower()
        industry = company_data.get('industry', '').lower()
        
        combined_text = f"{description} {name} {industry}"
        
        scores = {
            'manufacturing_score': 0,
            'robotics_score': 0,
            'unmanned_score': 0,
            'workforce_score': 0,
            'prime_contractor_bonus': 0,
            'total_score': 0
        }
        
        # Major prime contractor detection
        prime_contractors = [
            'bae systems', 'lockheed martin', 'northrop grumman', 'general dynamics',
            'huntington ingalls', 'raytheon', 'l3harris', 'boeing defense',
            'electric boat', 'newport news', 'bath iron works', 'ingalls shipbuilding'
        ]
        
        for contractor in prime_contractors:
            if contractor in name:
                scores['prime_contractor_bonus'] = 20  # Major bonus for prime contractors
                scores['manufacturing_score'] = 15  # Automatic high manufacturing score
                break
        
        # High-value company name keywords
        high_value_name_keywords = {
            'aerospace': 8, 'defense': 8, 'naval': 8, 'military': 7,
            'honeywell': 6, 'boeing': 6, 'systems': 5, 'technologies': 5,
            'corporation': 4, 'industries': 4, 'engineering': 4, 'solutions': 3,
            'manufacturing': 3, 'precision': 4, 'automation': 4, 'robotics': 5
        }
        
        # Standard keyword scoring (existing logic)
        manufacturing_keywords = {
            'manufacturing': 3, 'fabrication': 3, 'machining': 3, 'metal': 2,
            'cnc': 3, 'welding': 2, 'assembly': 2, 'production': 2,
            'machine': 2, 'tool': 2, 'sheet metal': 3
        }
        
        robotics_keywords = {
            'robotics': 4, 'automation': 4, 'robotic': 4, 'automated': 3,
            'fanuc': 4, 'kuka': 4, 'abb': 3, 'controls': 2
        }
        
        unmanned_keywords = {
            'unmanned': 5, 'autonomous': 5, 'uav': 5, 'drone': 4,
            'uuv': 5, 'usv': 5, 'remote': 2, 'guidance': 3
        }
        
        workforce_keywords = {
            'training': 4, 'academy': 5, 'certification': 4, 'apprenticeship': 4,
            'workforce': 3, 'education': 3, 'maritime training': 6, 'naval training': 7,
            'shipyard training': 6, 'welding certification': 5, 'safety training': 4,
            'technical training': 4, 'skills development': 3, 'crane operator': 4
        }
        
        # Apply keyword scoring
        for keyword, points in high_value_name_keywords.items():
            if keyword in name:
                scores['manufacturing_score'] += points
        
        for keyword, points in manufacturing_keywords.items():
            if keyword in combined_text:
                scores['manufacturing_score'] += points
        
        for keyword, points in robotics_keywords.items():
            if keyword in combined_text:
                scores['robotics_score'] += points
        
        for keyword, points in unmanned_keywords.items():
            if keyword in combined_text:
                scores['unmanned_score'] += points
        
        for keyword, points in workforce_keywords.items():
            if keyword in combined_text:
                scores['workforce_score'] += points
        
        scores['total_score'] = (scores['manufacturing_score'] + 
                               scores['robotics_score'] + 
                               scores['unmanned_score'] +
                               scores['workforce_score'] +
                               scores['prime_contractor_bonus'])
        
        return scores
    
    def get_prime_contractors_in_range(self) -> List[Dict]:
        """Get prime contractors within extended range"""
        prime_companies = []
        
        for contractor_name, contractor_data in PRIME_CONTRACTORS.items():
            for division in contractor_data['divisions']:
                distance = self._calculate_distance(division['lat'], division['lon'])
                
                # Use extended radius for prime contractors
                if distance <= self.config.prime_contractor_radius:
                    company = {
                        'name': division['name'],
                        'parent_company': contractor_name,
                        'location': division['location'],
                        'industry': 'Defense Contractor',
                        'description': f"Prime contractor division specializing in {', '.join(division['capabilities'])}",
                        'size': 'Large Corporation',
                        'capabilities': division['capabilities'],
                        'specialties': division['specialties'],
                        'lat': division['lat'],
                        'lon': division['lon'],
                        'distance_miles': round(distance, 1),
                        'website': f"Contact business development for details",
                        'phone': division['poc_info']['phone'],
                        'rating': 5.0,
                        'user_ratings_total': 0,
                        'is_prime_contractor': True,
                        # POC Information
                        'poc_business_development': division['poc_info']['business_development'],
                        'poc_procurement': division['poc_info']['procurement'],
                        'poc_phone': division['poc_info']['phone']
                    }
                    
                    # Add relevance scoring
                    scores = self._score_company_relevance(company)
                    company.update(scores)
                    
                    prime_companies.append(company)
        
        return prime_companies
    
    def search_companies(self) -> List[Dict]:
        """Enhanced search including prime contractors"""
        all_companies = []
        
        # Add prime contractors first
        if self.config.include_prime_contractors:
            st.info("🏢 Identifying prime contractors in extended range...")
            prime_contractors = self.get_prime_contractors_in_range()
            all_companies.extend(prime_contractors)
            st.success(f"Found {len(prime_contractors)} prime contractor divisions")
        
        # Add local companies using existing logic
        api_key = st.session_state.get('api_key', GOOGLE_PLACES_API_KEY)
        
        if api_key:
            st.info("🔍 Searching local suppliers using Google Places API...")
            local_companies = self.search_real_companies()
            all_companies.extend(local_companies)
        else:
            st.info("📋 Using demo data for local suppliers. Add API key for real search.")
            local_companies = self.generate_sample_companies()
            all_companies.extend(local_companies)
        
        # Sort by relevance score
        all_companies.sort(key=lambda x: x['total_score'], reverse=True)
        
        return all_companies[:self.config.target_company_count]
    
    def search_real_companies(self) -> List[Dict]:
        """Search for real local companies (existing logic)"""
        all_companies = []
        
        api_key = st.session_state.get('api_key', GOOGLE_PLACES_API_KEY)
        
        if not api_key:
            return []
        
        # Enhanced search queries including prime contractor searches
        search_queries = [
            "BAE Systems facility",
            "Lockheed Martin facility", 
            "Northrop Grumman facility",
            "General Dynamics facility",
            "Raytheon facility",
            "defense contractors",
            "aerospace manufacturing",
            "precision machining companies",
            "CNC machining services",
            "metal fabrication LLC",
            "machine shop",
            "custom manufacturing",
            "specialty manufacturing",
            "contract manufacturing",
            "precision manufacturing Inc",
            "maritime academy",
            "naval training center",
            "shipyard training",
            "welding certification",
            "maritime safety training",
            "technical training institute"
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
                if distance <= self.config.radius_miles and company['total_score'] >= 0:
                    company['is_prime_contractor'] = False  # Mark as local supplier
                    unique_companies.append(company)
        
        return unique_companies
    
    def search_google_places_text(self, query: str) -> List[Dict]:
        """Search Google Places using text search (existing logic)"""
        api_key = st.session_state.get('api_key', GOOGLE_PLACES_API_KEY)
        
        if not api_key:
            return []
        
        companies = []
        lat, lon = self.base_coords
        
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
                    place_lat = place.get('location', {}).get('latitude', 0)
                    place_lon = place.get('location', {}).get('longitude', 0)
                    
                    if place_lat and place_lon:
                        distance = geodesic(self.base_coords, (place_lat, place_lon)).miles
                        
                        if distance <= self.config.radius_miles:
                            name = place.get('displayName', {}).get('text', 'Unknown')
                            types = place.get('types', [])
                            
                            if self._is_manufacturing_related(name, types):
                                business_size = self._determine_business_size(name, types, place)
                                
                                company = {
                                    'name': name,
                                    'location': place.get('formattedAddress', 'Unknown'),
                                    'industry': ', '.join(types[:2]),
                                    'description': f"Business type: {', '.join(types[:2])}",
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
                
            elif response.status_code == 403:
                st.error("❌ API key doesn't have permission. Make sure 'Places API (New)' is enabled and billing is set up.")
            else:
                st.warning(f"Search for '{query}' returned status {response.status_code}")
            
        except Exception as e:
            st.warning(f"Error searching for '{query}': {str(e)}")
        
        return companies
    
    def _is_manufacturing_related(self, name: str, types: List[str]) -> bool:
        """Check if a business is manufacturing-related (existing logic)"""
        name_lower = name.lower()
        types_str = ' '.join(types).lower()
        
        exclude_keywords = [
            'mobile welding', 'roadside', 'automotive repair', 'auto repair',
            'truck repair', 'trailer repair', 'small engine', 'lawn mower'
        ]
        
        for exclude in exclude_keywords:
            if exclude in name_lower:
                return False
        
        manufacturing_keywords = [
            'manufacturing', 'fabrication', 'machining', 'industrial',
            'aerospace', 'defense', 'precision', 'cnc', 'automation', 
            'robotics', 'engineering', 'systems', 'technologies',
            'corporation', 'industries', 'solutions'
        ]
        
        training_keywords = [
            'training', 'academy', 'institute', 'education', 'certification',
            'apprenticeship', 'workforce', 'maritime', 'naval', 'shipyard',
            'technical college', 'vocational', 'skills center'
        ]
        
        business_types = [
            'manufacturer', 'contractor', 'engineering', 'technology',
            'industrial', 'aerospace', 'defense', 'school', 'university',
            'college', 'institute', 'academy', 'training_center'
        ]
        
        for keyword in manufacturing_keywords:
            if keyword in name_lower:
                return True
        
        for keyword in training_keywords:
            if keyword in name_lower:
                return True
                
        for btype in business_types:
            if btype in types_str:
                return True
        
        if 'welding' in name_lower and ('fabrication' in name_lower or 'manufacturing' in name_lower):
            return True
        
        return False
    
    def _determine_business_size(self, name: str, types: List[str], place_data: Dict) -> str:
        """Determine business size (existing logic)"""
        name_lower = name.lower()
        
        large_corp_indicators = [
            'honeywell', 'boeing', 'lockheed', 'raytheon', 'northrop', 'general dynamics',
            'bae systems', 'textron', 'collins aerospace', 'pratt whitney', 'rolls royce',
            'general electric', 'caterpillar', 'john deere', 'cummins', 'ford', 'gm'
        ]
        
        medium_indicators = [
            'corporation', 'corp', 'industries', 'international', 'group',
            'systems', 'technologies', 'holdings', 'enterprises'
        ]
        
        small_indicators = [
            'llc', 'inc', 'ltd', 'company', 'co', 'shop', 'works', 'services',
            'solutions', 'custom', 'specialty', 'precision', 'family', 'brothers'
        ]
        
        for indicator in large_corp_indicators:
            if indicator in name_lower:
                return 'Large Corporation'
        
        review_count = place_data.get('userRatingCount', 0)
        if review_count > 100:
            return 'Large Corporation'
        elif review_count > 20:
            return 'Medium Business'
        
        for indicator in medium_indicators:
            if indicator in name_lower:
                return 'Medium Business'
        
        for indicator in small_indicators:
            if indicator in name_lower:
                return 'Small Business'
        
        if review_count <= 10:
            return 'Small Business'
        
        return 'Medium Business'
    
    def _extract_capabilities_from_name_and_types(self, name: str, types: List[str]) -> List[str]:
        """Extract capabilities (existing logic)"""
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
            'assembly': 'Assembly Services',
            'training': 'Training Services',
            'academy': 'Maritime Academy',
            'certification': 'Certification Programs',
            'maritime': 'Maritime Training',
            'naval': 'Naval Training',
            'shipyard': 'Shipyard Training',
            'safety': 'Safety Training',
            'apprenticeship': 'Apprenticeship Programs'
        }
        
        for keyword, capability in capability_mapping.items():
            if keyword in name_lower:
                capabilities.append(capability)
        
        return capabilities if capabilities else ['General Services']
    
    def generate_sample_companies(self) -> List[Dict]:
        """Generate sample local companies (existing logic)"""
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
                'user_ratings_total': 23,
                'is_prime_contractor': False
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
                'user_ratings_total': 15,
                'is_prime_contractor': False
            }
        ]
        
        # Add distance and relevance scoring
        for company in sample_companies:
            distance = self._calculate_distance(company['lat'], company['lon'])
            company['distance_miles'] = round(distance, 1)
            
            scores = self._score_company_relevance(company)
            company.update(scores)
        
        return sample_companies