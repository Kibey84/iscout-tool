import os
import time
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import pandas as pd
import requests
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# Accurate distance (optional dependency)
try:
    from geopy.distance import geodesic
    GEOPY_AVAILABLE = True
except Exception:
    GEOPY_AVAILABLE = False

# -----------------------------
# API Configuration & Settings
# -----------------------------
GOOGLE_PLACES_API_KEY = os.environ.get("GOOGLE_PLACES_API_KEY", "")

# Requests defaults
HTTP_TIMEOUT = 15
USER_AGENT = "WBI-Naval-Search/1.0"
HEADERS_HTML = {"User-Agent": USER_AGENT, "Accept": "text/html,application/xhtml+xml"}

# Corporate fallback inboxes for primes (used only if local POC emails aren't found)
CORPORATE_POC_FALLBACK = {
    "lockheed": ["supplier.diversity@lockheedmartin.com"],
    "northrop": ["ngc.suppliercontact@ngc.com"],
    "raytheon": ["rmd.supplier@rtx.com", "ris.supplier@rtx.com"],
    "general dynamics": ["gdit.procurement@gdit.com"],
    "electric boat": ["supplierinfo@gdeb.com"],
    "huntington ingalls": ["nnsuppliercompliance@hii-co.com"],
    "bae systems": ["supplier.diversity@baesystems.com"],
    "l3harris": ["supplier@l3harris.com"],
    "boeing": ["supplier@boeing.com"],
    "textron": ["supplychain@textron.com"],
    "collins": ["supplier@collins.com"],
}

PRIME_BRANDS = [
    "Lockheed Martin",
    "Northrop Grumman",
    "Raytheon",
    "General Dynamics",
    "BAE Systems",
    "Huntington Ingalls",
    "L3Harris",
    "Boeing Defense",
    "Textron",
    "Collins Aerospace",
    "Electric Boat",
    "Newport News Shipbuilding",
    "Bath Iron Works",
]

# --------------------------------
# Config & Weighting / Scoring
# --------------------------------
@dataclass
class SearchConfig:
    base_location: str = "South Bend, Indiana"
    radius_miles: int = 60
    target_company_count: int = 150
    search_multi_state: bool = False  # reserved for future use
    enable_association_search: bool = False  # reserved for future use

    manufacturing_keywords: List[str] = field(default_factory=lambda: [
        "advanced manufacturing", "precision machining", "metal fabrication",
        "additive manufacturing", "3D printing", "CNC machining",
        "welding", "assembly", "fabrication", "machining", "casting",
        "forging", "sheet metal", "tool and die", "injection molding",
        "shipbuilding", "naval shipyard", "marine engineering", "hull fabrication"
    ])

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

MICROELECTRONICS_WEIGHTS = {
    "microelectronics": 25, "semiconductor": 20, "pcb manufacturing": 18,
    "electronics manufacturing": 15, "integrated circuits": 18, "ic design": 20,
    "naval electronics": 22, "maritime electronics": 20, "avionics": 15,
    "radar systems": 18, "sonar electronics": 18, "navigation electronics": 16,
    "communication systems": 14, "electronic warfare": 20, "signal processing": 16,
    "embedded systems": 14, "microprocessors": 16, "fpga": 15,
    "electronic components": 12, "circuits": 10
}

NAVAL_WEIGHTS = {
    "shipbuilding": 25, "naval shipyard": 28, "hull fabrication": 20,
    "marine engineering": 18, "submarine": 25, "naval systems": 22,
    "maritime": 15, "shipyard": 20, "vessel": 12, "marine": 12,
    "navy": 20, "naval": 18, "coast guard": 16
}

MANUFACTURING_WEIGHTS = {
    'precision machining': 12, 'cnc machining': 10, 'metal fabrication': 8,
    'welding': 6, 'manufacturing': 8, 'aerospace manufacturing': 12,
    'defense manufacturing': 10, 'additive manufacturing': 8
}

DEFENSE_WEIGHTS = {
    'defense contractor': 15, 'military contractor': 12, 'aerospace': 10,
    'defense systems': 12, 'military systems': 10, 'government contracting': 8
}

ROBOTICS_WEIGHTS = {'robotics': 10, 'automation': 8, 'robotic systems': 10, 'industrial automation': 8}
UNMANNED_WEIGHTS = {'uav': 10, 'uuv': 10, 'usv': 10, 'autonomous': 8, 'drone': 8, 'rov': 8, 'auv': 8}
WORKFORCE_WEIGHTS = {'training': 6, 'academy': 6, 'certification': 5, 'apprenticeship': 5, 'workforce': 4}

# --------------------------------
# Searcher
# --------------------------------
class EnhancedNavalSearcher:
    def __init__(self, config: SearchConfig):
        self.config = config

    # ---------- Helpers ----------
    def _distance_miles_from_base(self, lat: float, lon: float) -> float:
        base_coords = {
            'south bend': (41.6764, -86.2520),
            'norfolk': (36.8508, -76.2859),
            'san diego': (32.7157, -117.1611),
            'pearl harbor': (21.3099, -157.8581),
            'newport news': (37.0871, -76.4730),
            'bath': (43.9109, -69.8214),
            'groton': (41.3501, -72.0979),
        }
        base_lat, base_lon = 41.6764, -86.2520
        for key, coords in base_coords.items():
            if key in self.config.base_location.lower():
                base_lat, base_lon = coords
                break

        if not lat or not lon:
            return 999.0

        if GEOPY_AVAILABLE:
            return round(geodesic((base_lat, base_lon), (lat, lon)).miles, 1)
        # Fallback: crude approximation
        lat_diff = abs(lat - base_lat)
        lon_diff = abs(lon - base_lon)
        return round(((lat_diff ** 2 + lon_diff ** 2) ** 0.5) * 69, 1)

    def _is_relevant(self, name: str, types: List[str], query: str) -> bool:
        name_lower = name.lower()
        types_str = " ".join(types).lower()
        combined = f"{name_lower} {types_str}"

        # STRICT exclusions (removed school/university to allow training finds)
        hard_exclude = [
            'restaurant', 'food', 'catering', 'retail', 'store', 'bank', 'insurance',
            'real estate', 'hospital', 'hotel', 'gas station', 'pharmacy',
            'auto repair', 'automotive repair', 'hair salon', 'nail salon', 'spa',
            'cleaning services', 'janitorial', 'landscaping', 'roofing', 'flooring',
            'residential construction', 'home improvement', 'interior design'
        ]
        if any(ex in combined for ex in hard_exclude):
            return False

        # If the query itself is domain-specific, allow pass-through (scoring will demote junk).
        intent_terms = ['semiconductor', 'microelectronics', 'pcb', 'ship', 'naval', 'marine',
                        'defense', 'aerospace', 'cnc', 'machining', 'fabrication', 'automation',
                        'robotics', 'training', 'supplier', 'office', 'facility']
        if any(t in query.lower() for t in intent_terms):
            return True

        # Otherwise require at least a relevant indicator
        required_indicators = [
            'manufacturing', 'machining', 'fabrication', 'welding', 'casting',
            'forging', 'cnc', 'precision', 'metal', 'steel', 'aluminum',
            'assembly', 'production', 'shipyard', 'shipbuilding', 'marine', 'naval',
            'maritime', 'submarine', 'vessel', 'boat', 'hull', 'electronics',
            'microelectronics', 'semiconductor', 'pcb', 'circuits', 'radar', 'sonar',
            'aerospace', 'defense contractor', 'military contractor', 'avionics',
            'defense systems', 'robotics', 'automation', 'controls', 'systems integration',
            'maritime academy', 'naval training', 'shipyard training', 'welding certification',
            'supplier', 'office'
        ]
        return any(ind in combined for ind in required_indicators)

    def _business_size(self, name: str, place: Dict) -> str:
        name_lower = name.lower()
        review_count = place.get('userRatingCount', 0)
        rating = place.get('rating', 0)

        major_corps = [
            'honeywell', 'boeing', 'lockheed', 'lockheed martin', 'raytheon', 'northrop',
            'general dynamics', 'bae systems', 'textron', 'collins', 'pratt', 'rolls',
            'huntington ingalls', 'newport news shipbuilding', 'bath iron works',
            'electric boat', 'l3harris', 'intel', 'texas instruments'
        ]
        if any(c in name_lower for c in major_corps):
            return 'Fortune 500 / Major Corporation'

        if review_count > 200 or (review_count > 100 and rating > 4.0):
            return 'Large Corporation'
        if review_count > 50 or (review_count > 25 and rating > 4.2):
            return 'Medium Business'
        if review_count > 10:
            return 'Small-Medium Business'
        return 'Small Business'

    def _capabilities(self, name: str, types: List[str]) -> List[str]:
        name_lower = name.lower()
        types_str = ' '.join(types).lower()
        combined = f"{name_lower} {types_str}"
        out = set()
        mapping = {
            'cnc': 'CNC Machining', 'machining': 'Precision Machining', 'welding': 'Welding Services',
            'fabrication': 'Metal Fabrication', 'manufacturing': 'Manufacturing', 'casting': 'Metal Casting',
            'forging': 'Metal Forging', 'sheet metal': 'Sheet Metal Work', 'additive': '3D Printing/Additive Manufacturing',
            '3d printing': '3D Printing/Additive Manufacturing', 'microelectronics': 'Microelectronics Manufacturing',
            'semiconductor': 'Semiconductor Design/Fab', 'pcb': 'PCB Manufacturing', 'electronics': 'Electronics Manufacturing',
            'electronic components': 'Electronic Components', 'circuits': 'Circuit Design/Manufacturing',
            'radar': 'Radar Systems', 'sonar': 'Sonar Systems', 'avionics': 'Avionics Systems',
            'automation': 'Industrial Automation', 'robotics': 'Robotics Integration', 'robotic': 'Robotic Systems',
            'vision': 'Machine Vision Systems', 'control': 'Process Control Systems',
            'shipyard': 'Shipbuilding', 'shipbuilding': 'Shipbuilding', 'marine': 'Marine Engineering',
            'naval': 'Naval Systems', 'maritime': 'Maritime Systems', 'submarine': 'Submarine Systems',
            'hull': 'Hull Fabrication', 'aerospace': 'Aerospace Manufacturing', 'defense': 'Defense Systems',
            'military': 'Military Systems', 'training': 'Training Services', 'academy': 'Educational Academy',
            'certification': 'Certification Programs', 'apprenticeship': 'Apprenticeship Programs',
            'office': 'Prime Contractor Office'
        }
        for k, v in mapping.items():
            if k in combined:
                out.add(v)
        return list(out) if out else ['General Manufacturing']

    def _score_company(self, company: Dict) -> Dict:
        name = company['name'].lower()
        industry = company['industry'].lower()
        description = company['description'].lower()
        capabilities = ' '.join(company['capabilities']).lower()
        combined = f"{name} {industry} {description} {capabilities}"

        scores = {
            'manufacturing_score': 0.0,
            'microelectronics_score': 0.0,
            'robotics_score': 0.0,
            'unmanned_score': 0.0,
            'workforce_score': 0.0,
            'defense_score': 0.0,
            'naval_score': 0.0,
            'total_score': 0.0
        }

        for k, w in MICROELECTRONICS_WEIGHTS.items():
            if k in combined: scores['microelectronics_score'] += w
        for k, w in NAVAL_WEIGHTS.items():
            if k in combined: scores['naval_score'] += w
        for k, w in MANUFACTURING_WEIGHTS.items():
            if k in combined: scores['manufacturing_score'] += w
        for k, w in DEFENSE_WEIGHTS.items():
            if k in combined: scores['defense_score'] += w
        for k, w in ROBOTICS_WEIGHTS.items():
            if k in combined: scores['robotics_score'] += w
        for k, w in UNMANNED_WEIGHTS.items():
            if k in combined: scores['unmanned_score'] += w
        for k, w in WORKFORCE_WEIGHTS.items():
            if k in combined: scores['workforce_score'] += w

        rating = company.get('rating', 0)
        reviews = company.get('user_ratings_total', 0)
        quality = 5 if (rating >= 4.5 and reviews >= 10) else 3 if (rating >= 4.0 and reviews >= 5) else 0

        scores['total_score'] = (
            scores['manufacturing_score'] * 1.0 +
            scores['microelectronics_score'] * 2.0 +
            scores['naval_score'] * 1.5 +
            scores['defense_score'] * 1.2 +
            scores['robotics_score'] * 0.8 +
            scores['unmanned_score'] * 0.9 +
            scores['workforce_score'] * 0.7 +
            quality
        )
        return scores

    # ---------- Google Places ----------
    def _cached_places_search(self, body: Dict, headers: Dict) -> Dict:
        # (No caching decorator for older Streamlit)
        url = "https://places.googleapis.com/v1/places:searchText"
        try:
            r = requests.post(url, headers=headers, json=body, timeout=HTTP_TIMEOUT)
            r.raise_for_status()
            return r.json()
        except Exception:
            return {}

    def _search_google_places(self, query: str, api_key: str) -> List[Dict]:
        headers = {
            'Content-Type': 'application/json',
            'X-Goog-Api-Key': api_key,
            'X-Goog-FieldMask': (
                'places.displayName,places.formattedAddress,places.location,'
                'places.types,places.websiteUri,places.nationalPhoneNumber,'
                'places.rating,places.userRatingCount,places.businessStatus'
            )
        }

        results: List[Dict] = []
        body: Dict = {"textQuery": query, "maxResultCount": 20}

        page = 0
        while True:
            data = self._cached_places_search(body, headers)
            places = data.get('places', []) or []
            results.extend(self._process_places_response(places, query))

            # Pagination token (field name can vary)
            next_token = data.get('nextPageToken') or data.get('next_page_token') or None
            if not next_token or page >= 2:
                break
            page += 1
            time.sleep(1.0)  # brief wait for token validity
            body["pageToken"] = next_token

        return results

    def _process_places_response(self, places: List[Dict], query: str) -> List[Dict]:
        companies: List[Dict] = []
        for place in places:
            try:
                name = place.get('displayName', {}).get('text', 'Unknown')
                types = place.get('types', []) or []
                if not self._is_relevant(name, types, query):
                    continue

                lat = place.get('location', {}).get('latitude', 0)
                lon = place.get('location', {}).get('longitude', 0)
                distance = self._distance_miles_from_base(lat, lon)

                company = {
                    'name': name,
                    'location': place.get('formattedAddress', 'Unknown'),
                    'industry': ', '.join(types[:3]) if types else 'Manufacturing/Office',
                    'description': self._description_from_query(name, types, query),
                    'size': self._business_size(name, place),
                    'capabilities': self._capabilities(name, types),
                    'lat': lat,
                    'lon': lon,
                    'website': place.get('websiteUri', 'Not available'),
                    'phone': place.get('nationalPhoneNumber', 'Not available'),
                    'rating': place.get('rating', 0),
                    'user_ratings_total': place.get('userRatingCount', 0),
                    'types': types,
                    'distance_miles': distance,
                    'is_prime_office': self._seems_prime_office(name)
                }
                companies.append(company)
            except Exception:
                continue
        return companies

    def _description_from_query(self, name: str, types: List[str], query: str) -> str:
        base = f"Business type: {', '.join(types[:2]) if types else 'Manufacturing/Office'}"
        q = query.lower()
        if 'microelectronics' in q or 'semiconductor' in q: base += " · Microelectronics/Semiconductor"
        elif 'electronics' in q: base += " · Electronics"
        if 'naval' in q or 'marine' in q: base += " · Naval/Maritime"
        if 'aerospace' in q or 'defense' in q: base += " · Aerospace/Defense"
        if 'shipbuilding' in q or 'shipyard' in q: base += " · Shipbuilding"
        if 'automation' in q or 'robotics' in q: base += " · Automation/Robotics"
        if 'office' in q: base += " · Prime Contractor Office"
        return base

    def _seems_prime_office(self, name: str) -> bool:
        n = name.lower()
        return any(brand.lower() in n for brand in PRIME_BRANDS)

    # ---------- Prime office search ----------
    def _prime_office_queries(self, base_location: str) -> List[str]:
        q = []
        for brand in PRIME_BRANDS:
            q += [
                f"{brand} office near {base_location}",
                f"{brand} facility near {base_location}",
                f"{brand} supplier office near {base_location}",
            ]
        return q

    # ---------- POC Enrichment ----------
    def _fetch_html(self, url: str) -> str:
        # (No caching decorator for older Streamlit)
        try:
            r = requests.get(url, headers=HEADERS_HTML, timeout=HTTP_TIMEOUT)
            if r.status_code == 200 and "text" in r.headers.get("Content-Type", ""):
                return r.text
        except Exception:
            pass
        return ""

    def _candidate_contact_paths(self, base_url: str) -> List[str]:
        paths = ["", "/contact", "/contact-us", "/contacts", "/suppliers", "/supply-chain", "/supplier", "/doing-business", "/about/contact"]
        out = []
        for p in paths:
            if base_url.endswith("/") and p.startswith("/"):
                out.append(base_url[:-1] + p)
            elif not base_url.endswith("/") and not p.startswith("/"):
                out.append(base_url + "/" + p)
            else:
                out.append(base_url + p)
        return list(dict.fromkeys(out))

    def _extract_emails(self, html: str) -> List[str]:
        raw = set(re.findall(r'[\w\.-]+@[\w\.-]+\.\w+', html, flags=re.IGNORECASE))
        junk = {"example.com", "email.com", "domain.com"}
        return sorted(e for e in raw if not any(j in e.lower() for j in junk))

    def _filter_poc_emails(self, emails: List[str]) -> Tuple[List[str], List[str]]:
        bd_keywords = ["bd", "business", "development", "growth"]
        subk_keywords = ["subcontract", "subcontracts", "subk", "scm", "supply", "supplier", "procure", "purchasing"]
        prime_like, other = [], []
        for e in emails:
            el = e.lower()
            if any(k in el for k in bd_keywords) or any(k in el for k in subk_keywords):
                prime_like.append(e)
            else:
                other.append(e)
        return (sorted(set(prime_like)), sorted(set(other)))

    def _brand_from_name(self, name: str) -> Optional[str]:
        n = name.lower()
        for b in CORPORATE_POC_FALLBACK.keys():
            if b in n:
                return b
        for b in [x.lower() for x in PRIME_BRANDS]:
            if b.split()[0] in n:
                return b
        return None

    def enrich_pocs_for_company(self, company: Dict) -> Dict:
        poc_emails: List[str] = []
        poc_emails_secondary: List[str] = []

        website = company.get("website", "")
        if website and website != "Not available" and website.startswith("http"):
            for url in self._candidate_contact_paths(website):
                html = self._fetch_html(url)
                if not html:
                    continue
                emails = self._extract_emails(html)
                if emails:
                    pri, sec = self._filter_poc_emails(emails)
                    poc_emails.extend(pri)
                    poc_emails_secondary.extend(sec)
                if len(poc_emails) >= 2:
                    break

        if not poc_emails:
            brand_key = self._brand_from_name(company['name']) or self._brand_from_name(company.get('industry', '')) \
                        or self._brand_from_name(company.get('description', '')) or ""
            for key, inboxes in CORPORATE_POC_FALLBACK.items():
                if key in brand_key:
                    poc_emails_secondary.extend([f"{e} (corporate fallback)" for e in inboxes])
                    break

        if poc_emails:
            company['poc_emails'] = sorted(list(set(poc_emails)))
        if poc_emails_secondary:
            company['poc_emails_backup'] = sorted(list(set(poc_emails_secondary)))
        return company

    # ---------- Public search ----------
    def search_companies(self) -> List[Dict]:
        api_key = st.session_state.get('api_key', GOOGLE_PLACES_API_KEY)
        if not api_key:
            st.error("🔑 Google Places API key is REQUIRED for search.")
            return []

        base_location = self.config.base_location

        # Build queries (domain + primes)
        queries = self._enhanced_queries(base_location) + self._prime_office_queries(base_location)

        all_hits: List[Dict] = []
        progress = st.progress(0.0)
        for i, q in enumerate(queries):
            st.write(f"Searching: {q}")
            all_hits.extend(self._search_google_places(q, api_key))
            progress.progress((i + 1) / len(queries))
            time.sleep(0.6)  # gentle pacing
        progress.empty()

        # Post-process: dedupe and score
        unique: List[Dict] = []
        seen = set()
        for c in all_hits:
            bucket = (round(c.get('lat', 0.0), 3), round(c.get('lon', 0.0), 3))
            key = (c['name'].strip().lower(), bucket)
            if key in seen:
                continue
            seen.add(key)

            if c['distance_miles'] <= self.config.radius_miles:
                scores = self._score_company(c)
                c.update(scores)
                unique.append(c)

        # Prime office POC enrichment
        primes = [x for x in unique if x.get('is_prime_office')]
        if primes:
            with st.spinner(f"Enriching POCs for {len(primes)} prime offices..."):
                for p in primes:
                    self.enrich_pocs_for_company(p)

        unique.sort(key=lambda x: x['total_score'], reverse=True)
        return unique[: self.config.target_company_count]

    def _enhanced_queries(self, base_location: str) -> List[str]:
        return [
            f"microelectronics manufacturing near {base_location}",
            f"semiconductor companies near {base_location}",
            f"electronics manufacturing near {base_location}",
            f"PCB manufacturing near {base_location}",
            f"naval electronics near {base_location}",
            f"radar systems near {base_location}",
            f"sonar electronics near {base_location}",
            f"electronic components near {base_location}",
            f"shipbuilding company near {base_location}",
            f"marine engineering near {base_location}",
            f"naval architecture near {base_location}",
            f"hull fabrication near {base_location}",
            f"shipyard services near {base_location}",
            f"submarine systems near {base_location}",
            f"defense contractor manufacturing near {base_location}",
            f"aerospace manufacturing near {base_location}",
            f"military equipment manufacturer near {base_location}",
            f"naval systems manufacturer near {base_location}",
            f"CNC machining services near {base_location}",
            f"precision machining near {base_location}",
            f"metal fabrication shop near {base_location}",
            f"custom manufacturing near {base_location}",
            f"contract manufacturing near {base_location}",
            f"welding fabrication near {base_location}",
            f"additive manufacturing near {base_location}",
            f"robotics manufacturer near {base_location}",
            f"automation systems near {base_location}",
            f"control systems manufacturer near {base_location}",
            f"maritime training facility near {base_location}",
            f"welding school near {base_location}",
            f"technical training institute near {base_location}",
            f"electronics training near {base_location}",
        ]


# --------------------------------
# Visualization & Reporting
# --------------------------------
def create_company_map(companies: List[Dict], base_location: str):
    if not companies:
        return None
    df = pd.DataFrame(companies)
    df = df[(df['lat'] != 0) & (df['lon'] != 0)]
    if df.empty:
        return None

    base_coords = {
        'south bend': (41.6764, -86.2520),
        'norfolk': (36.8508, -76.2859),
        'san diego': (32.7157, -117.1611),
        'pearl harbor': (21.3099, -157.8581),
        'newport news': (37.0871, -76.4730),
        'bath': (43.9109, -69.8214),
        'groton': (41.3501, -72.0979)
    }
    base_lat, base_lon = 41.6764, -86.2520
    for key, coords in base_coords.items():
        if key in base_location.lower():
            base_lat, base_lon = coords
            break

    fig = go.Figure()
    fig.add_trace(go.Scattermapbox(
        lat=[base_lat], lon=[base_lon],
        mode='markers',
        marker=dict(size=20, color='blue'),
        text=[f'Search Center: {base_location}'],
        name='Base Location'
    ))

    color_vals = df['total_score']
    hover_text = []
    for _, row in df.iterrows():
        lines = [
            f"{row['name']}",
            f"Score: {row['total_score']:.1f} | Micro: {row['microelectronics_score']:.1f} | Naval: {row['naval_score']:.1f}",
            f"Distance: {row['distance_miles']} mi",
        ]
        if bool(row.get('is_prime_office', False)):
            lines.append("Prime Office: ✅")
        if isinstance(row.get('poc_emails'), list) and row['poc_emails']:
            lines.append("POC: " + ", ".join(row['poc_emails'][:2]))
        hover_text.append("<br>".join(lines))

    fig.add_trace(go.Scattermapbox(
        lat=df['lat'], lon=df['lon'], mode='markers',
        marker=dict(size=12, color=color_vals, colorscale='Viridis', showscale=True,
                    colorbar=dict(title="Enhanced Score")),
        text=hover_text, name='Companies'
    ))

    fig.update_layout(
        mapbox=dict(style='open-street-map', center=dict(lat=base_lat, lon=base_lon), zoom=8),
        height=600, title="Enhanced Naval Supplier & Prime Office Map"
    )
    return fig


def metrics_dashboard(companies: List[Dict]) -> str:
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
    prime_offices = len(df[df['is_prime_office'] == True])

    return f"""
    <div class="metric-grid">
        <div class="metric-card"><p class="metric-value">{total_companies}</p><p class="metric-label">Total Entities</p></div>
        <div class="metric-card"><p class="metric-value">{prime_offices}</p><p class="metric-label">Prime Offices</p></div>
        <div class="metric-card"><p class="metric-value">{microelectronics_companies}</p><p class="metric-label">Microelectronics</p></div>
        <div class="metric-card"><p class="metric-value">{naval_focused}</p><p class="metric-label">Naval/Maritime</p></div>
        <div class="metric-card"><p class="metric-value">{defense_contractors}</p><p class="metric-label">Defense Contractors</p></div>
        <div class="metric-card"><p class="metric-value">{high_relevance}</p><p class="metric-label">High Relevance</p></div>
        <div class="metric-card"><p class="metric-value">{small_businesses}</p><p class="metric-label">Small Businesses</p></div>
        <div class="metric-card"><p class="metric-value">{avg_distance:.1f} mi</p><p class="metric-label">Avg Distance</p></div>
        <div class="metric-card"><p class="metric-value">{quality_suppliers}</p><p class="metric-label">Quality Suppliers</p></div>
    </div>
    """


def executive_report(companies: List[Dict], config: SearchConfig) -> str:
    if not companies:
        return "No companies found for analysis."
    df = pd.DataFrame(companies)
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    total = len(companies)
    micro = len(df[df['microelectronics_score'] >= 10])
    naval = len(df[df['naval_score'] >= 10])
    high = len(df[df['total_score'] >= 15])

    top_overall = df.nlargest(3, 'total_score')
    top_micro = df.nlargest(3, 'microelectronics_score')
    top_naval = df.nlargest(3, 'naval_score')

    rep = f"""
# 🎯 WBI Naval Search — Enhanced Intelligence Report

**Generated:** {now}  
**Search Location:** {config.base_location}  
**Search Radius:** {config.radius_miles} miles  

## 📊 Executive Summary
- Entities evaluated: **{total}**
- Microelectronics-capable: **{micro}**
- Naval/Maritime focus: **{naval}**
- High-value overall: **{high}**
- Avg distance: **{df['distance_miles'].mean():.1f} mi**

## 🏆 Top Overall
"""
    for i, (_, row) in enumerate(top_overall.iterrows(), start=1):
        rep += f"\n**{i}. {row['name']}** — Score: {row['total_score']:.1f}, Micro: {row['microelectronics_score']:.1f}, Naval: {row['naval_score']:.1f}\n"

    if not top_micro.empty:
        rep += "\n## 🔬 Microelectronics Leaders\n"
        for i, (_, row) in enumerate(top_micro.iterrows(), start=1):
            rep += f"\n**{i}. {row['name']}** — Micro Score: {row['microelectronics_score']:.1f} · Capabilities: {', '.join(row['capabilities'][:4])}\n"

    if not top_naval.empty:
        rep += "\n## ⚓ Naval/Maritime Specialists\n"
        for i, (_, row) in enumerate(top_naval.iterrows(), start=1):
            rep += f"\n**{i}. {row['name']}** — Naval Score: {row['naval_score']:.1f} · Capabilities: {', '.join(row['capabilities'][:4])}\n"

    return rep

# --------------------------------
# Streamlit UI
# --------------------------------
def main():
    st.set_page_config(page_title="WBI Naval Search - Enhanced Supplier Intelligence", page_icon="⚓", layout="wide")

    st.title("⚓ Naval Search Pro — Enhanced (with Prime Offices & POCs)")

    # Session state
    if 'search_triggered' not in st.session_state:
        st.session_state.search_triggered = False
    if 'companies' not in st.session_state:
        st.session_state.companies = []

    # Sidebar
    st.sidebar.header("🔧 Search Configuration")
    if not GOOGLE_PLACES_API_KEY:
        st.sidebar.warning("No GOOGLE_PLACES_API_KEY in env.")
        api_key_input = st.sidebar.text_input("Google Places API Key", type="password")
        if api_key_input:
            st.session_state.api_key = api_key_input
            st.sidebar.success("API key set for this session.")
    else:
        st.sidebar.success("Env API key detected.")

    config = SearchConfig()

    presets = {
        "South Bend, Indiana (Naval Microelectronics Center)": "South Bend, Indiana",
        "Norfolk, Virginia (Naval Station Norfolk)": "Norfolk, Virginia",
        "San Diego, California (Naval Base San Diego)": "San Diego, California",
        "Pearl Harbor, Hawaii (Joint Base Pearl Harbor)": "Pearl Harbor, Hawaii",
        "Newport News, Virginia (Newport News Shipbuilding)": "Newport News, Virginia",
        "Bath, Maine (Bath Iron Works)": "Bath, Maine",
        "Groton, Connecticut (Electric Boat)": "Groton, Connecticut",
    }
    selected = st.sidebar.selectbox("Naval hub", list(presets.keys()))
    config.base_location = presets[selected]

    config.radius_miles = st.sidebar.slider("Search Radius (miles)", 10, 150, 60)
    config.target_company_count = st.sidebar.slider("Max Entities", 10, 400, 200)

    if st.sidebar.button("🔍 Run Enhanced Search", type="primary"):
        st.session_state.search_triggered = True
        st.session_state.companies = []

    # Execute search
    if st.session_state.get('search_triggered'):
        with st.spinner("Searching Google Places, including prime offices, and enriching POCs..."):
            searcher = EnhancedNavalSearcher(config)
            companies = searcher.search_companies()
            st.session_state.companies = companies
        st.session_state.search_triggered = False

    companies = st.session_state.get('companies', [])
    if not companies:
        st.info("No results yet. Configure options and run the enhanced search.")
        return

    # Metrics
    st.subheader("📊 Intelligence Dashboard")
    st.markdown("""
    <style>
    .metric-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:1rem;margin:1rem 0}
    .metric-card{background:#111827;border:1px solid #374151;border-radius:.75rem;padding:1rem;text-align:center}
    .metric-value{font-size:1.6rem;font-weight:700;color:#fff;margin:0}
    .metric-label{font-size:.75rem;color:#cbd5e0;text-transform:uppercase;letter-spacing:.05em;margin:0.25rem 0 0}
    </style>
    """, unsafe_allow_html=True)
    st.markdown(metrics_dashboard(companies), unsafe_allow_html=True)

    tab1, tab2, tab3, tab4 = st.tabs(["📋 Directory", "🗺️ Map", "📊 Analytics", "📄 Export"])

    with tab1:
        st.subheader("🏭 Supplier & Prime Office Directory")

        c1, c2, c3, c4, c5 = st.columns(5)
        with c1:
            min_total = st.slider("Min Total Score", 0, 60, 10)
        with c2:
            min_micro = st.slider("Min Micro Score", 0, 40, 0)
        with c3:
            min_naval = st.slider("Min Naval Score", 0, 40, 0)
        with c4:
            size_filter = st.selectbox("Company Size", ["All", "Small Business", "Small-Medium Business", "Medium Business", "Large Corporation", "Fortune 500 / Major Corporation"])
        with c5:
            show_primes_only = st.checkbox("Only Prime Offices", value=False)

        filtered = [c for c in companies if c['total_score'] >= min_total and
                    c['microelectronics_score'] >= min_micro and
                    c['naval_score'] >= min_naval]
        if size_filter != "All":
            filtered = [c for c in filtered if c['size'] == size_filter]
        if show_primes_only:
            filtered = [c for c in filtered if c.get('is_prime_office')]

        st.info(f"Showing {len(filtered)} of {len(companies)} entities")
        for c in filtered:
            badge = "🔥 Exceptional" if c['total_score'] >= 25 else "🟢 High" if c['total_score'] >= 15 else "🟡 Good" if c['total_score'] >= 10 else "🔴 Basic"
            title = f"{'🏛️ PRIME' if c.get('is_prime_office') else '🏭'} {c['name']} — {badge} (Score: {c['total_score']:.1f})"
            with st.expander(title):
                m1, m2, m3, m4, m5 = st.columns(5)
                m1.metric("Total", f"{c['total_score']:.1f}")
                m2.metric("Micro", f"{c['microelectronics_score']:.1f}")
                m3.metric("Naval", f"{c['naval_score']:.1f}")
                m4.metric("Defense", f"{c['defense_score']:.1f}")
                m5.metric("Mfg", f"{c['manufacturing_score']:.1f}")

                a, b = st.columns(2)
                with a:
                    st.markdown("**Company Info**")
                    st.write(f"📍 {c['location']}")
                    st.write(f"🌐 {c['website']}")
                    st.write(f"📞 {c['phone']}")
                    st.write(f"🏢 Size: {c['size']}")
                    st.write(f"📏 Distance: {c['distance_miles']} mi")
                with b:
                    st.markdown("**Capabilities**")
                    for cap in c['capabilities']:
                        st.write(f"• {cap}")
                    st.markdown("**Quality**")
                    st.write(f"⭐ {c['rating']:.1f} ({c['user_ratings_total']} reviews)")

                st.markdown(f"**Description:** {c['description']}")

                poc_primary = c.get('poc_emails', [])
                poc_backup = c.get('poc_emails_backup', [])
                if c.get('is_prime_office') or poc_primary or poc_backup:
                    st.markdown("**📇 Points of Contact**")
                    if poc_primary:
                        st.write("Primary:", ", ".join(poc_primary))
                    if poc_backup:
                        st.write("Backup:", ", ".join(poc_backup) if isinstance(poc_backup, list) else poc_backup)

    with tab2:
        st.subheader("🗺️ Map")
        fig = create_company_map(companies, config.base_location)
        if fig:
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No mappable coordinates found.")

    with tab3:
        st.subheader("📊 Analytics")
        df = pd.DataFrame(companies)
        left, right = st.columns(2)

        with left:
            st.plotly_chart(px.histogram(df, x='total_score', title='Enhanced Score Distribution', nbins=20), use_container_width=True)
            st.plotly_chart(px.scatter(df, x='microelectronics_score', y='naval_score',
                                       hover_name='name', title='Microelectronics vs Naval',
                                       size='total_score'), use_container_width=True)
        with right:
            cats = ['manufacturing_score', 'microelectronics_score', 'naval_score', 'defense_score']
            vals = [df[c].mean() for c in cats]
            st.plotly_chart(px.bar(x=['Manufacturing','Microelectronics','Naval','Defense'], y=vals, title='Average Scores by Category'),
                            use_container_width=True)

            caps = []
            for row in df['capabilities']:
                caps.extend(row)
            if caps:
                cap_counts = pd.Series(caps).value_counts().head(12)
                st.plotly_chart(px.bar(x=cap_counts.values, y=cap_counts.index, title='Top Capabilities', orientation='h'),
                                use_container_width=True)

    with tab4:
        st.subheader("📄 Export")
        df = pd.DataFrame(companies).copy()
        export_df = df[['name', 'location', 'distance_miles', 'size', 'industry',
                        'total_score', 'microelectronics_score', 'naval_score', 'defense_score',
                        'manufacturing_score', 'rating', 'user_ratings_total', 'phone', 'website',
                        'is_prime_office', 'poc_emails', 'poc_emails_backup']]

        export_df.columns = ['Company Name', 'Location', 'Distance (Miles)', 'Company Size', 'Industry',
                             'Total Score', 'Microelectronics Score', 'Naval Score', 'Defense Score',
                             'Manufacturing Score', 'Rating', 'Review Count', 'Phone', 'Website',
                             'Prime Office', 'POC Emails', 'POC Emails (Backup)']

        export_df['Search Location'] = config.base_location
        export_df['Search Date'] = datetime.now().strftime("%Y-%m-%d")

        st.dataframe(export_df, use_container_width=True, height=420)
        csv = export_df.to_csv(index=False)
        excel_csv = '\ufeff' + csv
        c1, c2 = st.columns(2)
        with c1:
            st.download_button("📥 Download CSV", data=csv, file_name=f"naval_companies_{datetime.now().strftime('%Y%m%d_%H%M')}.csv", mime="text/csv")
        with c2:
            st.download_button("📊 Download for Excel", data=excel_csv, file_name=f"naval_companies_{datetime.now().strftime('%Y%m%d_%H%M')}_excel.csv", mime="text/csv")

if __name__ == "__main__":
    main()
