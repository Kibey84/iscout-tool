# app.py  — Enhanced Naval Search (Option A: no cache_data) with UI polish, progress bar, and robust DataFrames

import os
import re
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import pandas as pd
import requests
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# Optional: accurate distance if geopy is installed
try:
    from geopy.distance import geodesic
    GEOPY_AVAILABLE = True
except Exception:
    GEOPY_AVAILABLE = False

# -----------------------------
# Config
# -----------------------------
st.set_page_config(page_title="WBI Naval Search - Enhanced Supplier Intelligence", page_icon="⚓", layout="wide")

GOOGLE_PLACES_API_KEY = os.environ.get("GOOGLE_PLACES_API_KEY", "")
HTTP_TIMEOUT = 15
USER_AGENT = "WBI-Naval-Search/1.0"
HEADERS_HTML = {"User-Agent": USER_AGENT, "Accept": "text/html,application/xhtml+xml"}

# Corporate fallbacks for primes — only used if local POC isn’t discoverable
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
    "Lockheed Martin", "Northrop Grumman", "Raytheon", "General Dynamics", "BAE Systems",
    "Huntington Ingalls", "L3Harris", "Boeing Defense", "Textron", "Collins Aerospace",
    "Electric Boat", "Newport News Shipbuilding", "Bath Iron Works",
]

# -----------------------------
# Styling (polished, minimal)
# -----------------------------
st.markdown("""
<style>
/* Dark theme polish */
.stApp { background: #0f172a; color: #e2e8f0; }
header[data-testid="stHeader"] { display: none; }
.stMainBlockContainer { padding-top: 0.5rem; }

/* Hero header */
.wbi-hero {
  background: radial-gradient(1200px 600px at 10% -10%, rgba(37,99,235,.3), transparent),
              linear-gradient(135deg,#0b1225 0%,#111827 70%);
  padding: 28px 16px 16px;
  border-bottom: 1px solid #1f2937;
}
.wbi-logo {
  display: flex; align-items: center; justify-content: center; gap:.75rem;
  color:#fff; font-weight:800; font-size: 36px; letter-spacing:.5px;
}
.wbi-sub { text-align:center; color:#94a3b8; margin-top:.25rem; }

/* Cards & metrics */
.wbi-card { background:#111827; border:1px solid #1f2937; border-radius:14px; padding:18px; }
.metric-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:12px;margin:14px 0}
.metric-card{background:#0b1225;border:1px solid #1f2937;border-radius:12px;padding:14px;text-align:center}
.metric-value{font-size:22px;font-weight:800;color:#fff;margin:0}
.metric-label{font-size:11px;color:#94a3b8;text-transform:uppercase;letter-spacing:.05em;margin:4px 0 0}

/* Tabs */
.stTabs [data-baseweb="tab-list"]{border-bottom:1px solid #1f2937;background:#0b1225}
.stTabs [data-baseweb="tab"]{color:#94a3b8;border:none;padding:10px 14px}
.stTabs [data-baseweb="tab"][aria-selected="true"]{color:#fff;border-bottom:3px solid #2563eb;background:#0b1225}

/* Buttons */
.stButton > button{background:linear-gradient(135deg,#1e40af,#2563eb);color:#fff;border:none;border-radius:12px;
  padding:10px 16px;font-weight:700;box-shadow:0 6px 24px rgba(37,99,235,.25)}
.stButton > button:hover{filter:brightness(1.08); transform: translateY(-1px);}

/* Expander */
.streamlit-expanderHeader{background:#0b1225;border:1px solid #1f2937;border-radius:10px;color:#e2e8f0}
.streamlit-expanderContent{background:#0f172a;border:1px solid #1f2937;border-top:none}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="wbi-hero">
  <div class="wbi-logo">⚓ WBI Naval Search Pro</div>
  <div class="wbi-sub">Supplier intelligence & prime office discovery with POC enrichment</div>
</div>
""", unsafe_allow_html=True)

# -----------------------------
# Scoring
# -----------------------------
@dataclass
class SearchConfig:
    base_location: str = "South Bend, Indiana"
    radius_miles: int = 60
    target_company_count: int = 200

    manufacturing_keywords: List[str] = field(default_factory=lambda: [
        "advanced manufacturing", "precision machining", "metal fabrication",
        "additive manufacturing", "3D printing", "CNC machining", "welding",
        "assembly", "fabrication", "machining", "casting", "forging",
        "sheet metal", "tool and die", "injection molding",
        "shipbuilding", "naval shipyard", "marine engineering", "hull fabrication"
    ])
    microelectronics_keywords: List[str] = field(default_factory=lambda: [
        "microelectronics", "semiconductor", "electronics manufacturing",
        "PCB manufacturing", "printed circuit board", "IC design", "integrated circuits",
        "electronic components", "naval electronics", "maritime electronics",
        "avionics", "radar systems", "sonar electronics", "navigation electronics",
        "communication systems", "electronic warfare", "signal processing",
        "embedded systems", "microprocessors", "FPGA", "analog circuits",
        "digital circuits", "RF electronics"
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
        "unmanned ground vehicles", "swarm robotics", "underwater drones",
        "maritime autonomous", "naval drones"
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
DEFENSE_WEIGHTS = {'defense contractor': 15, 'military contractor': 12, 'aerospace': 10,
                    'defense systems': 12, 'military systems': 10, 'government contracting': 8}
ROBOTICS_WEIGHTS = {'robotics': 10, 'automation': 8, 'robotic systems': 10, 'industrial automation': 8}
UNMANNED_WEIGHTS = {'uav': 10, 'uuv': 10, 'usv': 10, 'autonomous': 8, 'drone': 8, 'rov': 8, 'auv': 8}
WORKFORCE_WEIGHTS = {'training': 6, 'academy': 6, 'certification': 5, 'apprenticeship': 5, 'workforce': 4}

# -----------------------------
# Searcher
# -----------------------------
class EnhancedNavalSearcher:
    def __init__(self, config: SearchConfig):
        self.config = config

    # ------- geometry -------
    def _distance_miles_from_base(self, lat: float, lon: float) -> float:
        bases = {
            'south bend': (41.6764, -86.2520), 'norfolk': (36.8508, -76.2859),
            'san diego': (32.7157, -117.1611), 'pearl harbor': (21.3099, -157.8581),
            'newport news': (37.0871, -76.4730), 'bath': (43.9109, -69.8214),
            'groton': (41.3501, -72.0979)
        }
        base_lat, base_lon = 41.6764, -86.2520
        for key, coords in bases.items():
            if key in self.config.base_location.lower():
                base_lat, base_lon = coords
                break
        if not lat or not lon:
            return 999.0
        if GEOPY_AVAILABLE:
            return round(geodesic((base_lat, base_lon), (lat, lon)).miles, 1)
        # fallback
        lat_diff = abs(lat - base_lat); lon_diff = abs(lon - base_lon)
        return round(((lat_diff ** 2 + lon_diff ** 2) ** 0.5) * 69, 1)

    # ------- relevance / scoring -------
    def _is_relevant(self, name: str, types: List[str], query: str) -> bool:
        name_lower = name.lower()
        types_str = " ".join(types).lower()
        combined = f"{name_lower} {types_str} {query.lower()}"

        hard_exclude = [
            'restaurant', 'food', 'catering', 'retail', 'store', 'bank', 'insurance',
            'real estate', 'hospital', 'hotel', 'gas station', 'pharmacy',
            'auto repair', 'automotive repair', 'hair salon', 'nail salon', 'spa',
            'cleaning services', 'janitorial', 'landscaping', 'roofing', 'flooring',
            'residential construction', 'home improvement', 'interior design'
        ]
        if any(ex in combined for ex in hard_exclude):
            return False

        indicators = [
            'manufacturing', 'machining', 'fabrication', 'welding', 'casting', 'forging', 'cnc',
            'precision', 'metal', 'assembly', 'production', 'shipyard', 'shipbuilding',
            'marine', 'naval', 'maritime', 'submarine', 'vessel', 'electronics', 'microelectronics',
            'semiconductor', 'pcb', 'circuits', 'radar', 'sonar', 'aerospace', 'defense contractor',
            'military contractor', 'avionics', 'defense systems', 'robotics', 'automation',
            'controls', 'systems integration', 'maritime academy', 'naval training',
            'shipyard training', 'welding certification', 'supplier', 'office'
        ]
        return any(ind in combined for ind in indicators)

    def _business_size(self, name: str, place: Dict) -> str:
        name_lower = name.lower()
        reviews = place.get('userRatingCount', 0)
        rating = place.get('rating', 0)
        majors = ['honeywell','boeing','lockheed','lockheed martin','raytheon','northrop','general dynamics',
                  'bae systems','textron','collins','pratt','rolls','huntington ingalls',
                  'newport news shipbuilding','bath iron works','electric boat','l3harris','intel','texas instruments']
        if any(m in name_lower for m in majors):
            return "Fortune 500 / Major Corporation"
        if reviews > 200 or (reviews > 100 and rating > 4.0): return "Large Corporation"
        if reviews > 50 or (reviews > 25 and rating > 4.2):  return "Medium Business"
        if reviews > 10:                                     return "Small-Medium Business"
        return "Small Business"

    def _capabilities(self, name: str, types: List[str]) -> List[str]:
        name_lower = name.lower(); types_str = " ".join(types).lower()
        combined = f"{name_lower} {types_str}"
        out = set()
        mapping = {
            'cnc':'CNC Machining','machining':'Precision Machining','welding':'Welding Services',
            'fabrication':'Metal Fabrication','manufacturing':'Manufacturing','casting':'Metal Casting',
            'forging':'Metal Forging','sheet metal':'Sheet Metal Work','additive':'3D Printing/Additive Manufacturing',
            '3d printing':'3D Printing/Additive Manufacturing','microelectronics':'Microelectronics Manufacturing',
            'semiconductor':'Semiconductor Design/Fab','pcb':'PCB Manufacturing','electronics':'Electronics Manufacturing',
            'electronic components':'Electronic Components','circuits':'Circuit Design/Manufacturing',
            'radar':'Radar Systems','sonar':'Sonar Systems','avionics':'Avionics Systems',
            'automation':'Industrial Automation','robotics':'Robotics Integration','robotic':'Robotic Systems',
            'vision':'Machine Vision Systems','control':'Process Control Systems',
            'shipyard':'Shipbuilding','shipbuilding':'Shipbuilding','marine':'Marine Engineering',
            'naval':'Naval Systems','maritime':'Maritime Systems','submarine':'Submarine Systems',
            'hull':'Hull Fabrication','aerospace':'Aerospace Manufacturing','defense':'Defense Systems',
            'military':'Military Systems','training':'Training Services','academy':'Educational Academy',
            'certification':'Certification Programs','apprenticeship':'Apprenticeship Programs','office':'Prime Contractor Office'
        }
        for k, v in mapping.items():
            if k in combined: out.add(v)
        return list(out) if out else ["General Manufacturing"]

    def _score(self, c: Dict) -> Dict:
        name = c['name'].lower()
        industry = c['industry'].lower()
        description = c['description'].lower()
        caps = ' '.join(c['capabilities']).lower()
        combined = f"{name} {industry} {description} {caps}"
        s = dict(manufacturing_score=0.0, microelectronics_score=0.0, robotics_score=0.0,
                 unmanned_score=0.0, workforce_score=0.0, defense_score=0.0, naval_score=0.0, total_score=0.0)
        for k, w in MICROELECTRONICS_WEIGHTS.items():
            if k in combined: s['microelectronics_score'] += w
        for k, w in NAVAL_WEIGHTS.items():
            if k in combined: s['naval_score'] += w
        for k, w in MANUFACTURING_WEIGHTS.items():
            if k in combined: s['manufacturing_score'] += w
        for k, w in DEFENSE_WEIGHTS.items():
            if k in combined: s['defense_score'] += w
        for k, w in ROBOTICS_WEIGHTS.items():
            if k in combined: s['robotics_score'] += w
        for k, w in UNMANNED_WEIGHTS.items():
            if k in combined: s['unmanned_score'] += w
        for k, w in WORKFORCE_WEIGHTS.items():
            if k in combined: s['workforce_score'] += w
        rating = c.get('rating', 0); reviews = c.get('user_ratings_total', 0)
        quality = 5 if (rating >= 4.5 and reviews >= 10) else 3 if (rating >= 4.0 and reviews >= 5) else 0
        s['total_score'] = (s['manufacturing_score'] * 1.0 + s['microelectronics_score'] * 2.0 +
                            s['naval_score'] * 1.5 + s['defense_score'] * 1.2 + s['robotics_score'] * 0.8 +
                            s['unmanned_score'] * 0.9 + s['workforce_score'] * 0.7 + quality)
        return s

    # ------- Google Places -------
    def _places_search(self, body: Dict, headers: Dict) -> Dict:
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
            'X-Goog-FieldMask': ('places.displayName,places.formattedAddress,places.location,places.types,'
                                 'places.websiteUri,places.nationalPhoneNumber,places.rating,places.userRatingCount,'
                                 'places.businessStatus')
        }
        all_hits: List[Dict] = []
        body = {"textQuery": query, "maxResultCount": 20}
        page = 0
        while True:
            data = self._places_search(body, headers)
            places = data.get('places', []) or []
            all_hits.extend(self._process_places(places, query))
            next_token = data.get('nextPageToken') or data.get('next_page_token')
            if not next_token or page >= 2:
                break
            page += 1
            time.sleep(0.6)
            body["pageToken"] = next_token
        return all_hits

    def _process_places(self, places: List[Dict], query: str) -> List[Dict]:
        out = []
        for p in places:
            try:
                name = p.get('displayName', {}).get('text', 'Unknown')
                types = p.get('types', []) or []
                if not self._is_relevant(name, types, query):
                    continue
                lat = p.get('location', {}).get('latitude', 0.0)
                lon = p.get('location', {}).get('longitude', 0.0)
                dist = self._distance_miles_from_base(lat, lon)
                c = dict(
                    name=name,
                    location=p.get('formattedAddress', 'Unknown'),
                    industry=', '.join(types[:3]) if types else 'Manufacturing/Office',
                    description=self._desc_from_query(name, types, query),
                    size=self._business_size(name, p),
                    capabilities=self._capabilities(name, types),
                    lat=lat, lon=lon, website=p.get('websiteUri', 'Not available'),
                    phone=p.get('nationalPhoneNumber', 'Not available'),
                    rating=p.get('rating', 0), user_ratings_total=p.get('userRatingCount', 0),
                    types=types, distance_miles=dist,
                    is_prime_office=self._is_prime_office(name)
                )
                out.append(c)
            except Exception:
                continue
        return out

    def _desc_from_query(self, name: str, types: List[str], query: str) -> str:
        base = f"Type: {', '.join(types[:2]) if types else 'Manufacturing/Office'}"
        q = query.lower()
        if 'microelectronics' in q or 'semiconductor' in q: base += " · Microelectronics/Semiconductor"
        elif 'electronics' in q: base += " · Electronics"
        if 'naval' in q or 'marine' in q: base += " · Naval/Maritime"
        if 'aerospace' in q or 'defense' in q: base += " · Aerospace/Defense"
        if 'shipbuilding' in q or 'shipyard' in q: base += " · Shipbuilding"
        if 'automation' in q or 'robotics' in q: base += " · Automation/Robotics"
        if 'office' in q: base += " · Prime Contractor Office"
        return base

    def _is_prime_office(self, name: str) -> bool:
        n = name.lower()
        return any(b.lower() in n for b in PRIME_BRANDS)

    # ------- Build queries -------
    def _domain_queries(self, base_location: str) -> List[str]:
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

    def _prime_queries(self, base_location: str) -> List[str]:
        q = []
        for brand in PRIME_BRANDS:
            q += [
                f"{brand} office near {base_location}",
                f"{brand} facility near {base_location}",
                f"{brand} supplier office near {base_location}",
            ]
        return q

    # ------- POC enrichment -------
    def _fetch_html(self, url: str) -> str:
        try:
            r = requests.get(url, headers=HEADERS_HTML, timeout=HTTP_TIMEOUT)
            if r.status_code == 200 and "text" in r.headers.get("Content-Type", ""):
                return r.text
        except Exception:
            pass
        return ""

    def _contact_paths(self, base_url: str) -> List[str]:
        paths = ["", "/contact", "/contact-us", "/contacts", "/suppliers", "/supply-chain",
                 "/supplier", "/doing-business", "/about/contact"]
        out = []
        for p in paths:
            if base_url.endswith("/") and p.startswith("/"):
                out.append(base_url[:-1] + p)
            elif (not base_url.endswith("/")) and (not p.startswith("/")):
                out.append(base_url + "/" + p)
            else:
                out.append(base_url + p)
        return list(dict.fromkeys(out))

    def _extract_emails(self, html: str) -> List[str]:
        raw = set(re.findall(r'[\w\.-]+@[\w\.-]+\.\w+', html, flags=re.IGNORECASE))
        junk = {"example.com", "email.com", "domain.com"}
        return sorted(e for e in raw if not any(j in e.lower() for j in junk))

    def _brand_key(self, name_or_text: str) -> Optional[str]:
        n = (name_or_text or "").lower()
        for k in CORPORATE_POC_FALLBACK.keys():
            if k in n:
                return k
        for b in [x.lower() for x in PRIME_BRANDS]:
            if b.split()[0] in n:
                return b
        return None

    def enrich_poc_for_prime(self, c: Dict) -> Dict:
        poc_primary: List[str] = []
        poc_backup: List[str] = []
        site = c.get("website", "")
        if site and site.startswith("http"):
            for url in self._contact_paths(site):
                html = self._fetch_html(url)
                if not html:
                    continue
                emails = self._extract_emails(html)
                if emails:
                    # prefer supplier/procurement/business dev hints
                    pri = [e for e in emails if any(k in e.lower() for k in
                         ["supplier","procure","purchas","subcontract","bd","business","development","supplychain","supply"])]
                    if pri:
                        poc_primary.extend(pri)
                    else:
                        poc_backup.extend(emails)
                if len(poc_primary) >= 2:
                    break

        if not poc_primary:
            brand = self._brand_key(c.get("name","")) or self._brand_key(c.get("industry","")) or self._brand_key(c.get("description",""))
            for key, inboxes in CORPORATE_POC_FALLBACK.items():
                if brand and key in brand:
                    poc_backup.extend([f"{e} (corporate fallback)" for e in inboxes])
                    break

        if poc_primary: c["poc_emails"] = sorted(list(set(poc_primary)))[:3]
        if poc_backup:  c["poc_emails_backup"] = sorted(list(set(poc_backup)))[:3]
        return c

    # ------- Normalize (prevents KeyError) -------
    @staticmethod
    def normalize(company: Dict) -> Dict:
        # guarantee all expected fields exist
        defaults = {
            'name': 'Unknown', 'location': 'Unknown', 'distance_miles': 999.0, 'size': 'Small Business',
            'industry': 'Manufacturing/Office', 'total_score': 0.0, 'microelectronics_score': 0.0,
            'naval_score': 0.0, 'defense_score': 0.0, 'manufacturing_score': 0.0,
            'rating': 0.0, 'user_ratings_total': 0, 'phone': 'Not available', 'website': 'Not available',
            'is_prime_office': False, 'capabilities': [], 'lat': 0.0, 'lon': 0.0
        }
        for k, v in defaults.items():
            company.setdefault(k, v)
        # types must be list
        if not isinstance(company.get('capabilities'), list):
            company['capabilities'] = [str(company.get('capabilities'))]
        return company

    # ------- Public entry -------
    def search_companies(self) -> List[Dict]:
        api_key = st.session_state.get('api_key', GOOGLE_PLACES_API_KEY)
        if not api_key:
            st.error("🔑 Google Places API key is REQUIRED.")
            return []

        base = self.config.base_location
        queries = self._domain_queries(base) + self._prime_queries(base)

        # single progress bar, no noisy per-query prints
        progress = st.progress(0.0, text="Starting enhanced search…")
        all_hits: List[Dict] = []

        for i, q in enumerate(queries, start=1):
            progress.progress(i / len(queries), text=f"Searching ({i}/{len(queries)}): {q}")
            all_hits.extend(self._search_google_places(q, api_key))
            time.sleep(0.4)
        progress.empty()

        # de-dupe by (name + rounded coords)
        unique: List[Dict] = []
        seen = set()
        for c in all_hits:
            bucket = (c.get('name','').strip().lower(), round(c.get('lat',0.0),3), round(c.get('lon',0.0),3))
            if bucket in seen:
                continue
            seen.add(bucket)

            # distance gate + scoring
            if c.get('distance_miles', 999.0) <= self.config.radius_miles:
                c.update(self._score(c))
                unique.append(self.normalize(c))

        # POC enrichment for primes
        primes = [x for x in unique if x.get('is_prime_office')]
        if primes:
            with st.spinner(f"Enriching POCs for {len(primes)} prime offices…"):
                for p in primes:
                    self.enrich_poc_for_prime(p)

        unique.sort(key=lambda x: x['total_score'], reverse=True)
        return unique[: self.config.target_company_count]

# -----------------------------
# Visualization
# -----------------------------
def company_map(companies: List[Dict], base_location: str):
    if not companies: return None
    df = pd.DataFrame(companies)
    df = df[(df['lat'] != 0) & (df['lon'] != 0)]
    if df.empty: return None

    bases = {
        'south bend': (41.6764, -86.2520), 'norfolk': (36.8508, -76.2859),
        'san diego': (32.7157, -117.1611), 'pearl harbor': (21.3099, -157.8581),
        'newport news': (37.0871, -76.4730), 'bath': (43.9109, -69.8214), 'groton': (41.3501, -72.0979)
    }
    base_lat, base_lon = 41.6764, -86.2520
    for key, coords in bases.items():
        if key in base_location.lower():
            base_lat, base_lon = coords; break

    fig = go.Figure()
    fig.add_trace(go.Scattermapbox(
        lat=[base_lat], lon=[base_lon], mode='markers',
        marker=dict(size=18, color='blue'),
        text=[f"Search Center: {base_location}"], name='Base'
    ))

    hovers = []
    for _, r in df.iterrows():
        lines = [
            f"{'🏛️ PRIME ' if r.get('is_prime_office') else ''}{r['name']}",
            f"Score {r['total_score']:.1f} · Micro {r['microelectronics_score']:.1f} · Naval {r['naval_score']:.1f}",
            f"Dist {r['distance_miles']} mi"
        ]
        if r.get('poc_emails'): lines.append("POC: " + ", ".join(r['poc_emails'][:2]))
        hovers.append("<br>".join(lines))

    fig.add_trace(go.Scattermapbox(
        lat=df['lat'], lon=df['lon'], mode='markers',
        marker=dict(size=11, color=df['total_score'], colorscale='Viridis', showscale=True,
                    colorbar=dict(title="Score")),
        text=hovers, name='Entities'
    ))
    fig.update_layout(
        mapbox=dict(style='open-street-map', center=dict(lat=base_lat, lon=base_lon), zoom=8),
        height=600, title="Enhanced Supplier & Prime Office Map"
    )
    return fig

def metrics_html(companies: List[Dict]) -> str:
    if not companies: return ""
    df = pd.DataFrame(companies)
    def count(col, th): return len(df[df[col] >= th])
    total = len(df)
    primes = len(df[df['is_prime_office'] == True])
    micro = count('microelectronics_score', 10)
    naval = count('naval_score', 10)
    defense = count('defense_score', 8)
    high = count('total_score', 15)
    smalls = len(df[df['size'].str.contains('Small', na=False)])
    avgd = df['distance_miles'].mean()
    qual = len(df[(df['rating'] >= 4.0) & (df['user_ratings_total'] >= 10)])
    return f"""
    <div class="metric-grid">
      <div class="metric-card"><p class="metric-value">{total}</p><p class="metric-label">Total Entities</p></div>
      <div class="metric-card"><p class="metric-value">{primes}</p><p class="metric-label">Prime Offices</p></div>
      <div class="metric-card"><p class="metric-value">{micro}</p><p class="metric-label">Microelectronics</p></div>
      <div class="metric-card"><p class="metric-value">{naval}</p><p class="metric-label">Naval / Maritime</p></div>
      <div class="metric-card"><p class="metric-value">{defense}</p><p class="metric-label">Defense Contractors</p></div>
      <div class="metric-card"><p class="metric-value">{high}</p><p class="metric-label">High Relevance</p></div>
      <div class="metric-card"><p class="metric-value">{smalls}</p><p class="metric-label">Small Businesses</p></div>
      <div class="metric-card"><p class="metric-value">{avgd:.1f} mi</p><p class="metric-label">Avg Distance</p></div>
      <div class="metric-card"><p class="metric-value">{qual}</p><p class="metric-label">Quality Suppliers</p></div>
    </div>
    """

def executive_report(companies: List[Dict], config: SearchConfig) -> str:
    if not companies: return "No companies found."
    df = pd.DataFrame(companies)
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    total = len(df); micro = len(df[df['microelectronics_score'] >= 10])
    naval = len(df[df['naval_score'] >= 10]); high = len(df[df['total_score'] >= 15])
    top_overall = df.nlargest(3, 'total_score')
    top_micro = df.nlargest(3, 'microelectronics_score')
    top_naval = df.nlargest(3, 'naval_score')
    rep = f"""
# 🎯 WBI Naval Search — Executive Summary

**Generated:** {now}  
**Location:** {config.base_location} — **Radius:** {config.radius_miles} mi

- Entities evaluated: **{total}**
- Microelectronics-capable (≥10): **{micro}**
- Naval/Maritime focus (≥10): **{naval}**
- High overall (≥15): **{high}**
- Avg distance: **{df['distance_miles'].mean():.1f} mi**

## 🏆 Top Overall
"""
    for i, (_, r) in enumerate(top_overall.iterrows(), start=1):
        rep += f"\n**{i}. {r['name']}** — Score {r['total_score']:.1f} · Micro {r['microelectronics_score']:.1f} · Naval {r['naval_score']:.1f}"
    if not top_micro.empty:
        rep += "\n\n## 🔬 Microelectronics Leaders\n"
        for i, (_, r) in enumerate(top_micro.iterrows(), start=1):
            rep += f"\n**{i}. {r['name']}** — Micro {r['microelectronics_score']:.1f} · Capabilities: {', '.join(r['capabilities'][:4])}"
    if not top_naval.empty:
        rep += "\n\n## ⚓ Naval/Maritime Specialists\n"
        for i, (_, r) in enumerate(top_naval.iterrows(), start=1):
            rep += f"\n**{i}. {r['name']}** — Naval {r['naval_score']:.1f} · Capabilities: {', '.join(r['capabilities'][:4])}"
    return rep

# -----------------------------
# App
# -----------------------------
def main():
    # Session state
    if 'search_triggered' not in st.session_state: st.session_state.search_triggered = False
    if 'companies' not in st.session_state: st.session_state.companies = []

    # Sidebar
    st.sidebar.header("🔧 Search Configuration")
    if not GOOGLE_PLACES_API_KEY:
        st.sidebar.warning("No GOOGLE_PLACES_API_KEY in env.")
        api = st.sidebar.text_input("Google Places API Key", type="password")
        if api: st.session_state.api_key = api; st.sidebar.success("API key set.")
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
    config.base_location = presets[st.sidebar.selectbox("Naval hub", list(presets.keys()))]
    config.radius_miles = st.sidebar.slider("Search Radius (miles)", 10, 150, 60)
    config.target_company_count = st.sidebar.slider("Max Entities", 10, 400, 200)

    with st.sidebar:
        st.markdown("---")
        colA, colB = st.columns([1,1])
        with colA:
            run = st.button("🔍 Run Search", type="primary")
        with colB:
            show_logs = st.toggle("Show debug logs", value=False)
        if run:
            st.session_state.search_triggered = True
            st.session_state.companies = []

    # Search runner (results appear without long scrolling)
    if st.session_state.search_triggered:
        with st.spinner("Running enhanced search (domains + prime offices)…"):
            searcher = EnhancedNavalSearcher(config)
            companies = searcher.search_companies()
            st.session_state.companies = companies
        st.session_state.search_triggered = False

    companies = st.session_state.companies

    # Top summary + metrics
    st.markdown('<div class="wbi-card">', unsafe_allow_html=True)
    st.subheader("📊 Intelligence Dashboard")
    if companies:
        st.markdown(metrics_html(companies), unsafe_allow_html=True)
    else:
        st.info("No results yet. Configure options and click **Run Search**.")
    st.markdown('</div>', unsafe_allow_html=True)

    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📋 Directory", "🗺️ Map", "📊 Analytics", "📄 Export"])

    with tab1:
        st.subheader("🏭 Supplier & Prime Office Directory")
        c1, c2, c3, c4, c5 = st.columns(5)
        with c1: min_total = st.slider("Min Total Score", 0, 60, 10)
        with c2: min_micro = st.slider("Min Micro Score", 0, 40, 0)
        with c3: min_naval = st.slider("Min Naval Score", 0, 40, 0)
        with c4: size_filter = st.selectbox("Company Size", ["All","Small Business","Small-Medium Business","Medium Business","Large Corporation","Fortune 500 / Major Corporation"])
        with c5: primes_only = st.checkbox("Only Prime Offices", value=False)

        filtered = [c for c in companies if c['total_score'] >= min_total
                    and c['microelectronics_score'] >= min_micro
                    and c['naval_score'] >= min_naval]
        if size_filter != "All":
            filtered = [c for c in filtered if c['size'] == size_filter]
        if primes_only:
            filtered = [c for c in filtered if c.get('is_prime_office')]

        st.info(f"Showing {len(filtered)} of {len(companies)} entities")
        if not filtered:
            st.warning("No entities match the current filters.")
        else:
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
                        for cap in c['capabilities']: st.write(f"• {cap}")
                        st.markdown("**Quality**")
                        st.write(f"⭐ {c['rating']:.1f} ({c['user_ratings_total']} reviews)")
                    st.markdown(f"**Description:** {c['description']}")
                    if c.get('is_prime_office') or c.get('poc_emails') or c.get('poc_emails_backup'):
                        st.markdown("**📇 Points of Contact**")
                        if c.get('poc_emails'): st.write("Primary:", ", ".join(c['poc_emails']))
                        if c.get('poc_emails_backup'):
                            back = c['poc_emails_backup']
                            st.write("Backup:", ", ".join(back) if isinstance(back, list) else back)

    with tab2:
        st.subheader("🗺️ Map")
        fig = company_map(companies, config.base_location)
        if fig: st.plotly_chart(fig, use_container_width=True)
        else:   st.info("No mappable coordinates yet.")

    with tab3:
        st.subheader("📊 Analytics")
        if not companies:
            st.info("Run a search to see analytics.")
        else:
            df = pd.DataFrame(companies).copy()
            # ensure columns exist
            for col in ['total_score','microelectronics_score','naval_score','defense_score','manufacturing_score','capabilities']:
                if col not in df.columns: df[col] = 0
            left, right = st.columns(2)
            with left:
                st.plotly_chart(px.histogram(df, x='total_score', title='Score Distribution', nbins=20), use_container_width=True)
                st.plotly_chart(px.scatter(df, x='microelectronics_score', y='naval_score',
                                           hover_name='name', title='Microelectronics vs Naval',
                                           size='total_score'), use_container_width=True)
            with right:
                cats = ['manufacturing_score', 'microelectronics_score', 'naval_score', 'defense_score']
                vals = [float(df[c].mean() or 0) for c in cats]
                st.plotly_chart(px.bar(x=['Manufacturing','Microelectronics','Naval','Defense'], y=vals,
                                       title='Average Scores by Category'), use_container_width=True)
                caps = []
                for row in df['capabilities']: caps.extend(row if isinstance(row, list) else [row])
                if caps:
                    cap_counts = pd.Series(caps).value_counts().head(12)
                    st.plotly_chart(px.bar(x=cap_counts.values, y=cap_counts.index, title='Top Capabilities', orientation='h'),
                                    use_container_width=True)

    with tab4:
        st.subheader("📄 Export")
        if not companies:
            st.info("Run a search to export results.")
        else:
            df = pd.DataFrame(companies).copy()
            expected = ['name','location','distance_miles','size','industry','total_score','microelectronics_score',
                        'naval_score','defense_score','manufacturing_score','rating','user_ratings_total',
                        'phone','website','is_prime_office','poc_emails','poc_emails_backup']
            for col in expected:
                if col not in df.columns:
                    df[col] = "" if col in ['phone','website'] else (False if col=='is_prime_office' else 0)
            export_df = df[expected].rename(columns={
                'name':'Company Name','location':'Location','distance_miles':'Distance (Miles)',
                'size':'Company Size','industry':'Industry','total_score':'Total Score',
                'microelectronics_score':'Microelectronics Score','naval_score':'Naval Score',
                'defense_score':'Defense Score','manufacturing_score':'Manufacturing Score',
                'rating':'Rating','user_ratings_total':'Review Count','phone':'Phone','website':'Website',
                'is_prime_office':'Prime Office','poc_emails':'POC Emails','poc_emails_backup':'POC Emails (Backup)'
            })
            export_df['Search Location'] = config.base_location
            export_df['Search Date'] = datetime.now().strftime("%Y-%m-%d")

            st.dataframe(export_df, use_container_width=True, height=420)
            csv = export_df.to_csv(index=False)
            excel_csv = '\ufeff' + csv
            c1, c2 = st.columns(2)
            with c1:
                st.download_button("📥 Download CSV", data=csv,
                                   file_name=f"naval_companies_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                                   mime="text/csv")
            with c2:
                st.download_button("📊 Download for Excel", data=excel_csv,
                                   file_name=f"naval_companies_{datetime.now().strftime('%Y%m%d_%H%M')}_excel.csv",
                                   mime="text/csv")

if __name__ == "__main__":
    main()
