"""Reference tables for the freight pack — the layer that keeps records plausible.

Realism lives here, not in the renderer. A generator sampling uniformly over made-up
cities produces fluent nonsense; sampling lanes and commodities from tables shaped
like the real thing produces a corpus a domain expert recognises.

What is real, and what is not — stated so nobody mistakes one for the other:

  CITIES      real US freight hubs, ordered roughly by tonnage so `Sampler.zipf`
              reproduces the heavy tail of real lane frequency.
  COMMODITIES real HS chapter headings and codes (WCO Harmonized System).
  EQUIPMENT   real trailer types and their real weight limits.
  ACCESSORIALS real charge names used across US truckload.
  COMPANIES   **synthetic** — generated from real naming *patterns*, not copied
              from FMCSA. Real carriers are public record, but putting named real
              companies into fabricated disputes and overcharges is a different
              thing from using public reference data, and not one worth doing.

  TODO(pack): load the FMCSA Company Census for real DOT/MC numbers and fleet-size
  distributions when the corpus needs carrier-level realism (drill 4 will).
"""
from __future__ import annotations

# Real US freight hubs with real coordinates, roughly tonnage-ordered so
# `Sampler.zipf` reproduces the heavy tail of real lane frequency.
# (city, state, lat, lon)
CITIES = [
    ("Chicago", "IL", 41.88, -87.63), ("Los Angeles", "CA", 34.05, -118.24),
    ("Dallas", "TX", 32.78, -96.80), ("Atlanta", "GA", 33.75, -84.39),
    ("Houston", "TX", 29.76, -95.37), ("Newark", "NJ", 40.74, -74.17),
    ("Memphis", "TN", 35.15, -90.05), ("Columbus", "OH", 39.96, -83.00),
    ("Kansas City", "MO", 39.10, -94.58), ("Laredo", "TX", 27.51, -99.51),
    ("Savannah", "GA", 32.08, -81.09), ("Fresno", "CA", 36.74, -119.79),
    ("Phoenix", "AZ", 33.45, -112.07), ("Denver", "CO", 39.74, -104.99),
    ("Charlotte", "NC", 35.23, -80.84), ("Indianapolis", "IN", 39.77, -86.16),
    ("Salt Lake City", "UT", 40.76, -111.89), ("Portland", "OR", 45.52, -122.68),
    ("Nashville", "TN", 36.16, -86.78), ("Reno", "NV", 39.53, -119.81),
]

# Great-circle miles x this = driving miles. 1.17 is the usual US highway circuity
# factor. It under-reads mountain routes (Fresno->Reno crosses the Sierra and is
# really ~340 mi, not the ~225 this yields) — an approximation the corpus card
# declares rather than hides.
ROAD_CIRCUITY = 1.17

# Real HS headings. (chapter_code, description, reefer_required, density_class)
COMMODITIES = [
    ("0704", "Fresh lettuce, palletized", True, 60),
    ("0803", "Bananas, fresh", True, 65),
    ("0201", "Beef, chilled, boxed", True, 70),
    ("0406", "Cheese, refrigerated", True, 70),
    ("2202", "Bottled beverages, non-alcoholic", False, 100),
    ("3923", "Plastic packaging articles", False, 55),
    ("4819", "Cartons and corrugated boxes", False, 50),
    ("6109", "T-shirts, cotton, cartoned", False, 85),
    ("7326", "Steel articles, misc.", False, 125),
    ("8544", "Insulated wire and cable", False, 92),
    ("9403", "Furniture, knocked down", False, 70),
    ("3004", "Pharmaceutical preparations", True, 100),
]

# (name, max_payload_lb, reefer)
EQUIPMENT = [
    ("53ft dry van", 45000, False),
    ("53ft reefer", 43500, True),
    ("48ft flatbed", 48000, False),
    ("53ft dry van (air-ride)", 44000, False),
    ("40ft container drayage", 44000, False),
]

ACCESSORIALS = [
    "detention", "layover", "lumper fee", "driver assist", "reconsignment",
    "tarping", "stop-off", "after-hours delivery", "liftgate",
]

# Synthetic company names, built from real naming patterns. See module docstring.
_SURNAMES = [
    "Ashland", "Belmont", "Carver", "Dunlop", "Ellsworth", "Fairview", "Granger",
    "Halloran", "Ironside", "Jessup", "Kestrel", "Lawton", "Marbury", "Northgate",
    "Osgood", "Pinehurst", "Quillan", "Redstone", "Stanfield", "Thorne",
]
_SHIPPER_SUFFIX = ["Foods LLC", "Distributing Inc", "Brands Co", "Growers Co-op",
                   "Industries", "Supply Company", "Packaging Inc", "Farms LLC"]
_CARRIER_SUFFIX = ["Trucking LLC", "Transport Inc", "Logistics LLC", "Freight Lines",
                   "Carriers Inc", "Express LLC", "Motor Freight"]

SHIPPERS = [f"{s} {x}" for s in _SURNAMES for x in _SHIPPER_SUFFIX][:80]
CARRIERS = [f"{s} {x}" for s in reversed(_SURNAMES) for x in _CARRIER_SUFFIX][:70]

# Lane rate anchors, $/mile, by equipment. Rough but not arbitrary.
RATE_PER_MILE = {
    "53ft dry van": 2.10,
    "53ft reefer": 2.55,
    "48ft flatbed": 2.45,
    "53ft dry van (air-ride)": 2.25,
    "40ft container drayage": 2.80,
}

def road_miles(lat1: float, lon1: float, lat2: float, lon2: float) -> int:
    """Driving miles between two hubs — great-circle distance x circuity.

    The earlier version of this hashed the city-pair into a pseudo-distance, which
    priced Los Angeles -> Chicago at ~950 miles. Real cities with fabricated
    distances are the exact failure this pack exists to avoid: fluent nonsense that
    a domain expert spots in one glance and a model happily learns.
    """
    import math

    r_earth = 3958.8  # miles
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = p2 - p1
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlam / 2) ** 2
    great_circle = 2 * r_earth * math.asin(math.sqrt(a))
    return int(round(great_circle * ROAD_CIRCUITY))
