"""Value pools for the synthetic generator (names, organizations, geography, titles)."""

from __future__ import annotations

FIRST_NAMES: tuple[str, ...] = (
    "James", "Robert", "John", "Michael", "William", "David", "Richard", "Joseph", "Thomas",
    "Charles", "Christopher", "Daniel", "Matthew", "Anthony", "Mark", "Donald", "Steven", "Paul",
    "Andrew", "Joshua", "Kenneth", "Kevin", "Brian", "George", "Timothy", "Ronald", "Edward",
    "Jason", "Jeffrey", "Ryan", "Jacob", "Gary", "Nicholas", "Eric", "Jonathan", "Stephen",
    "Larry", "Justin", "Scott", "Brandon", "Benjamin", "Samuel", "Gregory", "Alexander", "Patrick",
    "Frank", "Raymond", "Jack", "Dennis", "Jerry", "Mary", "Patricia", "Jennifer", "Linda",
    "Elizabeth", "Barbara", "Susan", "Jessica", "Sarah", "Karen", "Lisa", "Nancy", "Betty",
    "Margaret", "Sandra", "Ashley", "Kimberly", "Emily", "Donna", "Michelle", "Carol", "Amanda",
    "Dorothy", "Melissa", "Deborah", "Stephanie", "Rebecca", "Sharon", "Laura", "Cynthia",
    "Kathleen", "Amy", "Angela", "Shirley", "Anna", "Brenda", "Pamela", "Emma", "Nicole", "Helen",
    "Samantha", "Katherine", "Christine", "Debra", "Rachel", "Carolyn", "Janet", "Catherine",
    "Maria", "Heather", "Diane", "Ruth", "Julie", "Olivia", "Joyce", "Virginia", "Victoria",
    "Kelly", "Lauren", "Christina", "Joan", "Evelyn", "Judith", "Megan", "Andrea", "Cheryl",
    "Hannah", "Jacqueline", "Martha", "Gloria", "Teresa", "Ann", "Sara", "Madison", "Frances",
    "Kathryn", "Janice", "Jean", "Abigail", "Alice", "Judy", "Sophia", "Grace", "Denise",
    "Amber", "Doris", "Marilyn", "Danielle", "Beverly", "Isabella", "Theresa", "Diana",
    "Natalie", "Brittany", "Charlotte", "Marie", "Kayla", "Alexis", "Lori", "Jose", "Luis",
    "Carlos", "Juan", "Miguel", "Wei", "Priya", "Ravi", "Anil", "Sunita", "Mohammed", "Fatima",
    "Chen", "Hiroshi", "Yuki", "Aisha", "Omar", "Elena", "Sofia", "Mateo",
)

LAST_NAMES: tuple[str, ...] = (
    "Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller", "Davis", "Rodriguez",
    "Martinez", "Hernandez", "Lopez", "Gonzalez", "Wilson", "Anderson", "Thomas", "Taylor",
    "Moore", "Jackson", "Martin", "Lee", "Perez", "Thompson", "White", "Harris", "Sanchez",
    "Clark", "Ramirez", "Lewis", "Robinson", "Walker", "Young", "Allen", "King", "Wright",
    "Scott", "Torres", "Nguyen", "Hill", "Flores", "Green", "Adams", "Nelson", "Baker", "Hall",
    "Rivera", "Campbell", "Mitchell", "Carter", "Roberts", "Gomez", "Phillips", "Evans", "Turner",
    "Diaz", "Parker", "Cruz", "Edwards", "Collins", "Reyes", "Stewart", "Morris", "Morales",
    "Murphy", "Cook", "Rogers", "Gutierrez", "Ortiz", "Morgan", "Cooper", "Peterson", "Bailey",
    "Reed", "Kelly", "Howard", "Ramos", "Kim", "Cox", "Ward", "Richardson", "Watson", "Brooks",
    "Chavez", "Wood", "James", "Bennett", "Gray", "Mendoza", "Ruiz", "Hughes", "Price", "Alvarez",
    "Castillo", "Sanders", "Patel", "Myers", "Long", "Ross", "Foster", "Jimenez", "Powell",
    "Jenkins", "Perry", "Russell", "Sullivan", "Bell", "Coleman", "Butler", "Henderson", "Barnes",
    "Gonzales", "Fisher", "Vasquez", "Simmons", "Romero", "Jordan", "Patterson", "Alexander",
    "Hamilton", "Graham", "Reynolds", "Griffin", "Wallace", "Moreno", "West", "Cole", "Hayes",
    "Bryant", "Herrera", "Gibson", "Ellis", "Tran", "Medina", "Aguilar", "Stevens", "Murray",
    "Ford", "Castro", "Marshall", "Owens", "Harrison", "Fernandez", "McDonald", "Woods",
    "Washington", "Kennedy", "Wells", "Vargas", "Henry", "Chen", "Freeman", "Webb", "Tucker",
    "Guzman", "Burns", "Crawford", "Olson", "Simpson", "Porter", "Hunter", "Gordon", "Mendez",
    "Silva", "Shaw", "Snyder", "Mason", "Dixon", "Munoz", "Hunt", "Hicks", "Holmes", "Palmer",
    "Wagner", "Black", "Robertson", "Boyd", "Rose", "Stone", "Salazar", "Fox", "Warren", "Mills",
    "Meyer", "Rice", "Schmidt", "Garza", "Daniels", "Ferguson", "Nichols", "Stephens", "Soto",
    "Weaver", "Ryan", "Gardner", "Payne", "Grant", "Dunn", "Kelley", "Spencer", "Hawkins",
    "Arnold", "Pierce", "Vazquez", "Hansen", "Peters", "Santos", "Hart", "Bradley", "Knight",
    "Elliott", "Cunningham", "Duncan", "Armstrong", "Hudson", "Carroll", "Lane", "Riley",
    "Andrews", "Alvarado", "Ray", "Delgado", "Berry", "Perkins", "Hoffman", "Johnston",
    "Matthews", "Pena", "Richards", "Contreras", "Willis", "Carpenter", "Lawrence", "Sandoval",
    "Guerrero", "George", "Chapman", "Rios", "Estrada", "Ortega", "Watkins", "Greene", "Nunez",
    "Wheeler", "Valdez", "Harper", "Burke", "Larson", "Santiago", "Maldonado", "Morrison",
    "Franklin", "Carlson", "Austin", "Dominguez", "Carr", "Lawson", "Jacobs", "O'Brien", "Lynch",
    "Singh", "Vega", "Bishop", "Montgomery", "Oliver", "Jensen", "Harvey", "Williamson",
    "Gilbert", "Dean", "Sims", "Espinoza", "Howell", "Li", "Wong", "Reid", "Hanson", "Le",
    "McCoy", "Garrett", "Burton", "Fuller", "Wang", "Weber", "Welch", "Rojas", "Lucas",
    "Marquez", "Fields", "Park", "Yang", "Little", "Banks", "Padilla", "Day", "Walsh", "Bowman",
    "Schultz", "Luna", "Fowler", "Mejia", "Schneider", "Kaufman", "Zimmerman", "Okafor", "Nakamura",
)

GENERATIONAL_SUFFIXES: tuple[str, ...] = ("Jr", "Sr", "II", "III")

# (credential, canonical title, [title variants shown in sources])
ROLES: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    ("MD", "PHYSICIAN", ("Physician", "Family Medicine Physician", "Family Practice Doctor",
                         "Internist", "Staff Physician", "Attending Physician", "MD", "Hospitalist")),
    ("DO", "PHYSICIAN", ("Physician", "Family Physician", "DO", "Primary Care Physician", "PCP")),
    ("NP", "NURSE PRACTITIONER", ("Nurse Practitioner", "NP", "ARNP", "Family Nurse Practitioner",
                                  "FNP", "Advanced Practice Nurse", "Nurse Practioner")),
    ("PA-C", "PHYSICIAN ASSISTANT", ("Physician Assistant", "PA-C", "PA", "Physician Associate",
                                     "Physicians Assistant")),
    ("RN", "REGISTERED NURSE", ("Registered Nurse", "RN", "Staff RN", "Staff Nurse", "Clinical Nurse",
                                "Charge Nurse")),
    ("MD", "MEDICAL DIRECTOR", ("Medical Director", "Director of Medicine", "Dir. of Medicine",
                                "Med Director", "Chief Medical Officer")),
    ("MD", "SURGEON", ("Orthopedic Surgeon", "General Surgeon", "Orthopaedic Surgeon", "Surgeon")),
)

# (zip5, city, state, lat, lon) - approximate real centroids so distance levels behave sensibly.
ZIPS: tuple[tuple[str, str, str, float, float], ...] = (
    ("62701", "Springfield", "IL", 39.7990, -89.6440), ("62704", "Springfield", "IL", 39.7740, -89.6870),
    ("60601", "Chicago", "IL", 41.8858, -87.6229), ("60611", "Chicago", "IL", 41.8950, -87.6200),
    ("60614", "Chicago", "IL", 41.9227, -87.6533), ("60201", "Evanston", "IL", 42.0540, -87.6940),
    ("61602", "Peoria", "IL", 40.6910, -89.5940), ("43215", "Columbus", "OH", 39.9650, -83.0030),
    ("43201", "Columbus", "OH", 39.9910, -83.0010), ("43212", "Columbus", "OH", 39.9870, -83.0440),
    ("44113", "Cleveland", "OH", 41.4830, -81.6960), ("44106", "Cleveland", "OH", 41.5060, -81.6060),
    ("45202", "Cincinnati", "OH", 39.1080, -84.5040), ("45402", "Dayton", "OH", 39.7580, -84.1990),
    ("46204", "Indianapolis", "IN", 39.7710, -86.1580), ("46802", "Fort Wayne", "IN", 41.0730, -85.1520),
    ("37203", "Nashville", "TN", 36.1510, -86.7890), ("38103", "Memphis", "TN", 35.1450, -90.0500),
    ("40202", "Louisville", "KY", 38.2510, -85.7550), ("63101", "Saint Louis", "MO", 38.6310, -90.1920),
    ("63110", "Saint Louis", "MO", 38.6250, -90.2660), ("64106", "Kansas City", "MO", 39.1050, -94.5730),
    ("53202", "Milwaukee", "WI", 43.0440, -87.8990), ("53703", "Madison", "WI", 43.0790, -89.3810),
    ("55401", "Minneapolis", "MN", 44.9840, -93.2680), ("48226", "Detroit", "MI", 42.3310, -83.0480),
    ("49503", "Grand Rapids", "MI", 42.9640, -85.6620), ("15222", "Pittsburgh", "PA", 40.4440, -79.9950),
    ("19103", "Philadelphia", "PA", 39.9520, -75.1740), ("30303", "Atlanta", "GA", 33.7530, -84.3900),
    ("28202", "Charlotte", "NC", 35.2270, -80.8430), ("75201", "Dallas", "TX", 32.7870, -96.7990),
    ("75204", "Dallas", "TX", 32.8010, -96.7880), ("77002", "Houston", "TX", 29.7570, -95.3650),
    ("77030", "Houston", "TX", 29.7040, -95.4010), ("78701", "Austin", "TX", 30.2710, -97.7430),
    ("80202", "Denver", "CO", 39.7510, -104.9970), ("85004", "Phoenix", "AZ", 33.4520, -112.0700),
    ("98101", "Seattle", "WA", 47.6100, -122.3340), ("97204", "Portland", "OR", 45.5180, -122.6750),
)

STREET_NAMES: tuple[str, ...] = (
    "Main", "Oak", "Maple", "Washington", "Park", "Lake", "Hill", "Elm", "Cedar", "Walnut", "Pine",
    "Church", "Lincoln", "Jefferson", "Madison", "Jackson", "Franklin", "Market", "River", "Union",
    "Center", "Highland", "Forest", "Spring", "Broadway", "College", "Mill", "Prospect", "Ridge",
    "Sunset", "Medical", "Hospital", "University", "Commerce", "Executive", "Professional",
)
STREET_TYPES: tuple[tuple[str, str], ...] = (
    ("Street", "St"), ("Avenue", "Ave"), ("Road", "Rd"), ("Drive", "Dr"), ("Boulevard", "Blvd"),
    ("Lane", "Ln"), ("Parkway", "Pkwy"), ("Court", "Ct"), ("Place", "Pl"), ("Way", "Way"),
)
DIRECTIONALS: tuple[tuple[str, str], ...] = (("North", "N"), ("South", "S"), ("East", "E"), ("West", "W"))

ORG_ADJECTIVES: tuple[str, ...] = (
    "Riverside", "Lakeview", "Northside", "Southside", "Westgate", "Eastland", "Summit", "Valley",
    "Prairie", "Heritage", "Community", "Regional", "Metro", "Unity", "Mercy", "Trinity", "Advocate",
    "Premier", "Horizon", "Evergreen", "Pinnacle", "Oakwood", "Cornerstone", "Harmony", "Ridgeview",
)
SAINTS: tuple[str, ...] = ("Saint Mary's", "Saint Luke's", "Saint Joseph", "Saint Francis", "Saint Vincent")
ORG_TYPES: tuple[str, ...] = (
    "Medical Center", "Family Practice", "Medical Group", "Health System", "Hospital", "Clinic",
    "Orthopedic Associates", "Physicians Group", "Primary Care Associates", "Pediatrics",
    "Internal Medicine Associates", "Surgical Associates", "Health Partners", "Medical Associates",
    "Cardiology Associates", "Women's Health Center", "Urgent Care",
)
LEGAL_SUFFIX_FORMS: tuple[tuple[str, ...], ...] = (
    ("Inc", "Inc.", "Incorporated", ""),
    ("LLC", "L.L.C.", "LLC.", ""),
    ("PLLC", "P.L.L.C.", ""),
    ("PC", "P.C.", ""),
    ("PA", "P.A.", ""),
    ("Corp", "Corp.", "Corporation", ""),
    ("Ltd", "Ltd.", "Limited", ""),
)
ORG_ABBREVIATIONS: dict[str, str] = {
    "Medical": "Med", "Center": "Ctr", "Hospital": "Hosp", "Associates": "Assoc", "Group": "Grp",
    "Saint": "St.", "Family": "Fam", "Practice": "Prac", "Physicians": "Phys", "Health": "Hlth",
    "Pediatrics": "Peds", "Surgical": "Surg", "Orthopedic": "Ortho", "and": "&", "Services": "Svcs",
    "Department": "Dept", "University": "Univ", "System": "Sys",
}
EMAIL_TLDS: tuple[str, ...] = ("org", "com", "health", "net")
