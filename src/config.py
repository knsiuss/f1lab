# -*- coding: utf-8 -*-
"""
config.py
Application constants and configuration.

:copyright: (c) 2025 F1 Analytics
:license: MIT
"""

import os
import logging
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, field

# Path Configuration

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent

# Data directories
DATA_DIR = PROJECT_ROOT / 'data'
CACHE_DIR = PROJECT_ROOT / 'cache'
NOTEBOOKS_DIR = PROJECT_ROOT / 'notebooks'
LOGS_DIR = PROJECT_ROOT / 'logs'


def _ensure_dirs():
    """Create data/cache/logs directories if they don't exist."""
    for directory in [DATA_DIR, CACHE_DIR, LOGS_DIR]:
        directory.mkdir(exist_ok=True)

# DATA FILE PATHS

@dataclass
class DataFiles:
    """Data file path configuration."""
    race_results: Path = DATA_DIR / 'Formula1_2025Season_RaceResults.csv'
    qualifying_results: Path = DATA_DIR / 'Formula1_2025Season_QualifyingResults.csv'
    sprint_results: Path = DATA_DIR / 'Formula1_2025Season_SprintResults.csv'
    sprint_qualifying: Path = DATA_DIR / 'Formula1_2025Season_SprintQualifyingResults.csv'

DATA_FILES = DataFiles()

# FASTF1 CONFIGURATION

@dataclass
class FastF1Config:
    """FastF1 API configuration."""
    cache_dir: Path = CACHE_DIR
    default_year: int = 2025
    retry_attempts: int = 3
    retry_delay: float = 1.0
    timeout: int = 30
    cache_enabled: bool = True
    # Historical seasons supported by FastF1 API
    min_supported_year: int = 2018
    max_supported_year: int = 2025
    
    def get_supported_years(self) -> list:
        """Get list of supported historical seasons."""
        return list(range(self.min_supported_year, self.max_supported_year + 1))

FASTF1_CONFIG = FastF1Config()

# F1 2025 SEASON DATA

# Official 2025 F1 Calendar
F1_2025_CALENDAR: List[Dict] = [
    {'round': 1, 'name': 'Australian Grand Prix', 'location': 'Melbourne', 'country': 'Australia'},
    {'round': 2, 'name': 'Chinese Grand Prix', 'location': 'Shanghai', 'country': 'China'},
    {'round': 3, 'name': 'Japanese Grand Prix', 'location': 'Suzuka', 'country': 'Japan'},
    {'round': 4, 'name': 'Bahrain Grand Prix', 'location': 'Sakhir', 'country': 'Bahrain'},
    {'round': 5, 'name': 'Saudi Arabian Grand Prix', 'location': 'Jeddah', 'country': 'Saudi Arabia'},
    {'round': 6, 'name': 'Miami Grand Prix', 'location': 'Miami', 'country': 'USA'},
    {'round': 7, 'name': 'Emilia Romagna Grand Prix', 'location': 'Imola', 'country': 'Italy'},
    {'round': 8, 'name': 'Monaco Grand Prix', 'location': 'Monte Carlo', 'country': 'Monaco'},
    {'round': 9, 'name': 'Spanish Grand Prix', 'location': 'Barcelona', 'country': 'Spain'},
    {'round': 10, 'name': 'Canadian Grand Prix', 'location': 'Montreal', 'country': 'Canada'},
    {'round': 11, 'name': 'Austrian Grand Prix', 'location': 'Spielberg', 'country': 'Austria'},
    {'round': 12, 'name': 'British Grand Prix', 'location': 'Silverstone', 'country': 'Great Britain'},
    {'round': 13, 'name': 'Belgian Grand Prix', 'location': 'Spa', 'country': 'Belgium'},
    {'round': 14, 'name': 'Hungarian Grand Prix', 'location': 'Budapest', 'country': 'Hungary'},
    {'round': 15, 'name': 'Dutch Grand Prix', 'location': 'Zandvoort', 'country': 'Netherlands'},
    {'round': 16, 'name': 'Italian Grand Prix', 'location': 'Monza', 'country': 'Italy'},
    {'round': 17, 'name': 'Azerbaijan Grand Prix', 'location': 'Baku', 'country': 'Azerbaijan'},
    {'round': 18, 'name': 'Singapore Grand Prix', 'location': 'Marina Bay', 'country': 'Singapore'},
    {'round': 19, 'name': 'United States Grand Prix', 'location': 'Austin', 'country': 'USA'},
    {'round': 20, 'name': 'Mexico City Grand Prix', 'location': 'Mexico City', 'country': 'Mexico'},
    {'round': 21, 'name': 'São Paulo Grand Prix', 'location': 'Interlagos', 'country': 'Brazil'},
    {'round': 22, 'name': 'Las Vegas Grand Prix', 'location': 'Las Vegas', 'country': 'USA'},
    {'round': 23, 'name': 'Qatar Grand Prix', 'location': 'Lusail', 'country': 'Qatar'},
    {'round': 24, 'name': 'Abu Dhabi Grand Prix', 'location': 'Yas Marina', 'country': 'UAE'},
]

# Dictionary mapping display names to FastF1/Official names
F1_2025_RACE_NAMES: Dict[str, str] = {r['name']: r['name'] for r in F1_2025_CALENDAR}

# All races are completed as of Dec 2025
F1_2025_COMPLETED_RACES: List[str] = [r['name'] for r in F1_2025_CALENDAR]

# F1 Points System
TEAM_COLORS: Dict[str, str] = {
    # Main team names - 2025 Official Colors
    'Mercedes': '#00D2BE',
    'Red Bull': '#1E41FF',
    'Ferrari': '#DC0000',
    'McLaren': '#FF8700',
    'Aston Martin': '#006F62',
    'Alpine': '#0090FF',
    'Williams': '#005AFF',
    'Racing Bulls': '#F6E500',  # Yellow 2025 (formerly VCARB)
    'Kick Sauber': '#00E701',  # Neon Green Stake 2025
    'Haas': '#E6002B',
    
    # Alternative/Full team names
    'Mercedes-AMG': '#00D2BE',
    'Mercedes-AMG Petronas F1 Team': '#00D2BE',
    'Red Bull Racing': '#1E41FF',
    'Red Bull Racing Honda RBPT': '#1E41FF',
    'Scuderia Ferrari': '#DC0000',
    'Scuderia Ferrari HP': '#DC0000',
    'McLaren Formula 1 Team': '#FF8700',
    'McLaren Mercedes': '#FF8700',
    'Aston Martin Aramco': '#006F62',
    'Aston Martin Aramco Mercedes': '#006F62',
    'BWT Alpine F1 Team': '#0090FF',
    'Alpine Renault': '#0090FF',
    'Williams Racing': '#005AFF',
    'Williams Mercedes': '#005AFF',
    'Visa Cash App RB': '#F6E500',  # 2025 Yellow
    'VCARB': '#F6E500',  # Old name, 2025 Yellow
    'RB': '#F6E500',  # 2025 Yellow
    'Racing Bulls Honda RBPT': '#F6E500',  # 2025 Yellow
    'Stake F1 Team Kick Sauber': '#00E701',  # 2025 Neon Green
    'Sauber': '#00E701',  # 2025 Neon Green
    'Kick Sauber Ferrari': '#00E701',  # 2025 Neon Green
    'MoneyGram Haas F1 Team': '#E6002B',
    'Haas Ferrari': '#E6002B',
}

# DRIVER PROFILES (2025 F1 Season)

# Driver Profile Dictionary with Biographies
DRIVER_PROFILES: Dict[str, Dict] = {
    "Max Verstappen": {
        "number": 1,
        "country": "Netherlands",
        "debut": 2015,
        "image_url": "https://media.formula1.com/content/dam/fom-website/drivers/M/MAXVER01_Max_Verstappen/maxver01.png.transform/2col/image.png",
        "bio": "Four-time World Champion (2021-2024). Red Bull's leading force known for his aggressive yet precise driving style. Holds the record for most wins in a single season (19). Seeking a fifth consecutive title in 2025."
    },
    "Sergio Perez": {
        "number": 11,
        "country": "Mexico",
        "debut": 2011,
        "image_url": "https://media.formula1.com/content/dam/fom-website/drivers/S/SERPER01_Sergio_Perez/serper01.png.transform/2col/image.png",
        "bio": "The most successful Mexican driver in F1 history. Known as the 'King of the Streets' for his prowess on street circuits. Provides crucial experience and points for Red Bull's constructor campaign."
    },
    "Lewis Hamilton": {
        "number": 44,
        "country": "United Kingdom",
        "debut": 2007,
        "image_url": "https://media.formula1.com/content/dam/fom-website/drivers/L/LEWHAM01_Lewis_Hamilton/lewham01.png.transform/2col/image.png",
        "bio": "Seven-time World Champion making a historic move to Ferrari for 2025. Statistical G.O.A.T. of Formula 1 with over 100 wins and poles. Aiming to capture an elusive eighth title in red."
    },
    "Charles Leclerc": {
        "number": 16,
        "country": "Monaco",
        "debut": 2018,
        "image_url": "https://media.formula1.com/content/dam/fom-website/drivers/C/CHALEC01_Charles_Leclerc/chalec01.png.transform/2col/image.png",
        "bio": "Ferrari's homegrown hero. One of the fastest qualifiers in the sport's history. Now partnered with Hamilton, he faces his biggest internal challenge yet while chasing his first World Championship."
    },
    "George Russell": {
        "number": 63,
        "country": "United Kingdom",
        "debut": 2019,
        "image_url": "https://media.formula1.com/content/dam/fom-website/drivers/G/GEORUS01_George_Russell/georus01.png.transform/2col/image.png",
        "bio": "Now the team leader at Mercedes following Hamilton's departure. Known for his consistency and qualifying speed ('Mr. Saturday'). Looking to lead the Silver Arrows back to championship glory."
    },
    "Andrea Kimi Antonelli": {
        "number": 12,
        "country": "Italy",
        "debut": 2025,
        "image_url": "https://media.formula1.com/content/dam/fom-website/drivers/A/ANDANT01_Andrea_Kimi_Antonelli/andant01.png.transform/2col/image.png",
        "bio": "The highly anticipated rookie replacing Hamilton at Mercedes. Skipped F3 to fast-track his route to F1. A Mercedes junior prodigy with immense potential and expectation."
    },
    "Lando Norris": {
        "number": 4,
        "country": "United Kingdom",
        "debut": 2019,
        "image_url": "https://media.formula1.com/content/dam/fom-website/drivers/L/LANNOR01_Lando_Norris/lannor01.png.transform/2col/image.png",
        "bio": "McLaren's spearhead who fought for the 2024 title. immensely popular and blisteringly fast. Looking to convert his consistent podium form into a sustained championship challenge."
    },
    "Oscar Piastri": {
        "number": 81,
        "country": "Australia",
        "debut": 2023,
        "image_url": "https://media.formula1.com/content/dam/fom-website/drivers/O/OSCPIA01_Oscar_Piastri/oscpia01.png.transform/2col/image.png",
        "bio": "Sensational talent who proved himself a race winner in his sophomore year. Cool, calm, and collected. Forms one of the strongest lineups on the grid with Norris at McLaren."
    },
    "Fernando Alonso": {
        "number": 14,
        "country": "Spain",
        "debut": 2001,
        "image_url": "https://media.formula1.com/content/dam/fom-website/drivers/F/FERALO01_Fernando_Alonso/feralo01.png.transform/2col/image.png",
        "bio": "The grid's veteran double World Champion. Renowned for his unmatched racecraft and tenacity. Continues to defy age at Aston Martin, pushing the team towards the front of the field."
    },
    "Lance Stroll": {
        "number": 18,
        "country": "Canada",
        "debut": 2017,
        "image_url": "https://media.formula1.com/content/dam/fom-website/drivers/L/LANSTR01_Lance_Stroll/lanstr01.png.transform/2col/image.png",
        "bio": "Aston Martin driver entering his 9th season. A podium finisher and pole sitter who excels in wet conditions. Looking to silence critics with consistent performances alongside Alonso."
    },
    "Pierre Gasly": {
        "number": 10,
        "country": "France",
        "debut": 2017,
        "image_url": "https://media.formula1.com/content/dam/fom-website/drivers/P/PIEGAS01_Pierre_Gasly/piegas01.png.transform/2col/image.png",
        "bio": "Race winner and Alpine's team leader. Known for his emotional Monza win and strong recovery drives. Leads the French team's efforts to move up the midfield."
    },
    "Jack Doohan": {
        "number": 7,
        "country": "Australia",
        "debut": 2025,
        "image_url": "https://media.formula1.com/content/dam/fom-website/drivers/J/JACDOO01_Jack_Doohan/jacdoo01.png.transform/2col/image.png",
        "bio": "Alpine Academy graduate promoted to a race seat. Son of motorcycle legend Mick Doohan. Impressed in testing and simulator roles, now ready to prove his worth on track."
    },
    "Alexander Albon": {
        "number": 23,
        "country": "Thailand",
        "debut": 2019,
        "image_url": "https://media.formula1.com/content/dam/fom-website/drivers/A/ALEALB01_Alexander_Albon/alealb01.png.transform/2col/image.png",
        "bio": "Williams' dependable team leader. Has single-handedly dragged the team into points contention in recent years. Now paired with Sainz in a formidable Williams lineup."
    },
    "Carlos Sainz": {
        "number": 55,
        "country": "Spain",
        "debut": 2015,
        "image_url": "https://media.formula1.com/content/dam/fom-website/drivers/C/CARSAI01_Carlos_Sainz/carsai01.png.transform/2col/image.png",
        "bio": "Multiple race winner joining Williams from Ferrari. Known as the 'Smooth Operator' for his intelligent race management. A massive signing for Williams' rebuilding project."
    },
    "Yuki Tsunoda": {
        "number": 22,
        "country": "Japan",
        "debut": 2021,
        "image_url": "https://media.formula1.com/content/dam/fom-website/drivers/Y/YUKTSU01_Yuki_Tsunoda/yuktsu01.png.transform/2col/image.png",
        "bio": "Racing Bulls' fiery speedster. Has matured into a consistent points scorer while maintaining his aggressive edge. The undisputed leader of the Red Bull junior team."
    },
    "Liam Lawson": {
        "number": 30,
        "country": "New Zealand",
        "debut": 2023,
        "image_url": "https://media.formula1.com/content/dam/fom-website/drivers/L/LIALAW01_Liam_Lawson/lialaw01.png.transform/2col/image.png",
        "bio": "Finally secures a full-time seat at Racing Bulls after impressive cameos. A Red Bull junior with high expectations to challenge his teammate and eyeing a future Red Bull Racing seat."
    },
    "Nico Hulkenberg": {
        "number": 27,
        "country": "Germany",
        "debut": 2010,
        "image_url": "https://media.formula1.com/content/dam/fom-website/drivers/N/NICHUL01_Nico_Hulkenberg/nichul01.png.transform/2col/image.png",
        "bio": "The veteran moves to Sauber (Audi) to spearhead their transition. An expert qualifier and reliable points scorer. Brings immense experience to the Swiss team's factory project."
    },
    "Gabriel Bortoleto": {
        "number": 5,
        "country": "Brazil",
        "debut": 2025,
        "image_url": "https://media.formula1.com/content/dam/fom-website/drivers/G/GABBOR01_Gabriel_Bortoleto/gabbor01.png.transform/2col/image.png",
        "bio": "F2 Champion making his F1 debut with Sauber. A McLaren junior talent poached by Audi. Represents Brazil's return to the F1 grid with high hopes for the future."
    },
    "Esteban Ocon": {
        "number": 31,
        "country": "France",
        "debut": 2016,
        "image_url": "https://media.formula1.com/content/dam/fom-website/drivers/E/ESTOCO01_Esteban_Ocon/estoco01.png.transform/2col/image.png",
        "bio": "Race winner joining Haas for a fresh start. Known for his uncompromising wheel-to-wheel racing. Brings race-winning experience to the American team alongside a rookie."
    },
    "Oliver Bearman": {
        "number": 87,
        "country": "United Kingdom",
        "debut": 2024,
        "image_url": "https://media.formula1.com/content/dam/fom-website/drivers/O/OLIBEA01_Oliver_Bearman/olibea01.png.transform/2col/image.png",
        "bio": "Ferrari Academy star making his full-time debut with Haas. Stunned the world with his stand-in performance for Ferrari in 2024. A young talent with immense promise."
    }
}

# Enhanced Driver Details (birthdate, social media)
DRIVER_DETAILS: Dict[str, Dict] = {
    "Max Verstappen": {
        "birthdate": "1997-09-30",
        "birthplace": "Hasselt, Belgium",
        "height_cm": 181,
        "weight_kg": 72,
        "twitter": "@Max33Verstappen",
        "instagram": "@maxverstappen1",
        "titles": 4,
        "wins": 62,
        "poles": 40,
        "podiums": 110,
        "fastest_laps": 32
    },
    "Lewis Hamilton": {
        "birthdate": "1985-01-07",
        "birthplace": "Stevenage, UK",
        "height_cm": 174,
        "weight_kg": 73,
        "twitter": "@LewisHamilton",
        "instagram": "@lewishamilton",
        "titles": 7,
        "wins": 104,
        "poles": 104,
        "podiums": 201,
        "fastest_laps": 67
    },
    "Charles Leclerc": {
        "birthdate": "1997-10-16",
        "birthplace": "Monte Carlo, Monaco",
        "height_cm": 180,
        "weight_kg": 70,
        "twitter": "@Charles_Leclerc",
        "instagram": "@charles_leclerc",
        "titles": 0,
        "wins": 7,
        "poles": 26,
        "podiums": 38,
        "fastest_laps": 9
    },
    "Lando Norris": {
        "birthdate": "1999-11-13",
        "birthplace": "Bristol, UK",
        "height_cm": 170,
        "weight_kg": 69,
        "twitter": "@LandoNorris",
        "instagram": "@landonorris",
        "titles": 0,
        "wins": 4,
        "poles": 9,
        "podiums": 26,
        "fastest_laps": 8
    },
    "Oscar Piastri": {
        "birthdate": "2001-04-06",
        "birthplace": "Melbourne, Australia",
        "height_cm": 178,
        "weight_kg": 70,
        "twitter": "@OscarPiastri",
        "instagram": "@oscarpiastri",
        "titles": 0,
        "wins": 2,
        "poles": 2,
        "podiums": 11,
        "fastest_laps": 3
    },
    "Carlos Sainz": {
        "birthdate": "1994-09-01",
        "birthplace": "Madrid, Spain",
        "height_cm": 178,
        "weight_kg": 66,
        "twitter": "@Carlossainz55",
        "instagram": "@carlossainz55",
        "titles": 0,
        "wins": 4,
        "poles": 6,
        "podiums": 25,
        "fastest_laps": 5
    },
    "George Russell": {
        "birthdate": "1998-02-15",
        "birthplace": "King's Lynn, UK",
        "height_cm": 185,
        "weight_kg": 70,
        "twitter": "@GeorgeRussell63",
        "instagram": "@georgerussell63",
        "titles": 0,
        "wins": 3,
        "poles": 5,
        "podiums": 16,
        "fastest_laps": 8
    },
    "Fernando Alonso": {
        "birthdate": "1981-07-29",
        "birthplace": "Oviedo, Spain",
        "height_cm": 171,
        "weight_kg": 68,
        "twitter": "@alo_oficial",
        "instagram": "@fernandoalo_oficial",
        "titles": 2,
        "wins": 32,
        "poles": 22,
        "podiums": 106,
        "fastest_laps": 24
    },
    "Sergio Perez": {
        "birthdate": "1990-01-26",
        "birthplace": "Guadalajara, Mexico",
        "height_cm": 173,
        "weight_kg": 63,
        "twitter": "@SChecoPerez",
        "instagram": "@schecoperez",
        "titles": 0,
        "wins": 6,
        "poles": 3,
        "podiums": 39,
        "fastest_laps": 11
    },
    "Pierre Gasly": {
        "birthdate": "1996-02-07",
        "birthplace": "Rouen, France",
        "height_cm": 177,
        "weight_kg": 70,
        "twitter": "@PierreGASLY",
        "instagram": "@pierregasly",
        "titles": 0,
        "wins": 1,
        "poles": 0,
        "podiums": 4,
        "fastest_laps": 3
    },
    "Yuki Tsunoda": {
        "birthdate": "2000-05-11",
        "birthplace": "Sagamihara, Japan",
        "height_cm": 159,
        "weight_kg": 54,
        "twitter": "@yukitsunoda07",
        "instagram": "@yukitsunoda0511",
        "titles": 0,
        "wins": 0,
        "poles": 0,
        "podiums": 0,
        "fastest_laps": 0
    },
    "Alex Albon": {
        "birthdate": "1996-03-23",
        "birthplace": "London, UK",
        "height_cm": 186,
        "weight_kg": 74,
        "twitter": "@alex_albon",
        "instagram": "@alex_albon",
        "titles": 0,
        "wins": 0,
        "poles": 0,
        "podiums": 2,
        "fastest_laps": 0
    }
}

# LOGGING CONFIGURATION

def setup_logging(
    level: int = logging.INFO,
    log_file: Optional[str] = None,
    format_string: Optional[str] = None
) -> logging.Logger:
    """
    Set up logging configuration.
    
    Args:
        level: Logging level (default: INFO)
        log_file: Optional log file path
        format_string: Optional custom format string
    
    Returns:
        Configured logger instance
    """
    if format_string is None:
        format_string = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    # Create logger
    logger = logging.getLogger('f1_visualization')
    logger.setLevel(level)
    
    # Remove existing handlers
    logger.handlers = []
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(logging.Formatter(format_string))
    logger.addHandler(console_handler)
    
    # File handler (optional)
    if log_file:
        log_path = LOGS_DIR / log_file
        file_handler = logging.FileHandler(log_path)
        file_handler.setLevel(level)
        file_handler.setFormatter(logging.Formatter(format_string))
        logger.addHandler(file_handler)
    
    return logger

# Default logger
logger = setup_logging()

# STREAMLIT CONFIGURATION

@dataclass
class StreamlitConfig:
    """Streamlit dashboard configuration."""
    page_title: str = "F1 2025 Season Dashboard"
    page_icon: str = "F1"
    layout: str = "wide"
    theme_primary_color: str = "#FF1E00"
    theme_background_color: str = "#0E1117"
    cache_ttl: int = 3600  # 1 hour

STREAMLIT_CONFIG = StreamlitConfig()

# DRIVER DATA MERGE

# Merge detailed stats into main profiles
for driver, details in DRIVER_DETAILS.items():
    # Normalize name matching (e.g. Alex Albon vs Alexander Albon)
    matched_name = None
    for profile_name in DRIVER_PROFILES.keys():
        if driver.split()[-1] == profile_name.split()[-1]: # Match by last name
            matched_name = profile_name
            break
            
    if matched_name:
        DRIVER_PROFILES[matched_name].update(details)

# Configure display names for social media
SOCIAL_MEDIA_CONFIG = {
    'twitter': {'icon': 'twitter', 'url_prefix': 'https://twitter.com/'},
    'instagram': {'icon': 'instagram', 'url_prefix': 'https://instagram.com/'}
}

if __name__ == '__main__':
    # Print configuration summary
    print("F1 Visualization Configuration")
    print("=" * 50)
    print(f"Project Root: {PROJECT_ROOT}")
    print(f"Data Directory: {DATA_DIR}")
    print(f"Cache Directory: {CACHE_DIR}")
    print(f"2025 Calendar: {len(F1_2025_CALENDAR)} races")
    print("\nConfig loaded successfully!")
