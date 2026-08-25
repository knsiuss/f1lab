# -*- coding: utf-8 -*-
"""
config.py
Application constants and configuration.

:copyright: (c) 2025 F1 Analytics
:license: MIT
"""

import sys
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional
from dataclasses import dataclass

# Ensure src/ is importable no matter how this module is reached
# (as `config` from a page, or as `src.config` from the test suite).
sys.path.insert(0, str(Path(__file__).parent))

from seasons import discover_available_years  # noqa: E402


def _current_calendar_year() -> int:
    """The year we are in right now; used to stream the in-progress season."""
    return datetime.now().year


# Path Configuration

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent

# Data directories
DATA_DIR = PROJECT_ROOT / 'data'
CACHE_DIR = PROJECT_ROOT / 'cache'
LOGS_DIR = PROJECT_ROOT / 'logs'


def _ensure_dirs():
    """Create data/cache/logs directories if they don't exist."""
    for directory in [DATA_DIR, CACHE_DIR, LOGS_DIR]:
        directory.mkdir(exist_ok=True)

# DATA FILE PATHS

# Season data files are located dynamically per year (see shared.load_race_data
# and loader), so there is no per-year filename config here.

# FASTF1 CONFIGURATION

@dataclass
class FastF1Config:
    """FastF1 API configuration."""
    cache_dir: Path = PROJECT_ROOT / 'f1_cache'
    default_year: int = 2025
    # Historical seasons supported by FastF1 API
    min_supported_year: int = 2018
    max_supported_year: int = 2025

    def get_supported_years(self) -> list:
        """Get list of supported historical seasons (includes the live season).

        Streaming is deliberately inclusive of the in-progress (current calendar)
        year so FastF1-backed pages can pull sessions online even before that
        season's result CSVs exist; CSV-backed pages degrade gracefully.
        """
        upper = max(self.max_supported_year, _current_calendar_year())
        return list(range(self.min_supported_year, upper + 1))

FASTF1_CONFIG = FastF1Config()


def _apply_season_discovery() -> None:
    """Derive season defaults from the CSVs actually present in data/.

    Seasons are discovered rather than hardcoded: dropping a
    Formula1_{year}Season_RaceResults.csv file into data/ is all it takes to
    opt the app into that year. We fall back to a seed of 2025 when the data
    directory is empty (e.g. a fresh checkout before CSVs are placed).
    """
    years = discover_available_years(DATA_DIR)
    if years:
        FASTF1_CONFIG.default_year = years[-1]
        # The API still supports later seasons once their data is present.
        FASTF1_CONFIG.max_supported_year = max(
            FASTF1_CONFIG.max_supported_year, years[-1]
        )
    else:
        FASTF1_CONFIG.default_year = 2025
        FASTF1_CONFIG.max_supported_year = 2025


_apply_season_discovery()

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

# STRATEGY SIMULATION MODEL CONFIGURATION

# Central place to tune the physics knobs used by the strategy simulator in
# model.py. Kept as data so behaviour is adjustable without editing code.
MODEL_CONFIG: Dict[str, object] = {
    # Per-lap time loss (seconds) due to tyre wear, by compound.
    'degradation_rates': {
        'SOFT': 0.12,    # High deg
        'MEDIUM': 0.08,  # Medium deg
        'HARD': 0.04,    # Low deg
        'INTER': 0.05,
        'WET': 0.05,
        'DEFAULT': 0.08,
    },
    # Lap-over-lap time gain from fuel burn (negative = faster). Avg ~0.06s.
    'fuel_gain_per_lap': -0.06,
    # Average time lost in the pits (seconds).
    'pit_loss_sec': 22.0,
    'default_base_lap_time': 90.0,
    'default_total_laps': 57,
    # Relative per-lap pace advantage of a chaser on fresher tyres (seconds).
    'fresh_tyre_advantage': -0.5,
}

# SETUP / AERO FINGERPRINT ANALYSIS (src/setupfingerprint.py)

# Methodology knobs for deriving a setup proxy from public telemetry: corner
# detection follows the smoothed-speed-minima approach; buckets split corners
# into mechanical-grip vs aerodynamic regimes. Tunable in one place.
AERO_CONFIG: Dict[str, object] = {
    'smooth_window': 5,           # rolling-mean window over the speed trace (samples)
    'min_speed_drop_kmh': 15.0,   # min braking into an apex to count as a corner
    'slow_max_kmh': 120.0,        # apex <= this -> slow corner (mechanical grip)
    'medium_max_kmh': 170.0,      # apex <= this -> medium corner
    'top_trap_laps': 3,           # median of K best clean SpeedST laps = top-speed proxy
    'min_corners_total': 4,       # fewer detected corners -> cornering block reported as None
    'brake_window_samples': 12,   # samples before apex scanned for peak deceleration
    'entry_scan_samples': 60,     # samples before apex scanned for the braking entry point
}

# PIT STOP MEASUREMENT SANITY

# A measured PitIn->PitOut duration outside this window is not one continuous
# stop (scramble across laps, missing data), so it falls back to the estimate.
PIT_STOP_CONFIG: Dict[str, float] = {
    'min_measured_sec': 2.0,
    'max_measured_sec': 120.0,
}

# SESSION LIVE-WINDOW

# Hours around a race weekend's EventDate treated as "live" for future-race
# gating (shared by the Analysis Center and Live & Replay pages).
SESSION_GUARD_HOURS: int = 48

# VISUALISATION TIMING

VIZ_CONFIG: Dict[str, float] = {
    'replay_slider_ms': 250,      # per-frame duration when scrubbing the replay slider
    'replay_play_ms': 400,        # per-frame duration during autoplay
}

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

if __name__ == '__main__':
    # Print configuration summary
    print("F1 Visualization Configuration")
    print("=" * 50)
    print(f"Project Root: {PROJECT_ROOT}")
    print(f"Data Directory: {DATA_DIR}")
    print(f"Cache Directory: {CACHE_DIR}")
    print(f"Discovered seasons: {discover_available_years(DATA_DIR)}")
    print(f"Supported season range: {FASTF1_CONFIG.get_supported_years()}")
    print("\nConfig loaded successfully!")
