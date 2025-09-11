import requests
from geopy.geocoders import Nominatim
from mcp.server.fastmcp import FastMCP
import argparse
from interpreters.logger import get_logger

# --------------------------------------------------------------------------- #
#  Logger setup
# --------------------------------------------------------------------------- #
logger = get_logger(__name__)

# --------------------------------------------------------------------------- #
#  FastMCP server instance
# --------------------------------------------------------------------------- #

mcp = FastMCP("weather")

# --------------------------------------------------------------------------- #
#  Setup
# --------------------------------------------------------------------------- #

geolocator = Nominatim(user_agent="memento_agent")
WEATHER_API_URL = "https://api.open-meteo.com/v1/forecast"

def get_coords(location: str):
    """Get latitude and longitude for a location string."""
    loc = geolocator.geocode(location)
    if not loc:
        raise ValueError(f"Could not find coordinates for location: {location}")
    return {"latitude": loc.latitude, "longitude": loc.longitude}

# --------------------------------------------------------------------------- #
#  Tools
# --------------------------------------------------------------------------- #

@mcp.tool()
async def get_current_weather(location: str) -> dict:
    """
    Gets the current weather for a specified location.
    :param location: The city and state, or country, e.g., "San Francisco, CA".
    :return: A dictionary with the current weather data.
    """
    coords = get_coords(location)
    params = {
        "latitude": coords["latitude"],
        "longitude": coords["longitude"],
        "current_weather": "true",
    }
    response = requests.get(WEATHER_API_URL, params=params)
    response.raise_for_status()
    return response.json()["current_weather"]


@mcp.tool()
async def get_weather_forecast(location: str, days: int = 3) -> dict:
    """
    Gets the weather forecast for a location for a number of days.
    :param location: The city and state, or country, e.g., "San Francisco, CA".
    :param days: The number of days to forecast (1-16).
    :return: A dictionary with the weather forecast data.
    """
    if not 1 <= days <= 16:
        raise ValueError("Forecast days must be between 1 and 16.")
    coords = get_coords(location)
    params = {
        "latitude": coords["latitude"],
        "longitude": coords["longitude"],
        "forecast_days": days,
        "daily": "weathercode,temperature_2m_max,temperature_2m_min,sunrise,sunset",
    }
    response = requests.get(WEATHER_API_URL, params=params)
    response.raise_for_status()
    return response.json()["daily"]

# --------------------------------------------------------------------------- #
#  Entrypoint
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set the logging level.",
    )
    args, _ = parser.parse_known_args()

    log_level = args.log_level.upper()
    logger.setLevel(log_level)
    for handler in logger.handlers:
        handler.setLevel(log_level)

    mcp.run(transport="stdio")
