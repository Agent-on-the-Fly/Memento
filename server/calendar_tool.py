import json
from datetime import datetime
from mcp.server.fastmcp import FastMCP

# --------------------------------------------------------------------------- #
#  FastMCP server instance
# --------------------------------------------------------------------------- #

mcp = FastMCP("calendar")

# --------------------------------------------------------------------------- #
#  Setup
# --------------------------------------------------------------------------- #

CALENDAR_FILE = "calendar.json"

def load_events():
    """Loads events from the JSON file."""
    try:
        with open(CALENDAR_FILE, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return []

def save_events(events):
    """Saves events to the JSON file."""
    with open(CALENDAR_FILE, "w") as f:
        json.dump(events, f, indent=4)

# --------------------------------------------------------------------------- #
#  Tools
# --------------------------------------------------------------------------- #

@mcp.tool()
async def add_event(summary: str, start_datetime: str, end_datetime: str, description: str = "") -> str:
    """
    Adds an event to the calendar. Datetimes should be in ISO format (YYYY-MM-DD HH:MM:SS).
    :param summary: The summary or title of the event.
    :param start_datetime: The start date and time of the event.
    :param end_datetime: The end date and time of the event.
    :param description: A description of the event.
    :return: A confirmation message.
    """
    events = load_events()
    event = {
        "summary": summary,
        "start": start_datetime,
        "end": end_datetime,
        "description": description,
    }
    events.append(event)
    save_events(events)
    return "Event added successfully."


@mcp.tool()
async def list_events(start_datetime: str, end_datetime: str) -> list:
    """
    Lists events within a given time range. Datetimes should be in ISO format (YYYY-MM-DD HH:MM:SS).
    :param start_datetime: The start of the time range.
    :param end_datetime: The end of the time range.
    :return: A list of events within the specified range.
    """
    events = load_events()
    start = datetime.fromisoformat(start_datetime)
    end = datetime.fromisoformat(end_datetime)

    upcoming_events = []
    for event in events:
        event_start = datetime.fromisoformat(event["start"])
        if start <= event_start < end:
            upcoming_events.append(event)

    return upcoming_events

# --------------------------------------------------------------------------- #
#  Entrypoint
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    mcp.run(transport="stdio")
