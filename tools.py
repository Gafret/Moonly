import datetime
from zoneinfo import ZoneInfo

from langchain_core.tools import tool


@tool(parse_docstring=True)
def get_current_time() -> dict:
    """Returns 'utc_time', 'local_time', 'utc_hour', 'local_hour'."""

    local = datetime.datetime.now(tz=ZoneInfo("Europe/Amsterdam"))
    utc = datetime.datetime.now(datetime.timezone.utc)

    return {
        "utc_time": str(utc),
        "utc_hour": utc.hour,
        "local": str(local),
        "local_hour": local.hour,
    }


@tool(parse_docstring=True)
def recommend_refreshment(local_hour: int) -> dict:
    """Given 'local_hour' suggests what user should eat and drink.

    Args:
        local_hour: local hour.
    """

    if local_hour < 17:
        return {"drink": "coffee", "snack": "cookies"}
    else:
        return {"drink": "kefir", "snack": "cookies"}