import json
import streamlit as st
import requests
from openai import OpenAI

st.title("Lab 5 The “What to Wear” Bot")
st.write(
    "Enter a city to get weather-based clothing suggestions and ideas for "
    "outdoor activities."
)


def get_current_weather(location, api_key, units="imperial"):
    """Fetch current weather for a location from OpenWeatherMap.
    location: e.g. 'Syracuse, NY, US' or 'Lima, Peru'
    Returns dict with temperature, feels_like, temp_min, temp_max, humidity, description, location.
    """
    url = (
        "https://api.openweathermap.org/data/2.5/weather"
        f"?q={location}&appid={api_key}&units={units}"
    )
    response = requests.get(url)
    if response.status_code == 401:
        raise Exception("Authentication failed: Invalid API key (401 Unauthorized)")
    if response.status_code == 404:
        error_message = response.json().get("message", "City not found")
        raise Exception(f"404 error: {error_message}")
    data = response.json()
    temp = data["main"]["temp"]
    feels_like = data["main"]["feels_like"]
    temp_min = data["main"]["temp_min"]
    temp_max = data["main"]["temp_max"]
    humidity = data["main"]["humidity"]
    description = data["weather"][0]["description"] if data.get("weather") else ""
    return {
        "location": location,
        "temperature": round(temp, 2),
        "feels_like": round(feels_like, 2),
        "temp_min": round(temp_min, 2),
        "temp_max": round(temp_max, 2),
        "humidity": round(humidity, 2),
        "description": description,
    }


# OpenAI and OpenWeatherMap API keys from secrets
if "openai_api_key" not in st.secrets:
    st.error("Add `openai_api_key` to `.streamlit/secrets.toml`.")
    st.stop()
if "openweathermap_api_key" not in st.secrets:
    st.error("Add `openweathermap_api_key` to `.streamlit/secrets.toml`.")
    st.stop()

openai_client = OpenAI(api_key=st.secrets["openai_api_key"])
weather_api_key = st.secrets["openweathermap_api_key"]

# Tool definition for OpenAI (get_current_weather)
weather_tool = {
    "type": "function",
    "function": {
        "name": "get_current_weather",
        "description": "Get the current weather for a given location. Use when the user asks about weather, what to wear, or outdoor activities for a place.",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "City and optional state/country, e.g. 'Syracuse, NY, US' or 'Lima, Peru'.",
                },
            },
            "required": ["location"],
        },
    },
}

# User inputs a city (not a chat); run the bot on button click
city = st.text_input(
    "City",
    placeholder="e.g. Syracuse, NY, US or Lima, Peru",
    key="lab5_city",
)
run_bot = st.button("Get clothing & activity suggestions")
if not city:
    st.info("Enter a city above, then click **Get clothing & activity suggestions**.")
    st.stop()
if not run_bot:
    st.stop()

# Default location when tool is called without one
DEFAULT_LOCATION = "Syracuse, NY"

messages = [
    {
        "role": "system",
        "content": (
            "You help users decide what to wear and what to do outdoors based on weather. "
            "When you need current weather, use the get_current_weather tool. "
            "If no location is provided, use Syracuse, NY."
        ),
    },
    {
        "role": "user",
        "content": f"What should I wear today and what outdoor activities do you suggest for {city}?",
    },
]

# First call: model may request weather via tool
response = openai_client.chat.completions.create(
    model="gpt-4o-mini",
    messages=messages,
    tools=[weather_tool],
    tool_choice="auto",
)

choice = response.choices[0]
if not choice.message.tool_calls:
    # No tool call; show the model’s reply (e.g. no weather needed)
    st.markdown(choice.message.content or "No response.")
    st.stop()

# Exactly one tool call expected: get_current_weather
tool_call = choice.message.tool_calls[0]
args = json.loads(tool_call.function.arguments)
location = (args.get("location") or city or DEFAULT_LOCATION).strip() or DEFAULT_LOCATION

with st.spinner(f"Fetching weather for {location}..."):
    try:
        weather = get_current_weather(location, weather_api_key)
    except Exception as e:
        st.error(str(e))
        st.stop()

# Second call: send weather to LLM and ask for clothing + activities
weather_text = (
    f"Current weather for {weather['location']}: "
    f"temperature {weather['temperature']}°F (feels like {weather['feels_like']}°F), "
    f"min {weather['temp_min']}°F, max {weather['temp_max']}°F, "
    f"humidity {weather['humidity']}%. "
    f"Conditions: {weather.get('description', 'N/A')}."
)
follow_up_messages = [
    *messages,
    choice.message,
    {
        "role": "tool",
        "tool_call_id": tool_call.id,
        "content": weather_text,
    },
]
follow_up_messages.append(
    {
        "role": "user",
        "content": (
            "Using this weather, suggest appropriate clothes to wear today and "
            "outdoor activities that fit the conditions. Be concise and practical."
        ),
    }
)

final = openai_client.chat.completions.create(
    model="gpt-4o-mini",
    messages=follow_up_messages,
)
st.markdown(final.choices[0].message.content)
