# Script för att dumpa kartdata från Considition API till JSON-fil.

from env import ConsiditionEnv
import json
import os
from dotenv import load_dotenv

load_dotenv()

BASE_URL = os.getenv("BASE_URL")
API_KEY = os.getenv("API_KEY")
MAP_NAME = "Windcity"  # Välj karta att dumpa.

# Hämtar kartdata från Considition API
env = ConsiditionEnv(BASE_URL, API_KEY, MAP_NAME)

map_data = env.map_obj

# Sparar kartdata till JSON-fil
out_file = f"map_dump_{MAP_NAME}.json"
with open(out_file, "w") as f:
    json.dump(map_data, f, indent=2)

print(f"Sparad {out_file}")
