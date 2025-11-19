# client.py
import requests

class ConsiditionClient:
    def __init__(self, base_url: str, api_key: str):
        self.base_url = base_url
        self.api_key = api_key
        self.headers = {"x-api-key": self.api_key, "Content-Type": "application/json"}
    
    def post_game(self, data: object):
        return self.request("POST", "/api/game", json=data)

    def get_map(self, map_name: str, seed=None):
        return self.request("GET", "/api/map", params={"mapName": map_name})

    def request(self, method: str, endpoint: str, **kwargs):
        url = f"{self.base_url}{endpoint}"
        try:
            response = requests.request(method, url, headers=self.headers, verify=False, **kwargs)
            response.raise_for_status()
            # Try to parse JSON, but fall back to text if not JSON
            try:
                return response.json()
            except Exception:
                return {"__raw_text__": response.text}
        except requests.exceptions.HTTPError as e:
            # Log helpful debug info: status + body
            body = None
            try:
                body = response.text
            except Exception:
                body = "<no body>"
            print("❌ HTTPError:", e)
            print("URL:", url)
            print("Status code:", getattr(response, "status_code", None))
            print("Response body:\n", body)
            # Re-raise so callers can handle / stop execution
            raise
        except Exception as e:
            print("❌ Request failed:", e)
            raise
