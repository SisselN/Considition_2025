# Kör den tränade DQN-modell mot Considition live API

import torch
import time
import json
from baseline_agent.client import ConsiditionClient
from env import ConsiditionEnv
from train_api_sim_4maps import DQN
import os
from dotenv import load_dotenv

load_dotenv()

def state_to_tensor(state):
    """Konverterar state-lista till tensor för modellinput."""
    return torch.tensor(state, dtype=torch.float32).unsqueeze(0)

# Konfiguration
API_KEY = os.getenv("API_KEY")
BASE_URL = os.getenv("BASE_URL")
MAP_NAME = "Pistonia"
MODEL_PATH = "dqn_api_multi_map_finetuned2.pth"
NUM_ACTIONS = 4

# Laddar modell.
policy_net = DQN(5, NUM_ACTIONS)
policy_net.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
policy_net.eval()
print(f"Laddad modell: {MODEL_PATH}")

def main():
    print(f"\nKör tränad modell på {MAP_NAME}...")

    # Sätter upp klient och miljö.
    client = ConsiditionClient(BASE_URL, API_KEY)
    env = ConsiditionEnv(BASE_URL, API_KEY, MAP_NAME)
    state = env.reset()
    total_reward = 0.0
    tick = 0

    while tick < env.total_ticks:
        actions = []
        for s in state:
            with torch.no_grad():
                q_vals = policy_net(state_to_tensor(s))
                action = int(torch.argmax(q_vals).item())
                actions.append(action)

        # Bygger giltiga rekommendationer för API:et.
        customer_recommendations = []
        nodes = env.map_obj.get("nodes", []) or []
        valid_node_ids = {str(n["id"]) for n in nodes if "id" in n}

        for node in nodes:
            for customer in node.get("customers", []) or []:
                cust_id = str(customer["id"])
                action = actions.pop(0) if actions else 0
                if action == 0:
                    continue

                recommendation = None
                if action == 1:
                    nearest_id, _ = env._find_nearest_station(node)
                    if nearest_id and str(nearest_id) in valid_node_ids:
                        recommendation = {
                            "nodeId": str(nearest_id),
                            "chargeTo": 0.9
                        }
                elif action == 2:
                    _, fastest_id = env._find_nearest_station(node)
                    if fastest_id and str(fastest_id) in valid_node_ids:
                        recommendation = {
                            "nodeId": str(fastest_id),
                            "chargeTo": 0.8
                        }
                elif action == 3:
                    if node.get("target", {}).get("Type") == "ChargingStation" and str(node["id"]) in valid_node_ids:
                        recommendation = {
                            "nodeId": str(node["id"]),
                            "chargeTo": 0.95
                        }

                if recommendation:
                    customer_recommendations.append({
                        "customerId": cust_id,
                        "chargingRecommendations": [recommendation]
                    })

        # Tar bort dubbletter.
        seen = set()
        unique_recommendations = []
        for rec in customer_recommendations:
            if rec["customerId"] not in seen:
                seen.add(rec["customerId"])
                unique_recommendations.append(rec)

        tick_payload = {
            "tick": tick,
            "customerRecommendations": unique_recommendations
        }
        input_payload = {"mapName": MAP_NAME, "ticks": [tick_payload]}

        print(f"\n--- Tick {tick} ---")
        print(f"Skickar {len(unique_recommendations)} giltiga rekommendationer...")

        # Skickar requests till API:et.
        try:
            response = client.post_game(input_payload)
        except Exception as e:
            print("Fel:", e)
            print("Hoppar över tick på grund av ogiltig payload.\n")
            tick += 1
            time.sleep(0.25)
            continue

        # Debug-utskrift av API-svar.
        print("API-svar (truncated):")
        print(json.dumps(response, indent=2)[:600])

        # Beräknar reward.
        reward = (
            response.get("totalScore", 0)
            or response.get("score", 0)
            or response.get("customerCompletionScore", 0)
            or 0
        )
        total_reward += reward
        print(f"Tick {tick}: Reward {reward:.2f} | Total {total_reward:.2f}")

        # Förbereder nästa tick.
        env.map_obj = response.get("map", env.map_obj)
        state = env.get_customer_features()
        tick += 1
        time.sleep(0.25)

    print("\nFärdig!")
    print(f"Total Reward: {total_reward:.2f}")

if __name__ == "__main__":
    main()
