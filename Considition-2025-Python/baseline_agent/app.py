import sys
import time
from client import ConsiditionClient
import os

def should_move_on_to_next_tick(response):
    return True

def generate_customer_recommendations(map_obj, current_tick):
    """Generate recommendations for each customer with their charging stops"""
    
    recommendations = []
    nodes = map_obj.get("nodes", []) or []
    # Collect charging stations with simple score (speed + availability)
    stations = []
    for node in nodes:
        target = node.get("target") or {}
        # print(f"Target: {node.get('target')}")
        if target.get("Type") == "ChargingStation":
            speed = float(target.get("chargeSpeedPerCharger", 0) or 0)
            # print(f"Speed: {speed}")
            avail = int(target.get("amountOfAvailableChargers", 0) or 0)
            # print(f"Avail: {avail}")
            stations.append({
                "nodeId": node.get("id"),
                "speed": speed,
                "available": avail
            })

    if not stations:
        return recommendations

    # sort stations by speed and availability (best first)
    stations.sort(key=lambda s: (s["speed"], s["available"]), reverse=True)

    def make_targets(current_frac: float):
        # progressive target fractions (0..1), only keep those > current_frac
        candidate = [0.33, 0.6, 0.95]
        targets = [t for t in candidate if t > current_frac + 1e-6]
        return targets[:3]

    # For each customer, build chargingRecommendations (nodeId + chargeTo (float))
    for node in nodes:
        for customer in node.get("customers", []) or []:
            cust_id = customer.get("id")
            # protect against malformed values
            current_charge = float(customer.get("chargeRemaining", 0) or 0)
            max_charge = float(customer.get("maxCharge", 1) or 1)
            # current fraction of charge (0..1)
            current_frac = current_charge / max_charge if max_charge > 0 else 0.0

            # only recommend if below threshold (tweakable)
            if current_frac >= 0.8:
                continue

            targets = make_targets(current_frac)
            if not targets:
                continue

            # choose top stations (one per target) but skip stations with no available chargers
            chosen = []
            si = 0
            for target in targets:
                # find next station with available chargers (rotate through list)
                found = None
                for _ in range(len(stations)):
                    s = stations[si % len(stations)]
                    si += 1
                    if s["available"] > 0:
                        found = s
                        break
                if not found:
                    # if no station currently available, still offer the best station (can be 0 available)
                    found = stations[0]
                # append recommendation (chargeTo as fraction)
                chosen.append({
                    "nodeId": found["nodeId"],
                    "chargeTo": float(round(target, 6))
                })

            if chosen:
                recommendations.append({
                    "customerId": cust_id,
                    "chargingRecommendations": chosen
                })

    return recommendations


def generate_tick(map_obj, current_tick):
    return {
        "tick": current_tick,
        "customerRecommendations": generate_customer_recommendations(map_obj, current_tick),
    }

def extract_states_from_map(map_obj):
    states = []
    nodes = map_obj.get("nodes", []) or []

    for node in nodes:
        for customer in node.get("customers", []) or []:
            # Simple state extraction example
            state = [
                float(customer.get("chargeRemaining", 0) or 0),
                float(customer.get("maxCharge", 1) or 1),
                float(customer.get("distanceToTarget", 0) or 0),
            ]
            states.append((customer.get("id"), state))
    return states

def main():
    api_key = os.getenv("API_KEY")
    # base_url = "http://localhost:8080"
    base_url = os.getenv("BASE_URL")
    map_name = "Batterytown"

    client = ConsiditionClient(base_url, api_key)
    map_obj = client.get_map(map_name)


    if not map_obj:
        print("Failed to fetch map!")
        sys.exit(1)

    final_score = 0
    good_ticks = []

    current_tick = generate_tick(map_obj, 0)
    input_payload = {
        "mapName": map_name,
        "ticks": [current_tick],
    }

    total_ticks = int(map_obj.get("ticks", 0))

    for i in range(total_ticks):
        while True:
            print(f"Playing tick: {i} with input: {input_payload}")
            start = time.perf_counter()
            game_response = client.post_game(input_payload)
            elapsed_ms = (time.perf_counter() - start) * 1000
            print(f"Tick {i} took: {elapsed_ms:.2f}ms")

            if not game_response:
                print("Got no game response")
                sys.exit(1)

            # Sum the scores directly (assuming they are numbers)
            final_score = (
                game_response.get("customerCompletionScore", 0)
                + game_response.get("kwhRevenue", 0)
                + game_response.get("score", 0)
            )

            if should_move_on_to_next_tick(game_response):
                good_ticks.append(current_tick)
                updated_map = game_response.get("map", map_obj) or map_obj
                current_tick = generate_tick(updated_map, i + 1)
                input_payload = {
                    "mapName": map_name,
                    #"playToTick": i + 1, # behövs för lokal miljö
                    "ticks": [*good_ticks, current_tick],
                }
                break

            updated_map = game_response.get("map", map_obj) or map_obj
            current_tick = generate_tick(updated_map, i)
            input_payload = {
                "mapName": map_name,
                "playToTick": i,
                "ticks": [*good_ticks, current_tick],
            }

    print(f"Final score: {final_score}")

if __name__ == "__main__":
    main()