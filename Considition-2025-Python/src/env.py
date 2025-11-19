# Enviroment wrapper för Considition API

import copy
import math
from baseline_agent.client import ConsiditionClient
from math import hypot

class ConsiditionEnv:
    """
    Environment wrapper:
    - mappar agentens diskreta action per kund till pathTo (gå till station)
      eller chargeTo (ladda vid station)
    - ger sammansatt reward med server-score-delta, laddningsdelta och bonus när kund försvinner
    """

    def __init__(self, base_url, api_key, map_name, seed=None):
        self.client = ConsiditionClient(base_url, api_key)
        self.map_name = map_name
        self.seed = seed
        self.reset()

    def reset(self, seed_offset=0):
        effective_seed = self.seed + seed_offset if self.seed is not None else None
        self.map_obj = self.client.get_map(self.map_name, effective_seed) if effective_seed else self.client.get_map(self.map_name)
        self.total_ticks = int(self.map_obj.get("ticks", 0))
        self.ticks_sent = []
        self.current_tick = 0
        self.last_score = 0.0
        self.prev_customers = self._flatten_customers(self.map_obj)
        self.node_by_id = {n["id"]: n for n in self.map_obj.get("nodes", [])}
        return self.get_customer_features()

    def _flatten_customers(self, map_obj):
        d = {}
        for node in map_obj.get("nodes", []) or []:
            for c in node.get("customers", []) or []:
                d[c["id"]] = copy.deepcopy(c)
        return d

    def _node_coords(self, node):
        if "posX" in node and "posY" in node:
            return float(node["posX"]), float(node["posY"])
        if "pos" in node:
            p = node["pos"]
            return float(p.get("x", 0.0)), float(p.get("y", 0.0))
        return None

    def _euclid(self, a, b):
        if a is None or b is None:
            return 999.0
        return hypot(a[0]-b[0], a[1]-b[1])

    def _find_nearest_station(self, node):
        nodes = self.map_obj.get("nodes", []) or []
        pos = self._node_coords(node)
        stations = []
        for n in nodes:
            if n.get("target", {}).get("Type") == "ChargingStation":
                targ = n.get("target", {})
                speed = float(targ.get("chargeSpeedPerCharger", 0) or 0)
                avail = int(targ.get("amountOfAvailableChargers", 0) or 0)
                stations.append((n, speed, avail))
        if not stations:
            return None, None
        nearest = min(stations, key=lambda t: self._euclid(pos, self._node_coords(t[0])))
        fastest = max(stations, key=lambda t: (t[1], t[2]))
        return nearest[0]["id"], fastest[0]["id"]

    # Observation/feature-extrahering
    def get_customer_features(self):
        """
        Extraherar kundfeatures i samma format som env.py — 5 features per kund.
        """
        feats = []
        nodes = self.map_obj.get("nodes", []) or []
        coords = {n["id"]: self._node_coords(n) for n in nodes}
        station_nodes = [n for n in nodes if n.get("target", {}).get("Type") == "ChargingStation"]
        max_dist = 1.0
        if station_nodes:
            all_coords = [coords[n["id"]] for n in station_nodes if coords.get(n["id"]) is not None]
            if all_coords:
                xs = [c[0] for c in all_coords]; ys = [c[1] for c in all_coords]
                max_dist = max(1.0, math.hypot(max(xs)-min(xs), max(ys)-min(ys)))

        for node in nodes:
            node_pos = coords.get(node["id"])
            for c in node.get("customers", []) or []:
                charge = float(c.get("chargeRemaining", 0) or 0)
                max_charge = float(c.get("maxCharge", 1) or 1)
                charge_frac = charge / max_charge if max_charge > 0 else 0.0
                isAtStation = 1.0 if node.get("target", {}).get("Type") == "ChargingStation" else 0.0
                dep = int(c.get("departureTick", self.total_ticks))
                ticks_to_depart = max(0, dep - self.current_tick)
                nearest_id, _ = self._find_nearest_station(node)
                nearest_pos = coords.get(nearest_id) if nearest_id else None
                dist_to_station = self._euclid(node_pos, nearest_pos) / max_dist
                goal_node = None
                goal_id = c.get("toNode")
                if goal_id:
                    goal_node = coords.get(goal_id)
                dist_to_goal = self._euclid(node_pos, goal_node) / max_dist
                feats.append([
                    float(charge_frac),
                    float(isAtStation),
                    float(ticks_to_depart) / max(1, self.total_ticks),
                    float(dist_to_station),
                    float(dist_to_goal),
                ])
        return feats

    # Steg i miljön.
    def step(self, actions):
        """
        Skickar actions till API:et och beräknar reward.
        """
        nodes = self.map_obj.get("nodes", []) or []
        customers_flat = []
        for node in nodes:
            for c in node.get("customers", []) or []:
                customers_flat.append((node, c))

        tick_actions = []
        for (node, cust), act in zip(customers_flat, actions):
            try:
                a = int(act)
            except Exception:
                a = 0

            isAtStation = (node.get("target", {}).get("Type") == "ChargingStation")
            if a == 0:
                continue
            if a == 1:
                nearest_id, _ = self._find_nearest_station(node)
                if nearest_id:
                    tick_actions.append({"customerId": str(cust["id"]), "pathTo": [nearest_id]})
                continue
            if a == 2:
                _, fastest_id = self._find_nearest_station(node)
                if fastest_id:
                    tick_actions.append({"customerId": str(cust["id"]), "pathTo": [fastest_id]})
                continue
            if a == 3:
                if isAtStation:
                    tick_actions.append({"customerId": str(cust["id"]), "chargeTo": 0.95})
                else:
                    _, fastest_id = self._find_nearest_station(node)
                    if fastest_id:
                        tick_actions.append({"customerId": str(cust["id"]), "pathTo": [fastest_id]})
                continue

        tick_payload = {"tick": self.current_tick, "customerRecommendations": tick_actions}
        self.ticks_sent.append(tick_payload)
        input_payload = {"mapName": self.map_name, "playToTick": self.current_tick + 1, "ticks": self.ticks_sent}

        try:
            game_response = self.client.post_game(input_payload)
        except Exception as e:
            print(f"Error posting game: {e}")
            return self.get_customer_features(), 0.0, True

        prev_score = self.last_score
        new_score = (
            game_response.get("customerCompletionScore", 0)
            + game_response.get("kwhRevenue", 0)
            + game_response.get("score", 0)
        )
        base_delta = float(new_score - prev_score)
        self.last_score = new_score

        new_map = game_response.get("map", self.map_obj) or self.map_obj
        new_customers = self._flatten_customers(new_map)
        completed = 0
        charge_gain_reward = 0.0

        for cid, prev_c in self.prev_customers.items():
            if cid not in new_customers:
                completed += 1
            else:
                new_c = new_customers[cid]
                try:
                    prev_charge = float(prev_c.get("chargeRemaining", 0) or 0)
                    new_charge = float(new_c.get("chargeRemaining", 0) or 0)
                    if new_charge > prev_charge:
                        charge_gain_reward += (new_charge - prev_charge) * 100.0
                except Exception:
                    pass

        self.prev_customers = new_customers

        reward = base_delta + charge_gain_reward + completed * 50.0
        self.map_obj = new_map
        self.current_tick += 1
        done = self.current_tick >= self.total_ticks

        next_feats = self.get_customer_features()
        return next_feats, float(reward), done

    def sample_action(self):
        """
        Returnerar slumpmässiga actions för alla kunder i nuvarande tick.
        """
        customers = [c for node in self.map_obj.get("nodes", []) for c in node.get("customers", []) or []]
        import random
        return [random.randint(0, 3) for _ in customers]
