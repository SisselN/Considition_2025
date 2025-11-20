# Simulerad miljö som efterliknar Considition API-beteendet.
import random

class ConsiditionEnv:
    """
    Simulerad miljö som efterliknar Considition API-beteende:
    Använder samma featurestruktur som env.py men kör lokalt utan nätverksanrop.
    """

    def __init__(self, map_names=None, max_ticks=300):
        """
        Initialisera den simulerade miljön.
        """
        self.map_names = map_names or ["Batterytown", "Clutchfield", "Turbohill", "Thunderroad", "Windcity"]
        self.max_ticks = max_ticks
        self.reset()

    def reset(self):
        """
        Startar om miljön till starttillståndet.
        """
        self.map_name = random.choice(self.map_names)
        self.tick = 0
        self.done = False

        # generera kunder
        self.num_customers = 200
        self.customers = [
            {
                "charge": random.uniform(0.2, 0.9),
                "max_charge": 1.0,
                "at_station": random.random() < 0.2,
                "departureTick": random.randint(100, self.max_ticks),
                "dist_to_station": random.random(),
                "dist_to_goal": random.random(),
                "active": True,
            }
            for _ in range(self.num_customers)
        ]
        return self.get_customer_features()

    def get_customer_features(self):
        """
        Samma struktur som env.py — 5 features per kund.
        """
        feats = []
        for c in self.customers:
            if not c["active"]:
                continue
            feats.append([
                c["charge"],
                float(c["at_station"]),
                c["departureTick"] / self.max_ticks,
                c["dist_to_station"],
                c["dist_to_goal"],
            ])
        return feats

    def step(self, actions):
        """
        Simulerar kundernas beteende och beräknar reward baserat på actions.
        """
        total_reward = 0.0

        for c, act in zip(self.customers, actions):
            if not c["active"]:
                continue

            # 0: gör inget
            if act == 0:
                total_reward -= 0.05  # ineffektivt att stå still

            # 1: åk till närmaste station.
            elif act == 1:
                c["at_station"] = True
                c["dist_to_station"] = 0.0
                c["dist_to_goal"] *= 0.95
                total_reward += 0.2

            # 2: åk till snabbaste station.
            elif act == 2:
                c["at_station"] = True
                c["dist_to_station"] = 0.0
                c["charge"] = min(1.0, c["charge"] + 0.15)
                total_reward += 0.4

            # 3: ladda till 95% (om vid station).
            elif act == 3:
                if c["at_station"]:
                    gain = max(0, 0.95 - c["charge"])
                    c["charge"] += 0.3 * gain
                    total_reward += gain * 3.0
                else:
                    total_reward -= 0.2  # försökte ladda men ej vid station

            # Rörelse och avresa.
            c["dist_to_goal"] = max(0, c["dist_to_goal"] - 0.02 * c["charge"])
            c["charge"] = max(0, c["charge"] - 0.03)
            c["departureTick"] -= 1

            if c["dist_to_goal"] < 0.05 and c["departureTick"] > 0:
                total_reward += 2.0  # bonus för att nå mål
                c["active"] = False
            elif c["departureTick"] <= 0:
                c["active"] = False  # missad avgång, ingen reward

        self.tick += 1
        done = self.tick >= self.max_ticks or all(not c["active"] for c in self.customers)
        next_state = self.get_customer_features()

        return next_state, total_reward, done

    def sample_action(self):
        """
        Slumpmässiga actions för test.
        """
        num_active = sum(1 for c in self.customers if c["active"])
        return [random.randint(0, 3) for _ in range(num_active)]
