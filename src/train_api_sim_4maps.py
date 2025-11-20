# Tränar en DQN-agent på alla träningskartor i Considition API-simulatorn.

import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
from tqdm import trange
from env_api_simulated import ConsiditionEnv

# Hyperparametrar
BATCH_SIZE = 64
GAMMA = 0.95
LR = 1e-3
EPS_START = 1.0
EPS_END = 0.05
EPS_DECAY = 0.995
TARGET_UPDATE = 20
MEMORY_SIZE = 20000
NUM_EPISODES = 1000
MODEL_PATH = "dqn_api_multi_map.pth"

# DQN Nätverk
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim)
        )
    def forward(self, x):
        return self.net(x)

# Replay Buffer
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    def push(self, s, a, r, s2, d):
        self.buffer.append((s, a, r, s2, d))
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        s, a, r, s2, d = zip(*batch)
        return (
            torch.tensor(s, dtype=torch.float32),
            torch.tensor(a, dtype=torch.int64),
            torch.tensor(r, dtype=torch.float32),
            torch.tensor(s2, dtype=torch.float32),
            torch.tensor(d, dtype=torch.float32),
        )
    def __len__(self):
        return len(self.buffer)

# Träningsloop Multi-Map
def train_multi_map():
    maps = ["Batterytown", "Turbohill", "Clutchfield", "Thunderroad"]  # alla fyra träningskartor
    input_dim = 5
    num_actions = 4

    policy_net = DQN(input_dim, num_actions)
    target_net = DQN(input_dim, num_actions)
    target_net.load_state_dict(policy_net.state_dict())

    optimizer = optim.Adam(policy_net.parameters(), lr=LR)
    memory = ReplayBuffer(MEMORY_SIZE)
    epsilon = EPS_START
    rewards_per_ep = []

    # Progressionsindikator
    progress = trange(NUM_EPISODES, desc="Tränar DQN Multi-Map", ncols=100)

    for episode in progress:
        # Välj slumpmässig karta
        map_name = random.choice(maps)
        env = ConsiditionEnv()
        state = env.reset()
        total_reward = 0
        done = False

        while not done:
            actions = []
            for s in state:
                if random.random() < epsilon:
                    actions.append(random.randint(0, num_actions - 1))
                else:
                    with torch.no_grad():
                        q_vals = policy_net(torch.tensor(s, dtype=torch.float32))
                        actions.append(int(torch.argmax(q_vals).item()))

            next_state, reward, done = env.step(actions)
            total_reward += reward

            for s, a in zip(state, actions):
                s2 = random.choice(next_state) if len(next_state) > 0 else s
                memory.push(s, a, reward / len(actions), s2, done)

            state = next_state

            if len(memory) >= BATCH_SIZE:
                s_b, a_b, r_b, s2_b, d_b = memory.sample(BATCH_SIZE)
                q_vals = policy_net(s_b)
                q_val = q_vals.gather(1, a_b.unsqueeze(1)).squeeze(1)
                with torch.no_grad():
                    next_q = target_net(s2_b).max(1)[0]
                    target = r_b + GAMMA * next_q * (1 - d_b)
                loss = nn.functional.mse_loss(q_val, target)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        epsilon = max(EPS_END, epsilon * EPS_DECAY)
        rewards_per_ep.append(total_reward)

        if episode % TARGET_UPDATE == 0:
            target_net.load_state_dict(policy_net.state_dict())

        progress.set_postfix({
            "map": map_name,
            "reward": f"{total_reward:.2f}",
            "eps": f"{epsilon:.2f}"
        })

        if (episode + 1) % 25 == 0:
            torch.save(policy_net.state_dict(), MODEL_PATH)

    # Spara slutmodell
    torch.save(policy_net.state_dict(), MODEL_PATH)
    print(f"\nTräning färdig! Modell sparad till {MODEL_PATH}")

if __name__ == "__main__":
    train_multi_map()
