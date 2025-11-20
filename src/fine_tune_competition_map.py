# Finetuning av DQN-modell på tävlingskartan "Pistonia".

import torch
import torch.nn as nn
import torch.optim as optim
import random
from tqdm import trange
from env_api_simulated import ConsiditionEnv
from train_api_sim_4maps import DQN, ReplayBuffer

# Finetuning-hyperparametrar
BATCH_SIZE = 64
GAMMA = 0.95
LR = 5e-4  # Låg inlärningshastighet för finetuning
EPS_START = 0.2 
EPS_END = 0.05
EPS_DECAY = 0.995
TARGET_UPDATE = 15
MEMORY_SIZE = 10000
NUM_EPISODES = 300 
BASE_MODEL_PATH = "dqn_api_multi_map.pth"
FINE_TUNED_MODEL_PATH = "dqn_api_multi_map_finetuned2.pth"

def fine_tune():
    """
    Finetuning av DQN-modell på tävlingskartan "Pistonia".
    Tar den förtränade modellen från train_api_sim_4maps.py och finjusterar den här.
    Sparar den finjusterade modellen som en ny fil.
    """
    env = ConsiditionEnv(map_names="Pistonia")
    input_dim = 5
    num_actions = 4

    # Laddar tränad modell
    policy_net = DQN(input_dim, num_actions)
    policy_net.load_state_dict(torch.load(BASE_MODEL_PATH, map_location=torch.device("cpu")))

    target_net = DQN(input_dim, num_actions)
    target_net.load_state_dict(policy_net.state_dict())

    optimizer = optim.Adam(policy_net.parameters(), lr=LR)
    memory = ReplayBuffer(MEMORY_SIZE)
    epsilon = EPS_START

    rewards_per_ep = []
    progress = trange(NUM_EPISODES, desc="Finetuning DQN", ncols=100)

    for episode in progress:
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

            # Träna minibatch
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
            "reward": f"{total_reward:.2f}",
            "eps": f"{epsilon:.2f}"
        })

    # Spara den finjusterade modellen.
    torch.save(policy_net.state_dict(), FINE_TUNED_MODEL_PATH)
    print(f"\nFinetuning klar! Sparad som {FINE_TUNED_MODEL_PATH}")

if __name__ == "__main__":
    fine_tune()
