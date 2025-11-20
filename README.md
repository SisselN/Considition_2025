## Considition 2025 - En Deep Q Network-agent
Det här projktet är mitt bidrag till Considition 2025 och består av en Deep Q Network-agent (DQN).

Agenten tränas på fyra träningskartor som hämtats via API. Den tränade moellen finetunas sedan i två steg, först på de ursprungliga fyra träningskartorna samt en femte, därefter på tävlingskartan.

Agenten använder ett deep Q-network med två fullt anslutna lager och ReLU-aktiveringar för att approximera Q-funktionen.

Replay Buffer: Erfarenheter lagras i en deque och slumpmässiga minibatcher används för stabil träning.

Epsilon-greedy policy: Utforskning minskar gradvis under träningen enligt definierade hyperparametrar.

Mål-nätverk (Target Network): Uppdateras periodiskt för att stabilisera Q-learning.

Multi-map träning: Varje episod väljer slumpmässigt en av de fyra träningskartorna för att generalisera agentens beteende.

Sparande av modell: Modellens vikt sparas regelbundet och slutligen efter alla episoder för senare finetuning eller inferens.

Projektet illustrerar hur förstärkningsinlärning kan tillämpas på en multi-map miljö med en DQN-agent som lär sig optimala handlingar över tid.



### Projektstruktur

├── README.md  
├── requirements.txt  
├── baseline_agent/  
│ ├── app.py # Grundläggande klient för API  
│ └── client.py  
├── maps/  
│ ├── map_dump_Batterytown.json  
│ ├── map_dump_Clutchfield.json  
│ ├── map_dump_Turbohill.json  
│ └── map_dump_Windcity.json  
├── models/  
│ ├── dqn_api_multi_map.pth # Tränad DQN-modell på de fyra träningskartor som finns i maps.  
│ ├── dqn_api_multi_map_finetuned.pth # DQN-modell som finetunats på samtliga fem träningskartor. 
│ └── dqn_api_multi_map_finetuned2.pth # DQN-modell som finetunats på tävlingskartan.  
└── src/  
├── dump_map.py # Script för att exportera/inspektera kartor.  
├── env.py # Grundläggande miljöklass.  
├── env_api_simulated.py # Lokal simulerad miljö för träning/finetuning.  
├── fine_tune_competition_map.py # Finetuning på tävlingskartan.  
├── play_model.py # Kör tränad modell mot API.  
└── train_api_sim_4maps.py # Tränar DQN-modell på fyra träningskartor.  

---

## Funktioner

- Träning av DQN-agent på flera kartor med lokal simulerad miljö (`src/env_api_simulated.py`)
- Finetuning av redan tränad modell på ny kartdata (`src/fine_tune_competition_map.py`)
- Körning av tränad modell mot live API (`src/play_model.py`)
- Export och inspektion av kartor (`src/dump_map.py`)
- Baseline-agent för jämförelse (`baseline_agent/`)

---

## Kör projektet

1. **Installera beroenden**:

```bash
pip install -r requirements.txt


#### Sissel Nevestveit - november 2025
