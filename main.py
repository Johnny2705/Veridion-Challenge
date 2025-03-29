import requests
from time import sleep
import numpy as np
from sentence_transformers import SentenceTransformer, util

# 1. Incarcam modelul
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# 2. Config de retea
host = "http://172.18.4.158:8000"
post_url = f"{host}/submit-word"
get_url = f"{host}/get-word"
status_url = f"{host}/status"

NUM_ROUNDS = 10

# 3. Cuvintele jucatorului cu costuri
PLAYER_WORDS = [
    {"id": 1, "word": "Feather", "cost": 1},
    {"id": 2, "word": "Coal", "cost": 1},
    {"id": 3, "word": "Pebble", "cost": 1},
    {"id": 4, "word": "Leaf", "cost": 2},
    {"id": 5, "word": "Paper", "cost": 2},
    {"id": 6, "word": "Rock", "cost": 2},
    {"id": 7, "word": "Water", "cost": 3},
    {"id": 8, "word": "Twig", "cost": 3},
    {"id": 9, "word": "Sword", "cost": 4},
    {"id": 10, "word": "Shield", "cost": 4},
    {"id": 11, "word": "Gun", "cost": 5},
    {"id": 12, "word": "Flame", "cost": 5},
    {"id": 13, "word": "Rope", "cost": 5},
    {"id": 14, "word": "Disease", "cost": 6},
    {"id": 15, "word": "Cure", "cost": 6},
    {"id": 16, "word": "Bacteria", "cost": 6},
    {"id": 17, "word": "Shadow", "cost": 7},
    {"id": 18, "word": "Light", "cost": 7},
    {"id": 19, "word": "Virus", "cost": 7},
    {"id": 20, "word": "Sound", "cost": 8},
    {"id": 21, "word": "Time", "cost": 8},
    {"id": 22, "word": "Fate", "cost": 8},
    {"id": 23, "word": "Earthquake", "cost": 9},
    {"id": 24, "word": "Storm", "cost": 9},
    {"id": 25, "word": "Vaccine", "cost": 9},
    {"id": 26, "word": "Logic", "cost": 10},
    {"id": 27, "word": "Gravity", "cost": 10},
    {"id": 28, "word": "Robots", "cost": 10},
    {"id": 29, "word": "Stone", "cost": 11},
    {"id": 30, "word": "Echo", "cost": 11},
    {"id": 31, "word": "Thunder", "cost": 12},
    {"id": 32, "word": "Karma", "cost": 12},
    {"id": 33, "word": "Wind", "cost": 13},
    {"id": 34, "word": "Ice", "cost": 13},
    {"id": 35, "word": "Sandstorm", "cost": 13},
    {"id": 36, "word": "Laser", "cost": 14},
    {"id": 37, "word": "Magma", "cost": 14},
    {"id": 38, "word": "Peace", "cost": 14},
    {"id": 39, "word": "Explosion", "cost": 15},
    {"id": 40, "word": "War", "cost": 15},
    {"id": 41, "word": "Enlightenment", "cost": 15},
    {"id": 42, "word": "Nuclear Bomb", "cost": 16},
    {"id": 43, "word": "Volcano", "cost": 16},
    {"id": 44, "word": "Whale", "cost": 17},
    {"id": 45, "word": "Earth", "cost": 17},
    {"id": 46, "word": "Moon", "cost": 17},
    {"id": 47, "word": "Star", "cost": 18},
    {"id": 48, "word": "Tsunami", "cost": 18},
    {"id": 49, "word": "Supernova", "cost": 19},
    {"id": 50, "word": "Antimatter", "cost": 19},
    {"id": 51, "word": "Plague", "cost": 20},
    {"id": 52, "word": "Rebirth", "cost": 20},
    {"id": 53, "word": "Tectonic Shift", "cost": 21},
    {"id": 54, "word": "Gamma-Ray Burst", "cost": 22},
    {"id": 55, "word": "Human Spirit", "cost": 23},
    {"id": 56, "word": "Apocalyptic Meteor", "cost": 24},
    {"id": 57, "word": "Earth’s Core", "cost": 25},
    {"id": 58, "word": "Neutron Star", "cost": 26},
    {"id": 59, "word": "Supermassive Black Hole", "cost": 35},
    {"id": 60, "word": "Entropy", "cost": 45}
]

# 4. Embeddinguri precalculate
WORD_EMB = {w['word']: model.encode(w['word'], convert_to_tensor=True) for w in PLAYER_WORDS}

# 5. Bonus de sinergie - relații de întărire
# Redefine the synergy bonus mapping after kernel reset

BONUS_SYNERGY = {
    "fire": ["ice", "cold", "freeze", "extinguish"],
    "ice": ["fire", "heat", "burn", "melt"],
    "light": ["dark", "shadow", "night", "obscure"],
    "dark": ["light", "shine", "radiance", "glow"],
    "water": ["fire", "lava", "burn", "flame"],
    "earth": ["flood", "collapse", "shake", "quake"],
    "shield": ["sword", "attack", "weapon", "strike"],
    "sword": ["armor", "defense", "shield", "barrier"],
    "logic": ["chaos", "emotion", "instinct", "impulse", "irrational", "random", "myth"],
    "time": ["death", "decay", "aging", "end"],
    "vaccine": ["virus", "disease", "infection", "bacteria"],
    "cure": ["illness", "sick", "plague", "symptom"],
    "peace": ["war", "conflict", "battle", "aggression"],
    "gravity": ["float", "levitate", "air", "weightless", "zero-g"],
    "laser": ["darkness", "blind", "shadow", "cover"],
    "explosion": ["structure", "building", "wall", "construction"],
    "wind": ["smoke", "cloud", "gas", "ash"],
    "sound": ["silence", "quiet", "mute", "hush"],
    "entropy": ["order", "structure", "pattern", "law"],
    "rebirth": ["death", "end", "despair", "decay"],
    "magma": ["ice", "freeze", "cold", "chill"],
    "storm": ["calm", "peace", "clear", "still"],
    "thunder": ["whisper", "silence", "mute", "quiet"],
    "human spirit": ["hopeless", "despair", "collapse", "apathy"],
    "nuclear bomb": ["city", "civilization", "structure", "society"],
}

BONUS_SYNERGY


BONUS_VECTORS = {k: model.encode(v, convert_to_tensor=True) for k, v in BONUS_SYNERGY.items()}

def synergy_score(word):
    emb = model.encode(word, convert_to_tensor=True)
    score = 0
    for domain, vectors in BONUS_VECTORS.items():
        sim = util.cos_sim(emb, vectors).max().item()
        score = max(score, sim)
    return score

def choose_best_word(unknown_word, alpha=0.6, beta=0.05, gamma=1.2):
    unknown_emb = model.encode(unknown_word, convert_to_tensor=True)
    best = None
    best_score = -float('inf')

    for word in PLAYER_WORDS:
        cand_emb = WORD_EMB[word['word']]
        sim = util.cos_sim(unknown_emb, cand_emb).item()
        sync = synergy_score(word['word'])
        cost_pen = word['cost'] * beta
        score = gamma * sync + alpha * sim - cost_pen
        if score > best_score:
            best_score = score
            best = word
    return best

def play_game(player_id):
    for round_id in range(1, NUM_ROUNDS + 1):
        while True:
            try:
                res = requests.get(get_url).json()
                if isinstance(res, list):
                    system_data = next((r for r in res if r['round'] == round_id), {})
                else:
                    system_data = res
                if system_data.get('round') == round_id:
                    break
            except:
                pass
            sleep(0.5)

        sys_word = system_data.get('word', '')
        print(f"\n=== Round {round_id} ===")
        print(f"System word: {sys_word}")

        chosen = choose_best_word(sys_word)
        print(f"Chosen word: {chosen['word']} (Cost: {chosen['cost']})")

        payload = {"player_id": player_id, "word_id": chosen['id'], "round_id": round_id}
        try:
            print("Post:", requests.post(post_url, json=payload).json())
        except:
            print("Post error")

        try:
            print("Status:", requests.get(status_url).json())
        except:
            print("Status error")

        print("===========================\n")
        sleep(1)

if __name__ == "__main__":
    play_game("l1a8Ki1v7A")