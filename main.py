import requests
from time import sleep
import random
import spacy
import numpy as np

# Încărcăm modelul spaCy (asigură-te că ai descărcat modelul: python -m spacy download en_core_web_md)
nlp = spacy.load("en_core_web_md")

# Configurații de rețea
host = "http://172.18.4.158:8000"
post_url = f"{host}/submit-word"
get_url = f"{host}/get-word"
status_url = f"{host}/status"

NUM_ROUNDS = 5

# Lista de cuvinte a jucătorului cu costuri
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

DOMAIN_MAP = {
    "Feather": "light",
    "Coal": "energy",
    "Pebble": "nature",
    "Leaf": "nature",
    "Paper": "information",
    "Rock": "strength",
    "Water": "fluid",
    "Twig": "nature",
    "Sword": "weapon",
    "Shield": "defense",
    "Gun": "weapon",
    "Flame": "fire",
    "Rope": "connection",
    "Disease": "negative",
    "Cure": "healing",
    "Bacteria": "nature",
    "Shadow": "mysterious",
    "Light": "positive",
    "Virus": "negative",
    "Sound": "communication",
    "Time": "abstract",
    "Fate": "abstract",
    "Earthquake": "destruction",
    "Storm": "destruction",
    "Vaccine": "healing",
    "Logic": "abstract",
    "Gravity": "physical",
    "Robots": "technology",
    "Stone": "nature",
    "Echo": "sound",
    "Thunder": "nature",
    "Karma": "abstract",
    "Wind": "nature",
    "Ice": "cold",
    "Sandstorm": "destruction",
    "Laser": "technology",
    "Magma": "fire",
    "Peace": "abstract",
    "Explosion": "destruction",
    "War": "conflict",
    "Enlightenment": "abstract",
    "Nuclear Bomb": "destruction",
    "Volcano": "nature",
    "Whale": "nature",
    "Earth": "nature",
    "Moon": "celestial",
    "Star": "celestial",
    "Tsunami": "destruction",
    "Supernova": "celestial",
    "Antimatter": "science",
    "Plague": "negative",
    "Rebirth": "abstract",
    "Tectonic Shift": "destruction",
    "Gamma-Ray Burst": "celestial",
    "Human Spirit": "abstract",
    "Apocalyptic Meteor": "destruction",
    "Earth’s Core": "physical",
    "Neutron Star": "celestial",
    "Supermassive Black Hole": "celestial",
    "Entropy": "abstract"
}

DOMAIN_STRENGTH = {
    "light": 1,
    "energy": 2,
    "nature": 3,
    "information": 1,
    "strength": 3,
    "fluid": 2,
    "weapon": 4,
    "defense": 4,
    "fire": 5,
    "connection": 1,
    "negative": 1,
    "healing": 3,
    "mysterious": 2,
    "positive": 3,
    "communication": 1,
    "abstract": 6,
    "destruction": 5,
    "technology": 4,
    "cold": 2,
    "conflict": 5,
    "celestial": 4,
    "science": 4,
    "physical": 3
}

DOMAIN_WORDS = {
    "destruction": ["bomb", "explosion", "nuclear", "tsunami"],
    "healing": ["cure", "vaccine", "healing", "medicine"],
    "weapon": ["sword", "gun", "rifle", "weapon"],
    "nature": ["rock", "stone", "earth", "leaf", "water", "tree"],
    "abstract": ["logic", "time", "fate", "enlightenment", "philosophy", "greed", "excess", "avarice"]
}

domain_vectors = {}
for domain, words in DOMAIN_WORDS.items():
    vectors = []
    for word in words:
        doc = nlp(word)
        if doc.has_vector:
            vectors.append(doc.vector)
    if vectors:
        domain_vectors[domain] = np.mean(vectors, axis=0)
    else:
        domain_vectors[domain] = np.zeros(nlp.vocab.vectors_length)

def assign_domain(word):
    """
    Atribuie un domeniu pentru cuvinte necunoscute folosind similaritatea cosine.
    """
    doc = nlp(word)
    if not doc.has_vector:
        return "abstract"
    word_vector = doc.vector
    best_domain = None
    best_similarity = -1
    for domain, vector in domain_vectors.items():
        norm_product = np.linalg.norm(word_vector) * np.linalg.norm(vector) + 1e-10
        similarity = np.dot(word_vector, vector) / norm_product
        if similarity > best_similarity:
            best_similarity = similarity
            best_domain = domain
    return best_domain

def get_abstraction(word):
    """
    Calculează "puterea" unui cuvânt pe baza domeniului său semantic.
    Dacă cuvântul este cunoscut, se folosește mapping-ul; altfel, se atribuie prin NLP.
    """
    domain = DOMAIN_MAP.get(word)
    if domain is None:
        domain = assign_domain(word)
    return DOMAIN_STRENGTH.get(domain, 0)

def adaptive_what_beats(unknown_word, player_words=PLAYER_WORDS):
    unknown_score = get_abstraction(unknown_word)
    winning_candidates = [c for c in player_words if get_abstraction(c['word']) > unknown_score]
    if winning_candidates:
        return min(winning_candidates, key=lambda c: c['cost'])
    else:
        return min(player_words, key=lambda c: c['cost'])

def play_game(player_id):
    """
    Așteaptă runda corectă de la server, apoi alege cuvântul potrivit.
    Serverul returnează un JSON de forma:
    {
      "word": "<cuvânt_sistem>",
      "round": <număr_rundă>
    }
    """
    for round_id in range(1, NUM_ROUNDS + 1):
        round_num = -1
        
        # Așteptăm până primim runda care corespunde lui round_id
        while round_num != round_id:
            response = requests.get(get_url)
            try:
                json_response = response.json()
            except ValueError:
                json_response = {}
            
            # Extragem cuvântul și runda
            unknown_word = json_response.get('word', '')
            round_num = json_response.get('round', -1)
            
            print(f"Waiting for round {round_id}, received round {round_num} - word: {unknown_word}")
            sleep(1)  # Mică pauză pentru a evita spam-ul

        # Odată ce round_num == round_id, putem face selecția
        print(f"\n=== Handling round {round_id} ===")
        print(f"System word: {unknown_word} (Power: {get_abstraction(unknown_word)}, Domain: {assign_domain(unknown_word)})")

        if round_id > 1:
            status = requests.get(status_url)
            print("Status:", status.json())
        
        chosen_candidate = adaptive_what_beats(unknown_word)
        print(f"Chosen word: {chosen_candidate['word']} (Power: {get_abstraction(chosen_candidate['word'])}, Cost: {chosen_candidate['cost']})")
        
        data = {"player_id": player_id, "word_id": chosen_candidate['id'], "round_id": round_id}
        response = requests.post(post_url, json=data)
        try:
            resp_data = response.json()
        except ValueError:
            resp_data = response.text
        print("Round", round_id, "response:", resp_data)
        print("====================================\n")
        sleep(1)

if __name__ == "__main__":
    player_id = "player123"
    play_game(player_id)
