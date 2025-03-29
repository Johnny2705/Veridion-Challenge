import requests
from time import sleep
import spacy
import numpy as np

# 1. Încarcă modelul spaCy mare
# Asigură-te că rulezi în terminal: python -m spacy download en_core_web_lg
nlp = spacy.load("en_core_web_lg")

# 2. Configurații de rețea
host = "http://172.18.4.158:8000"
post_url = f"{host}/submit-word"
get_url = f"{host}/get-word"
status_url = f"{host}/status"

NUM_ROUNDS = 5

# 3. Lista de cuvinte a jucătorului cu costuri
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

# 4. Mapping pentru cuvinte cunoscute: asociază domeniul semantic
# Am extins mapping-ul pentru a fi mai specific. De exemplu:
# - "Coal", "Pebble", "Rock", "Stone" devin "mineral"
# - "Leaf", "Twig" devin "plant"
# - "Whale" devine "aquatic"
DOMAIN_MAP = {
    "Feather": "light",
    "Coal": "mineral",
    "Pebble": "mineral",
    "Leaf": "plant",
    "Paper": "information",
    "Rock": "mineral",
    "Water": "fluid",
    "Twig": "plant",
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
    "Stone": "mineral",
    "Echo": "communication",
    "Thunder": "elemental",  # considerăm thunder ca fiind un fenomen elementar
    "Karma": "abstract",
    "Wind": "fluid",
    "Ice": "cold",
    "Sandstorm": "destruction",
    "Laser": "technology",
    "Magma": "fire",
    "Peace": "abstract",
    "Explosion": "destruction",
    "War": "conflict",
    "Enlightenment": "abstract",
    "Nuclear Bomb": "destruction",
    "Volcano": "elemental",
    "Whale": "aquatic",
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
    "Entropy": "abstract",
    # Exemplu pentru particule fine:
    "Dust": "powder",
    "Ash": "powder"
}

# 5. Scorurile domeniilor (puterea fiecărui domeniu)
DOMAIN_STRENGTH = {
    "light": 1,
    "energy": 2,
    "mineral": 3,
    "nature": 3,
    "plant": 2,
    "animal": 3,
    "aquatic": 4,
    "powder": 1,
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
    "physical": 3,
    "elemental": 5,
    "magical": 7  # poți folosi dacă adaugi domeniul "magical" în mapping
}

# 6. Domenii și prototipuri (pentru cuvinte necunoscute)
DOMAIN_WORDS = {
    "destruction": ["bomb", "explosion", "nuclear", "tsunami"],
    "healing": ["cure", "vaccine", "healing", "medicine"],
    "weapon": ["sword", "gun", "rifle", "weapon"],
    "mineral": ["coal", "pebble", "rock", "stone", "ore", "crystal"],
    "plant": ["leaf", "twig", "vine", "flower", "grass"],
    "nature": ["earth", "tree"],
    "aquatic": ["whale", "dolphin", "shark", "fish", "seal"],
    "powder": ["dust", "ash", "silt", "powder"],
    "abstract": ["logic", "time", "fate", "enlightenment", "philosophy", "greed", "excess", "avarice"],
    "elemental": ["volcano", "thunder", "storm"],
    # Poți adăuga și alte domenii, de exemplu:
    "magical": ["dragon", "phoenix", "unicorn", "spell"]
}

# 7. Calcul vectori medii pentru fiecare domeniu (pentru clasificarea cuvintelor necunoscute)
domain_vectors = {}
for domain, words in DOMAIN_WORDS.items():
    vectors = []
    for w in words:
        doc = nlp(w)
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
    for d, vec in domain_vectors.items():
        norm_product = np.linalg.norm(word_vector) * np.linalg.norm(vec) + 1e-10
        similarity = np.dot(word_vector, vec) / norm_product
        if similarity > best_similarity:
            best_similarity = similarity
            best_domain = d
    return best_domain

def get_abstraction(word):
    """
    Calculează 'puterea' unui cuvânt pe baza domeniului său semantic.
    Dacă cuvântul este cunoscut (prezent în DOMAIN_MAP), se folosește mapping-ul;
    altfel, se atribuie un domeniu prin NLP.
    """
    domain = DOMAIN_MAP.get(word)
    if domain is None:
        domain = assign_domain(word)
    return DOMAIN_STRENGTH.get(domain, 0)

# 8. Algoritmul de selecție cu penalizare și bonus de similaritate:
def effective_cost(candidate, unknown_word, penalty_factor=10, sim_factor=5):
    """
    Calculează costul efectiv pentru un candidat:
      - Dacă puterea candidatului >= puterea cuvântului necunoscut: cost efectiv = cost
      - Altfel: cost efectiv = cost + penalty_factor * (unknown_score - candidate_score)
    Se scade un bonus bazat pe similaritatea cosine dintre cuvintele candidat și necunoscut.
    """
    unknown_score = get_abstraction(unknown_word)
    candidate_score = get_abstraction(candidate['word'])
    base_cost = candidate['cost']
    if candidate_score >= unknown_score:
        effective = base_cost
    else:
        effective = base_cost + penalty_factor * (unknown_score - candidate_score)
    
    similarity = nlp(candidate['word']).similarity(nlp(unknown_word))
    effective -= sim_factor * similarity
    return effective

def adaptive_what_beats(unknown_word, player_words=PLAYER_WORDS, penalty_factor=10, sim_factor=5):
    best_candidate = None
    best_eff_cost = float('inf')
    for candidate in player_words:
        cost_eff = effective_cost(candidate, unknown_word, penalty_factor, sim_factor)
        if cost_eff < best_eff_cost:
            best_eff_cost = cost_eff
            best_candidate = candidate
    return best_candidate

# 9. Funcția de joc: pentru fiecare rundă, facem un singur apel GET, apoi POST, apoi GET status.
def play_game(player_id):
    """
    Pentru fiecare rundă:
      - Face un apel GET și preia obiectul cu "word" și "round" corespunzător.
      - Folosește adaptive_what_beats pentru a alege cuvântul.
      - Trimite alegerea prin POST și apoi preia statusul jocului.
    Se presupune că JSON-ul de la GET are forma:
    {
      "word": "<cuvânt_sistem>",
      "round": <număr_rundă>
    }
    """
    for round_id in range(1, NUM_ROUNDS + 1):
        round_num = -1
        unknown_word = ""
        system_data = {}
        while round_num != round_id:
            response = requests.get(get_url)
            try:
                json_response = response.json()
            except ValueError:
                json_response = {}
            
            if isinstance(json_response, list):
                system_data = next((item for item in json_response if item.get('round') == round_id), {})
            else:
                system_data = json_response
            
            unknown_word = system_data.get('word', '')
            round_num = system_data.get('round', -1)
            print(f"Waiting for round {round_id}, received round {round_num} - word: {unknown_word}")
            sleep(1)
        
        print(f"\n=== Handling round {round_id} ===")
        word_domain = assign_domain(unknown_word)
        word_power = get_abstraction(unknown_word)
        print(f"System word: {unknown_word} (Power: {word_power}, Domain: {word_domain})")
        
        chosen_candidate = adaptive_what_beats(unknown_word)
        chosen_power = get_abstraction(chosen_candidate['word'])
        print(f"Chosen word: {chosen_candidate['word']} (Power: {chosen_power}, Cost: {chosen_candidate['cost']})")
        
        data = {"player_id": player_id, "word_id": chosen_candidate['id'], "round_id": round_id}
        post_response = requests.post(post_url, json=data)
        try:
            print("Post response:", post_response.json())
        except ValueError:
            print("Post response:", post_response.text)
        
        status_response = requests.get(status_url)
        try:
            print("Status:", status_response.json())
        except ValueError:
            print("Status: no valid JSON")
        print("====================================\n")
        sleep(1)

if __name__ == "__main__":
    player_id = "player123"
    play_game(player_id)
