"""Wikipedia scraper: fetches a curated list of important articles via the API.

Saves .md files to data/raw/wikipedia/. Tracks progress so you can Ctrl+C and resume.

Usage:
    python scripts/scrape_wiki.py                  # scrape all curated articles
    python scripts/scrape_wiki.py --max 50         # stop after 50
    python scripts/scrape_wiki.py --reset          # start over
"""

import argparse
import json
import os
import re
import sys
import time

import requests

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from configs.default import SpakieConfig

WIKI_API = "https://en.wikipedia.org/w/api.php"
HEADERS = {"User-Agent": "SpakieLLM/1.0 (educational language model project)"}

OUTPUT_DIR = "data/raw/wikipedia"
PROGRESS_FILE = "data/raw/wikipedia/.wiki_progress.json"

# ~500 important articles across major topics
CURATED_ARTICLES = [
    # Science
    "Physics", "Chemistry", "Biology", "Mathematics", "Astronomy",
    "Evolution", "Atom", "DNA", "Cell (biology)", "Quantum mechanics",
    "General relativity", "Thermodynamics", "Electromagnetism", "Organic chemistry",
    "Genetics", "Ecology", "Geology", "Periodic table", "Molecule", "Photosynthesis",
    "Natural selection", "Big Bang", "Black hole", "Star", "Planet",
    "Solar System", "Sun", "Moon", "Earth", "Mars",
    "Speed of light", "Gravity", "Energy", "Force", "Wave",
    "Electron", "Proton", "Neutron", "Nuclear fission", "Nuclear fusion",
    "Vaccine", "Virus", "Bacteria", "Antibiotic", "Enzyme",
    # Technology
    "Computer", "Internet", "Artificial intelligence", "Machine learning",
    "Algorithm", "Programming language", "Software", "Computer science",
    "World Wide Web", "Operating system", "Database", "Encryption",
    "Transistor", "Semiconductor", "Microprocessor", "Computer network",
    "Python (programming language)", "JavaScript", "Linux", "Open-source software",
    "Deep learning", "Neural network", "Natural language processing",
    "Robotics", "Blockchain", "Cloud computing", "Cybersecurity",
    "Smartphone", "Telephone", "Television", "Radio", "Electricity",
    "Nuclear power", "Solar energy", "Renewable energy", "Steam engine",
    "Printing press", "Automobile", "Airplane", "Rocket", "Satellite",
    # Mathematics
    "Calculus", "Algebra", "Geometry", "Statistics", "Probability",
    "Number theory", "Set theory", "Logic", "Pi", "Prime number",
    "Pythagorean theorem", "Linear algebra", "Differential equation",
    "Graph theory", "Topology", "Complex number", "Infinity",
    "Mathematical proof", "Euclidean geometry", "Trigonometry",
    # History
    "History of the world", "Ancient Egypt", "Ancient Greece", "Ancient Rome",
    "Roman Empire", "Byzantine Empire", "Ottoman Empire", "British Empire",
    "Middle Ages", "Renaissance", "Industrial Revolution", "Age of Enlightenment",
    "French Revolution", "American Revolution", "Russian Revolution",
    "World War I", "World War II", "Cold War", "Colonialism", "Imperialism",
    "Slavery", "Civil rights movement", "Apartheid", "Decolonization",
    "History of China", "History of India", "History of Japan",
    "Mesopotamia", "Silk Road", "Mongol Empire", "Persian Empire",
    "Alexander the Great", "Julius Caesar", "Genghis Khan", "Napoleon",
    "Abraham Lincoln", "Mahatma Gandhi", "Martin Luther King Jr.",
    "Nelson Mandela", "Winston Churchill", "Franklin D. Roosevelt",
    # Geography
    "Africa", "Asia", "Europe", "North America", "South America",
    "Australia", "Antarctica", "Pacific Ocean", "Atlantic Ocean", "Indian Ocean",
    "Amazon River", "Nile", "Sahara", "Himalayas", "Alps",
    "United States", "United Kingdom", "China", "India", "Russia",
    "Japan", "Germany", "France", "Brazil", "Canada",
    "Mexico", "Australia", "South Africa", "Nigeria", "Egypt",
    "Indonesia", "Turkey", "Italy", "Spain", "South Korea",
    "New York City", "London", "Tokyo", "Paris", "Beijing",
    "Rome", "Berlin", "Moscow", "Cairo", "Mumbai",
    # Philosophy & Religion
    "Philosophy", "Ethics", "Epistemology", "Metaphysics", "Aesthetics",
    "Socrates", "Plato", "Aristotle", "Immanuel Kant", "Friedrich Nietzsche",
    "Existentialism", "Utilitarianism", "Stoicism", "Rationalism", "Empiricism",
    "Religion", "Christianity", "Islam", "Hinduism", "Buddhism",
    "Judaism", "Atheism", "Mythology", "Theology", "Secularism",
    "Bible", "Quran", "Jesus", "Muhammad", "Gautama Buddha",
    # Arts & Literature
    "Art", "Music", "Literature", "Poetry", "Theatre",
    "Film", "Photography", "Architecture", "Sculpture", "Painting",
    "William Shakespeare", "Homer", "Leo Tolstoy", "Charles Dickens",
    "Mark Twain", "Jane Austen", "Franz Kafka", "Fyodor Dostoevsky",
    "Novel", "Short story", "Essay", "Drama", "Comedy",
    "Classical music", "Jazz", "Rock music", "Hip hop music", "Opera",
    "Ludwig van Beethoven", "Wolfgang Amadeus Mozart", "Johann Sebastian Bach",
    "Leonardo da Vinci", "Michelangelo", "Pablo Picasso", "Vincent van Gogh",
    # Politics & Society
    "Democracy", "Communism", "Capitalism", "Socialism", "Fascism",
    "Human rights", "Constitution", "Law", "Government", "Politics",
    "United Nations", "European Union", "NATO", "International law",
    "Economics", "Inflation", "Gross domestic product", "Globalization",
    "Poverty", "Education", "University", "Literacy", "Public health",
    "Feminism", "Racism", "Immigration", "Climate change", "Sustainability",
    "War", "Peace", "Diplomacy", "Terrorism", "Nuclear weapon",
    # Language
    "Language", "English language", "Spanish language", "Mandarin Chinese",
    "Arabic", "French language", "Hindi", "Linguistics", "Grammar",
    "Alphabet", "Writing system", "Translation", "Etymology", "Semantics",
    # Medicine & Health
    "Medicine", "Surgery", "Anatomy", "Physiology", "Neuroscience",
    "Heart", "Brain", "Cancer", "Diabetes", "HIV/AIDS",
    "Mental health", "Depression (mood)", "Nutrition", "Exercise",
    "Pandemic", "Immune system", "Blood", "Organ (biology)", "Disease",
    # Psychology
    "Psychology", "Consciousness", "Memory", "Intelligence", "Emotion",
    "Personality psychology", "Cognitive science", "Behaviorism",
    "Sigmund Freud", "Carl Jung",
    # Food & Agriculture
    "Agriculture", "Food", "Water", "Wheat", "Rice",
    "Cooking", "Fermentation", "Coffee", "Tea", "Beer",
    # Sports
    "Olympic Games", "Football", "Basketball", "Cricket", "Tennis",
    "Baseball", "Swimming", "Athletics (sport)", "Chess", "Martial arts",
    # Space
    "Space exploration", "NASA", "International Space Station",
    "Apollo program", "Hubble Space Telescope", "Milky Way",
    "Galaxy", "Universe", "Asteroid", "Comet",
    # Animals & Nature
    "Mammal", "Bird", "Fish", "Insect", "Reptile",
    "Dinosaur", "Dog", "Cat", "Horse", "Elephant",
    "Whale", "Shark", "Lion", "Eagle", "Bee",
    "Tree", "Flower", "Forest", "Ocean", "River",
    "Climate", "Weather", "Earthquake", "Volcano", "Tsunami",
    "Biodiversity", "Endangered species", "Conservation biology",
    "Coral reef", "Rainforest",
]


def load_json(path, default):
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return default


def save_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)


def fetch_article(title: str) -> str | None:
    """Fetch article plain text via MediaWiki API."""
    params = {
        "action": "query",
        "titles": title,
        "prop": "extracts",
        "explaintext": True,
        "format": "json",
    }
    for attempt in range(3):
        try:
            resp = requests.get(WIKI_API, params=params, headers=HEADERS, timeout=30)
            if resp.status_code == 429 or resp.status_code >= 500:
                wait = 2 ** attempt * 5
                print(f"  Rate limited ({resp.status_code}), waiting {wait}s...")
                time.sleep(wait)
                continue
            resp.raise_for_status()
            pages = resp.json()["query"]["pages"]
            for page in pages.values():
                text = page.get("extract", "")
                if text and len(text) > 200:
                    return text
        except Exception as e:
            print(f"  Failed: {e}")
            time.sleep(2)
    return None


def title_to_filename(title: str) -> str:
    name = title.replace("/", "_").replace("\\", "_").replace(":", "_")
    name = re.sub(r'[<>"|?*]', "", name)
    return name[:120] + ".md"


def scrape(max_articles: int = 0, reset: bool = False):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if reset and os.path.exists(PROGRESS_FILE):
        os.remove(PROGRESS_FILE)
        print("Progress cleared.")

    progress = load_json(PROGRESS_FILE, {"done": []})
    done = set(progress["done"])

    remaining = [t for t in CURATED_ARTICLES if t not in done]
    total = len(CURATED_ARTICLES)

    print(f"Curated list: {total} articles")
    print(f"Already done:  {len(done)}")
    print(f"Remaining:     {len(remaining)}")
    print("Press Ctrl+C to stop.\n")

    count = 0
    try:
        for title in remaining:
            text = fetch_article(title)
            if text is None:
                # A timeout/rate-limit exhaustion and a missing extract both
                # arrive as None. Do not mark either as completed: leaving the
                # title out of `done` makes the next invocation retry it.
                print(f"  RETRY  {title} (not fetched)")
                continue

            filepath = os.path.join(OUTPUT_DIR, title_to_filename(title))
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(f"# {title}\n\n{text}\n")

            progress["done"].append(title)
            done.add(title)
            count += 1

            print(f"[{len(done):>3}/{total}] {title} ({len(text):,} chars)")
            save_json(PROGRESS_FILE, progress)

            if max_articles and count >= max_articles:
                print(f"\nReached max ({max_articles}).")
                break

            time.sleep(0.5)

    except KeyboardInterrupt:
        save_json(PROGRESS_FILE, progress)
        print(f"\n\nStopped. Got {count} this session, {len(done)}/{total} total.")
        print("Run again to continue.")


def main():
    parser = argparse.ArgumentParser(description="Scrape curated Wikipedia articles")
    parser.add_argument("--max", type=int, default=0, help="Max articles (0 = all)")
    parser.add_argument("--reset", action="store_true", help="Start over")
    args = parser.parse_args()
    scrape(max_articles=args.max, reset=args.reset)


if __name__ == "__main__":
    main()
