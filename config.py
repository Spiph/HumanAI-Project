import os

# Ollama
OLLAMA_API_URL = "http://localhost:11434/api/chat"
DEFAULT_MODEL = "gemma3:1b"

# Directories
USER_DATA_DIR = "./user_data"
os.makedirs(USER_DATA_DIR, exist_ok=True)

# MCQ generation
MAX_MCQ_GENERATION_ATTEMPTS = 5

# PDF section mapping
SECTION_HEADER_MAP = {
    "abstract": "abstract",
    "introduction": "introduction",
    #... (remaining mappings)
}

HEADER_REGEX = r"(?im)^\s*(?:\d+(?:\.\d+)*[.)]?\s*)?(?P<header>" + "|".join(SECTION_HEADER_MAP.keys()) + r")\s*$"