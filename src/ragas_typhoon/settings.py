import os
from dotenv import load_dotenv

load_dotenv(".env")

OPENAI_KEY = os.getenv("OPENAI_KEY", "")

TYPHOON_API_KEY = os.getenv("TYPHOON_API_KEY", "")

RAGAS_APP_TOKEN = os.getenv("RAGAS_APP_TOKEN", "")