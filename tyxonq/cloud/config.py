from dotenv import load_dotenv
import os

# Load environment variables from .env file if it exists
load_dotenv()

tyxonq_base_url = os.getenv("TYXONQ_BASE_URL", "https://api.tyxonq.com/qau-cloud/tyxonq/")
tyxonq_api_version = os.getenv("TYXONQ_API_VERSION", "v1")