"""
ARC-FL ML Service Configuration
Loads all environment variables and provides typed access throughout the app.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env from the ml-service directory
load_dotenv(Path(__file__).parent.parent / ".env")


class Config:
    # Storage paths
    STORAGE_ROOT: str = os.getenv("STORAGE_ROOT", "storage")
    VALIDATION_DATASET_PATH: str = os.getenv("VALIDATION_DATASET_PATH", "")

    # Security
    ML_SERVICE_API_KEY: str = os.getenv("ML_SERVICE_API_KEY", "")

    # Model architecture constants (CICIoT2023 dataset)
    MODEL_FEATURES: int = int(os.getenv("MODEL_FEATURES", "45"))
    MODEL_CLASSES: int = int(os.getenv("MODEL_CLASSES", "34"))

    # Service
    PORT: int = int(os.getenv("PORT", "8000"))
    HOST: str = os.getenv("HOST", "127.0.0.1")

    # Derived paths
    @property
    def global_models_dir(self) -> Path:
        return Path(self.STORAGE_ROOT) / "global-models"

    @property
    def candidates_dir(self) -> Path:
        path = Path(self.STORAGE_ROOT) / "candidates"
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def client_updates_dir(self) -> Path:
        return Path(self.STORAGE_ROOT) / "client-updates"


config = Config()
