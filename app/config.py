"""Central configuration for ImageArcAI.

Values can be overridden with environment variables (e.g. MAX_UPLOAD_MB=100).
"""
from pathlib import Path
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # upload limits
    max_upload_mb: int = 50
    max_images_per_job: int = 200

    # allowed image extensions
    allowed_exts: set[str] = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}

    # model location (not bundled in this public sample)
    model_path: Path = Path("models/best.pt")

    # default output language for CSV files
    default_language: str = "hr"

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024


settings = Settings()
