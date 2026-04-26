from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    """
    Manages application configuration using Pydantic.
    Loads settings from environment variables or a .env file.
    """
    COAL_MIN_MW: int = 200
    COAL_MAX_MW: int = 2000
    COAL_RAMP_RATE_MW_PER_HOUR: int = 300
    TIME_STEP_MINUTES: int = 15

    CO2_TONS_PER_MWH_COAL: float = 0.98
    COST_INR_PER_MWH_COAL: float = 4580.0

settings = Settings()
