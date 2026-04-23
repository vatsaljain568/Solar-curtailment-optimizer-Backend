from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    """
    Manages application configuration using Pydantic.
    Loads settings from environment variables or a .env file.
    """

    # These are like the default values for the optimizer parameters. They can be overridden by environment variables.
    COAL_MIN_MW: int = 400
    COAL_MAX_MW: int = 800
    COAL_RAMP_RATE_MW_PER_HOUR: int = 80
    TIME_STEP_MINUTES: int = 15

    CO2_TONS_PER_MWH_COAL: float = 0.98
    COST_INR_PER_MWH_COAL: float = 4000.0


settings = Settings()