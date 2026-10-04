from app.core.config import get_settings
print("Loading settings...")
settings = get_settings()   
print(f"Settings loaded: {settings.app_name}, Environment: {settings.app_env}, OpenAI Model: {settings.openai_model}")