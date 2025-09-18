"""
Performance configuration for Streamlit application
"""

# Cache configuration
CACHE_CONFIG = {
    'api_health_check_ttl': 300,  # 5 minutes
    'model_info_ttl': 600,        # 10 minutes
    'user_data_ttl': 1800,        # 30 minutes
    'static_data_ttl': 3600,      # 1 hour
}

# Session state optimization
SESSION_STATE_CONFIG = {
    'max_prediction_history': 10,  # Keep only last 10 predictions
    'cleanup_interval': 300,       # Clean up old data every 5 minutes
}

# UI optimization
UI_CONFIG = {
    'lazy_load_charts': True,      # Load charts only when visible
    'debounce_input_ms': 500,      # Debounce user input
    'pagination_size': 20,         # Items per page for large lists
}

# Memory management
MEMORY_CONFIG = {
    'max_cache_size_mb': 100,      # Maximum cache size in MB
    'auto_cleanup': True,          # Automatically cleanup old cache
}