"""
Caching utilities for Streamlit applications
Provides better caching mechanisms that work with class instances
"""
import time
import streamlit as st
from typing import Any, Callable, Optional
import hashlib
import json


def cache_with_session_state(
    key_prefix: str, 
    ttl_seconds: int = 300,
    exclude_self: bool = True
):
    """
    Decorator for caching function results in session state
    
    Args:
        key_prefix: Prefix for cache key
        ttl_seconds: Time to live in seconds (default 5 minutes)
        exclude_self: Whether to exclude 'self' parameter from cache key
    """
    def decorator(func: Callable) -> Callable:
        def wrapper(*args, **kwargs):
            # Generate cache key - handle args properly
            try:
                # For global functions (no self parameter)
                cache_args = list(args)
                
                # Convert any non-serializable objects to strings
                serializable_args = []
                for arg in cache_args:
                    try:
                        json.dumps(arg)
                        serializable_args.append(arg)
                    except (TypeError, ValueError):
                        # Convert non-serializable objects to string representation
                        serializable_args.append(str(arg))
                
                cache_key_data = {
                    'func': func.__name__,
                    'prefix': key_prefix,
                    'args': serializable_args,
                    'kwargs': kwargs
                }
                
                # Create a hash of the arguments for the cache key
                cache_key_str = json.dumps(cache_key_data, sort_keys=True, default=str)
                cache_key = f"{key_prefix}_{hashlib.md5(cache_key_str.encode()).hexdigest()}"
                
            except Exception as e:
                # Fallback to simple cache key based on function name and prefix
                cache_key = f"{key_prefix}_{func.__name__}"
            
            current_time = time.time()
            
            # Check if cached result exists and is still valid
            if (cache_key in st.session_state and 
                'timestamp' in st.session_state[cache_key] and
                current_time - st.session_state[cache_key]['timestamp'] < ttl_seconds):
                return st.session_state[cache_key]['result']
            
            # Execute function and cache result
            result = func(*args, **kwargs)
            st.session_state[cache_key] = {
                'result': result,
                'timestamp': current_time
            }
            
            return result
        return wrapper
    return decorator


def clear_cache_by_prefix(prefix: str):
    """Clear all cached items with the given prefix"""
    keys_to_remove = [key for key in st.session_state.keys() if isinstance(key, str) and key.startswith(prefix)]
    for key in keys_to_remove:
        del st.session_state[key]


def get_cache_stats() -> dict:
    """Get statistics about cached items"""
    cache_keys = [key for key in st.session_state.keys() 
                  if isinstance(st.session_state[key], dict) and 
                  'timestamp' in st.session_state[key]]
    
    current_time = time.time()
    active_cache = 0
    expired_cache = 0
    
    for key in cache_keys:
        cache_item = st.session_state[key]
        # Assume default TTL of 300 seconds if not specified
        if current_time - cache_item['timestamp'] < 300:
            active_cache += 1
        else:
            expired_cache += 1
    
    return {
        'total_cache_items': len(cache_keys),
        'active_cache_items': active_cache,
        'expired_cache_items': expired_cache
    }


def cleanup_expired_cache(max_age_seconds: int = 3600):
    """Clean up expired cache items older than max_age_seconds"""
    current_time = time.time()
    keys_to_remove = []
    
    for key, value in st.session_state.items():
        if (isinstance(value, dict) and 
            'timestamp' in value and
            current_time - value['timestamp'] > max_age_seconds):
            keys_to_remove.append(key)
    
    for key in keys_to_remove:
        del st.session_state[key]
    
    return len(keys_to_remove)


# Global cache functions for common API operations
@st.cache_data(ttl=300, show_spinner=False)
def cached_health_check(base_url: str):
    """Cached health check function
    Returns backend JSON (dict) on success, otherwise None.
    """
    import requests
    try:
        response = requests.get(f"{base_url}/health", timeout=30)
        if response.status_code == 200:
            try:
                return response.json()
            except Exception:
                return {"status": "healthy"}
    except Exception:
        pass
    return None


@st.cache_data(ttl=600, show_spinner=False)
def cached_model_info(base_url: str) -> Optional[dict]:
    """Cached model info function"""
    import requests
    try:
        response = requests.get(f"{base_url}/version", timeout=30)
        if response.status_code == 200:
            return response.json()
    except Exception:
        pass
    return None


# Simple cache_data function for compatibility
def cache_data(func=None, *, ttl=None, max_entries=None, show_spinner=True, persist=None, experimental_allow_widgets=False):
    """
    Simple cache_data wrapper for compatibility
    Falls back to st.cache_data if available, otherwise uses session state caching
    """
    if func is None:
        # Called as @cache_data(ttl=300)
        def decorator(f):
            return cache_data(f, ttl=ttl, max_entries=max_entries, show_spinner=show_spinner, 
                            persist=persist, experimental_allow_widgets=experimental_allow_widgets)
        return decorator
    
    # Called as @cache_data
    try:
        # Use Streamlit's built-in cache_data if available
        return st.cache_data(func, ttl=ttl, max_entries=max_entries, show_spinner=show_spinner, 
                           persist=persist, experimental_allow_widgets=experimental_allow_widgets)
    except AttributeError:
        # Fallback to session state caching
        return cache_with_session_state(f"cache_{func.__name__}", ttl_seconds=ttl or 300)(func)