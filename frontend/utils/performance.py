"""
Performance monitoring utilities for Streamlit application
"""
import time
import streamlit as st
from functools import wraps
import psutil
import os

def performance_timer(func_name):
    """Decorator to measure function execution time"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if st.get_option("runner.fastReruns"):
                start_time = time.time()
                result = func(*args, **kwargs)
                end_time = time.time()
                
                # Store performance data in session state
                if 'performance_data' not in st.session_state:
                    st.session_state.performance_data = {}
                
                st.session_state.performance_data[func_name] = {
                    'execution_time': end_time - start_time,
                    'timestamp': time.time()
                }
                return result
            else:
                return func(*args, **kwargs)
        return wrapper
    return decorator

def show_performance_metrics():
    """Display performance metrics in sidebar"""
    if 'performance_data' in st.session_state and st.session_state.performance_data:
        with st.sidebar.expander("⚡ Performance Metrics", expanded=False):
            for func_name, data in st.session_state.performance_data.items():
                exec_time = data['execution_time']
                color = "green" if exec_time < 1 else "orange" if exec_time < 3 else "red"
                st.markdown(f"**{func_name}:** :{color}[{exec_time:.2f}s]")

def memory_usage():
    """Get current memory usage"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024  # MB

def cleanup_session_state():
    """Clean up old session state data"""
    current_time = time.time()
    
    # Remove old performance data (keep last 10 entries)
    if 'performance_data' in st.session_state:
        perf_data = st.session_state.performance_data
        if len(perf_data) > 10:
            # Keep only the 10 most recent entries
            sorted_data = sorted(perf_data.items(), key=lambda x: x[1]['timestamp'])
            st.session_state.performance_data = dict(sorted_data[-10:])
    
    # Remove old prediction history
    if 'prediction_history' in st.session_state:
        history = st.session_state.prediction_history
        if len(history) > 10:
            st.session_state.prediction_history = history[-10:]