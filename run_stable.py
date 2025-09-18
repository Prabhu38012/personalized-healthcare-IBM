#!/usr/bin/env python3
"""
Stable Healthcare System Startup Script
Ensures services stay running without premature shutdown
"""
import subprocess
import sys
import time
import os
import signal
from pathlib import Path

def start_backend_stable():
    """Start the backend server with stable configuration"""
    print("🚀 Starting Healthcare Backend Server (Stable Mode)...")
    print("📍 Available at: http://localhost:8000")
    print("📚 API Documentation: http://localhost:8000/docs")
    print("-" * 50)
    
    # Use direct uvicorn command without reload for stability
    backend_cmd = [
        sys.executable, "-m", "uvicorn",
        "backend.app:app",
        "--host", "0.0.0.0",
        "--port", "8000",
        "--workers", "1",
        "--timeout-keep-alive", "30"
    ]
    
    # Set environment variables for stability
    env = os.environ.copy()
    env['PYTHONUNBUFFERED'] = '1'
    env['PYTHONDONTWRITEBYTECODE'] = '1'
    
    return subprocess.Popen(
        backend_cmd, 
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
        bufsize=1
    )

def start_frontend_stable():
    """Start the frontend server with stable configuration"""
    print("🎨 Starting Healthcare Frontend (Stable Mode)...")
    print("📍 Available at: http://localhost:8501")
    print("-" * 50)
    
    frontend_cmd = [
        sys.executable, "-m", "streamlit",
        "run", "frontend/app.py",
        "--server.port", "8501",
        "--server.headless", "true",
        "--server.enableCORS", "false",
        "--server.enableXsrfProtection", "false"
    ]
    
    # Set environment variables for stability
    env = os.environ.copy()
    env['PYTHONUNBUFFERED'] = '1'
    env['STREAMLIT_SERVER_ENABLE_CORS'] = 'false'
    
    return subprocess.Popen(
        frontend_cmd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
        bufsize=1
    )

def monitor_process(name, process):
    """Monitor a process and log its output"""
    try:
        while True:
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break
            if output:
                print(f"[{name}] {output.strip()}")
    except Exception as e:
        print(f"[{name}] Monitor error: {e}")

def main():
    """Main startup function with enhanced stability"""
    print("🏥 Healthcare System Launcher (Stable)")
    print("=" * 50)
    
    # Check if we're in the right directory
    if not Path("backend/app.py").exists():
        print("❌ Error: Please run this script from the project root directory")
        sys.exit(1)
    
    # Kill any existing processes on these ports
    print("🧹 Cleaning up existing processes...")
    try:
        subprocess.run(["taskkill", "/F", "/IM", "python.exe"], 
                      capture_output=True, check=False)
        time.sleep(2)
    except:
        pass
    
    try:
        choice = input("Choose startup mode:\n1. Backend only\n2. Frontend only\n3. Both (recommended)\nEnter choice (1-3): ").strip()
        
        processes = []
        
        if choice in ["1", "3"]:
            backend_process = start_backend_stable()
            processes.append(("Backend", backend_process))
            print("⏳ Waiting for backend to initialize...")
            time.sleep(5)  # Give backend more time to start
        
        if choice in ["2", "3"]:
            frontend_process = start_frontend_stable()
            processes.append(("Frontend", frontend_process))
        
        if not processes:
            print("❌ Invalid choice. Exiting.")
            return
        
        print("\n✅ Services started successfully!")
        print("🔄 Monitoring services for stability...")
        print("Press Ctrl+C to stop all services")
        
        # Enhanced monitoring loop
        try:
            consecutive_checks = 0
            while True:
                all_running = True
                
                for name, process in processes:
                    if process.poll() is not None:
                        print(f"⚠️ {name} process stopped unexpectedly (exit code: {process.returncode})")
                        # Try to read any remaining output
                        try:
                            remaining_output = process.stdout.read()
                            if remaining_output:
                                print(f"[{name}] Final output: {remaining_output}")
                        except:
                            pass
                        all_running = False
                        break
                
                if not all_running:
                    print("❌ Service failure detected. Exiting.")
                    break
                
                consecutive_checks += 1
                if consecutive_checks % 30 == 0:  # Every 30 seconds
                    print(f"✅ Services running stable for {consecutive_checks} seconds")
                
                time.sleep(1)
                
        except KeyboardInterrupt:
            print("\n🛑 Stopping services...")
            for name, process in processes:
                try:
                    # Send SIGTERM first
                    process.terminate()
                    
                    # Wait up to 10 seconds for graceful shutdown
                    try:
                        process.wait(timeout=10)
                        print(f"✅ {name} stopped gracefully")
                    except subprocess.TimeoutExpired:
                        # Force kill if needed
                        process.kill()
                        process.wait()
                        print(f"🔥 {name} force stopped")
                        
                except Exception as e:
                    print(f"⚠️ Error stopping {name}: {e}")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
