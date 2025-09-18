#!/usr/bin/env python3
"""
Install Medical Analysis Dependencies
Handles installation of medical NLP dependencies with fallbacks
"""

import subprocess
import sys
import os

def install_package(package):
    """Install a package using pip"""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        return True
    except subprocess.CalledProcessError:
        print(f"Failed to install {package}")
        return False

def main():
    """Install medical analysis dependencies"""
    print("🏥 Installing Medical Analysis Dependencies...")
    
    # Core dependencies (install first)
    core_deps = [
        "spacy>=3.6.1,<3.7.0",
        "scispacy>=0.5.3", 
        "nltk>=3.8.1",
        "reportlab>=4.0.0",
        "requests-cache>=1.1.0",
        "sentence-transformers>=2.2.0",
        "textstat>=0.7.3",
        "wordcloud>=1.9.2",
        "fuzzywuzzy>=0.18.0",
        "python-Levenshtein>=0.21.0"
    ]
    
    print("Installing core dependencies...")
    for dep in core_deps:
        print(f"Installing {dep}...")
        install_package(dep)
    
    # Install spaCy models
    print("\nInstalling spaCy models...")
    spacy_models = [
        "https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.3/en_core_sci_sm-0.5.3.tar.gz",
        "en_core_web_sm"  # Fallback model
    ]
    
    for model in spacy_models:
        print(f"Installing {model}...")
        if not install_package(model):
            print(f"Skipping {model} - will use fallback")
    
    # Optional dependencies
    optional_deps = [
        "medspacy>=1.0.0"
    ]
    
    print("\nInstalling optional dependencies...")
    for dep in optional_deps:
        print(f"Installing {dep}...")
        if not install_package(dep):
            print(f"Skipping {dep} - functionality will be limited")
    
    print("\n✅ Installation complete!")
    print("Note: Some advanced features may be limited if optional dependencies failed to install.")

if __name__ == "__main__":
    main()
