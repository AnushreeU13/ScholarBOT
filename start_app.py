
import os
import sys
import webbrowser
from threading import Timer
import subprocess

def open_browser():
    url = "http://localhost:8501"
    print(f"Opening {url} in browser...")
    try:
        # Try chrome explicitly
        try:
            browser = webbrowser.get("google-chrome")
            browser.open(url)
        except:
            webbrowser.open(url)
    except Exception as e:
        print(f"Failed to launch browser: {e}")

def main():
    print("Starting ScholarBOT v12...")
    # Schedule browser open
    Timer(3, open_browser).start()
    
    # Run Streamlit (Force port 8501)
    # Using shell=True to allow killing via Ctrl+C if run manually
    cmd = [sys.executable, "-m", "streamlit", "run", "12_app.py", "--server.port=8501", "--server.headless=true"]
    try:
        subprocess.run(cmd)
    except KeyboardInterrupt:
        print("\nStopping...")

if __name__ == "__main__":
    main()
