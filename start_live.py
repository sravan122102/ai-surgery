import os
import sys
import subprocess
import threading
import time
import re
import urllib.request
from app import app

CF_URL = "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-windows-amd64.exe"
CF_EXE = "cloudflared.exe"

def download_cloudflared():
    if not os.path.exists(CF_EXE):
        print("[*] Downloading secure tunneling agent (Cloudflared)...")
        try:
            urllib.request.urlretrieve(CF_URL, CF_EXE)
            print("[*] Download complete.")
        except Exception as e:
            print(f"[!] Failed to download Cloudflared: {e}")
            sys.exit(1)

def run_flask():
    # We suppress flask logs to not clutter the cloudflare URL output, though debug mode prints some stuff.
    app.run(port=5000, use_reloader=False)

def start_tunnel():
    print("[*] Starting secure tunnel to your local machine...\n")
    # Cloudflared prints its logs to stderr, we capture both to find the URL
    process = subprocess.Popen(
        [CF_EXE, "tunnel", "--url", "http://127.0.0.1:5000"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True
    )
    
    url_found = False
    for line in iter(process.stdout.readline, ''):
        # We print lines so the user can see if there is an error, but we filter out boring info logs
        if "trycloudflare.com" in line and not url_found:
            match = re.search(r'(https://[a-zA-Z0-9-]+\.trycloudflare\.com)', line)
            if match:
                public_url = match.group(1)
                with open("live_url.txt", "w") as f:
                    f.write(public_url)
                print("\n" + "="*70)
                print("🎉 YOUR SURGISCORE PROJECT IS NOW LIVE! 🎉\n")
                print(f"🌐 PUBLIC URL: {public_url}")
                print("="*70 + "\n")
                print("Share the link above with the judges. Keep this terminal window open.\n")
                url_found = True

def main():
    download_cloudflared()
    
    # Start Flask in a background thread
    flask_thread = threading.Thread(target=run_flask, daemon=True)
    flask_thread.start()
    
    # Give Flask a couple seconds to warm up
    time.sleep(3)
    
    # Start Cloudflared tunnel
    start_tunnel()

if __name__ == '__main__':
    main()
