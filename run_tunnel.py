"""Keeps the public tunnel up on one fixed address and brings it back if it drops.

localtunnel hands out a random address when the requested name is still held by its
server (which happens for a few seconds after a drop), so this retries until it gets
the address the website is configured to call (frontend/public/config.json).
"""
import re
import subprocess
import sys
import time

SUBDOMAIN = "fodpcb-predictive-api"
WANT = f"https://{SUBDOMAIN}.loca.lt"
PORT = 8000
NPX = "npx.cmd" if sys.platform == "win32" else "npx"


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def kill_tree(proc):
    if sys.platform == "win32":
        subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)], capture_output=True)
    else:
        proc.kill()


while True:
    log(f"starting tunnel, want {WANT}")
    proc = subprocess.Popen([NPX, "localtunnel", "--port", str(PORT), "--subdomain", SUBDOMAIN],
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    right_address = None
    for line in proc.stdout:
        m = re.search(r"your url is: (\S+)", line)
        if not m:
            log(line.strip())
            continue
        right_address = m.group(1).rstrip("/") == WANT
        if right_address:
            log(f"tunnel is up on {WANT}")
        else:
            log(f"got {m.group(1)} instead, the name is still held by the server; retrying")
            kill_tree(proc)
            break
    kill_tree(proc)
    wait = 12 if right_address is False else 3
    log(f"tunnel stopped, restarting in {wait} s")
    time.sleep(wait)
