import time
import shutil
import os
import signal
import subprocess
import datetime

THRESHOLD = 97
MOUNT_POINT = '/'
CHECK_INTERVAL_SECONDS = 10
LOG_FILE = '/root/disk_monitor.log'

def log(message):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(LOG_FILE, 'a') as f:
        f.write(f"[{timestamp}] {message}\n")
    print(f"[{timestamp}] {message}")

def get_disk_usage_percent():
    total, used, free = shutil.disk_usage(MOUNT_POINT)
    return (used / total) * 100

def kill_training_process():
    try:
        # Find python processes running imitate_episodes.py
        # Using pgrep -f to match full command line
        result = subprocess.check_output(["pgrep", "-f", "imitate_episodes.py"])
        pids = [int(p) for p in result.split()]
        
        if not pids:
            log("Disk threshold reached, but no 'imitate_episodes.py' process found.")
            return

        for pid in pids:
            log(f"CRITICAL: Disk usage at {get_disk_usage_percent():.2f}%. Killing process {pid} (imitate_episodes.py)")
            try:
                os.kill(pid, signal.SIGKILL) # Force kill to ensure immediate stop
                log(f"Process {pid} killed.")
            except ProcessLookupError:
                log(f"Process {pid} already gone.")
            except Exception as e:
                log(f"Failed to kill {pid}: {e}")

    except subprocess.CalledProcessError:
        log("Disk threshold reached, but no running 'imitate_episodes.py' found (pgrep returned 1).")
    except Exception as e:
        log(f"Error finding/killing process: {e}")

def main():
    log(f"Starting Disk Safety Monitor. Threshold: {THRESHOLD}%, Check Interval: {CHECK_INTERVAL_SECONDS}s")
    
    while True:
        try:
            usage = get_disk_usage_percent()
            # log(f"Current usage: {usage:.2f}%") # Verbose logging optional
            
            if usage >= THRESHOLD:
                kill_training_process()
                # Wait a bit longer after killing to allow cleanup or prevent tight loop if deletion is slow
                time.sleep(30) 
            
        except Exception as e:
            log(f"Monitor loop error: {e}")
        
        time.sleep(CHECK_INTERVAL_SECONDS)

if __name__ == "__main__":
    main()
