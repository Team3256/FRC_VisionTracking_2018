import subprocess
import time
subprocess.Popen(["python", "stream.py"], shell=True)
time.sleep(5)
subprocess.Popen(["python", "cube_detector.py"], shell=True)
