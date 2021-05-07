import GPUtil
import time

GPUs = GPUtil.getGPUs()

while True:
    time.sleep(1)
    for gpu in GPUs:
        print(gpu.id, gpu.load)