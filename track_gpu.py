import GPUtil

GPUs = GPUtil.getGPUs()

while True:
    for gpu in GPUs:
        print(gpu.id, gpu.load)