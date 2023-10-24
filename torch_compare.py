import torch
import time

cpu_file = open("cpu_time.txt", "a")
gpu_file = open("gpu_time.txt", "a")
for i in range(8180, 8192):
  #print(i)
  A = torch.randn(i, i)
  A.to(device="cpu")
  start_time = time.time()
  B = torch.inverse(A)
  cpu_time = (time.time() - start_time) * 1000
  #cpu_file.write(f"{cpu_time:.2f}\n")
  #print(f"{cpu_time:.2f}")
  A.to(device="cuda:0")
  start_time = time.time()
  B = torch.inverse(A)
  gpu_time = (time.time() - start_time) * 1000
  #gpu_file.write(f"{gpu_time:.2f}\n")
  print(f"{gpu_time:.2f}, {cpu_time:.2f}")
  
cpu_file.close()
gpu_file.close()