import torch

# 是否有 CUDA 可用
print("CUDA 可用:", torch.cuda.is_available())

# CUDA 版本
print("CUDA 版本:", torch.version.cuda)

# GPU 数量
print("GPU 数量:", torch.cuda.device_count())

# 当前设备名称
for i in range(torch.cuda.device_count()):
    print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
