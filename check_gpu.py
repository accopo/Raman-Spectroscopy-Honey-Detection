import torch

print("="*40)
print("PyTorch版本:", torch.__version__)
print("GPU是否可用:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("找到的GPU数量:", torch.cuda.device_count())
    print("GPU型号:", torch.cuda.get_device_name(0))
else:
    print("哎呀，没找到GPU，是不是在登录节点直接跑的？")
print("="*40)