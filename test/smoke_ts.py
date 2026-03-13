import torch
m = torch.jit.load("export_torchscript/dlv3p_mobilenetv2_fullft_seed_13__seed13.ts", map_location="cpu")
x = torch.randn(1,3,256,256)
y = m(x)
print(type(y), y.shape)
print(m.code[:400])

x = torch.randn(1, 3, 256, 256)
y1 = m(x)
y2 = m(x)
print((y1 - y2).abs().max().item())