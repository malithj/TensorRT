import torch
from torch2trt import torch2trt
from resnet import *


def main():
    model = resnet18(pretrained=True, quantize=True)
    model.eval().cuda()
    input = torch.rand((1, 3, 224, 224)).cuda()
    model_trt = torch2trt(model, [input])
    out = model_trt(input)
    out1 = model(input)
    print(torch.max(torch.abs(out - out1)))


if __name__ == '__main__':
    main()
