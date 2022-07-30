import torch.nn as nn
import torchvision
import torch.nn.functional as torch_F

class Estimator(nn.Module):

    def __init__(self,opt):
        super().__init__()
        self.opt = opt
        self.feature_extractor = torchvision.models.resnet18(pretrained=True)
        feature_in = self.feature_extractor.conv1.out_channels
        self.feature_extractor.conv1 = nn.Conv2d(4, feature_in, kernel_size=7, stride=2, padding=3, bias=False)
        self.feature_extractor.fc = nn.Linear(self.feature_extractor.fc.in_features, 2)

    def forward(self,inputs):
        trig = self.feature_extractor(inputs)
        output = torch_F.normalize(trig, dim=1, p=2)
        return output