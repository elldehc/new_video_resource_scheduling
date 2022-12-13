from torchvision.models.detection import FasterRCNN,fasterrcnn_resnet50_fpn
from torchvision.models.resnet import resnet101,ResNet,Bottleneck
from torchvision.models.detection.backbone_utils import _resnet_fpn_extractor, _validate_trainable_layers
from torchvision.ops import misc as misc_nn_ops
from torchvision.models.feature_extraction import create_feature_extractor
from torch.nn import Identity


def fasterrcnn_resnet101_fpn(
    progress=True, num_classes=91, pretrained_backbone=True, trainable_backbone_layers=None, **kwargs
):
    trainable_backbone_layers = _validate_trainable_layers(
        pretrained_backbone, trainable_backbone_layers, 5, 3
    )
    backbone = resnet101(pretrained=pretrained_backbone, progress=progress, norm_layer=misc_nn_ops.FrozenBatchNorm2d)
    backbone = _resnet_fpn_extractor(backbone, trainable_backbone_layers)
    model = FasterRCNN(backbone, num_classes, **kwargs)
    return model

class HalfResNet(ResNet):
    def __init__(self,split_pos):
        super().__init__(Bottleneck, [3, 4, 23, 3],norm_layer=misc_nn_ops.FrozenBatchNorm2d)
        self.split_pos=split_pos
        if split_pos>=1:
            self.layer1=Identity()
        if split_pos>=2:
            self.layer2=Identity()
        if split_pos>=3:
            self.layer3=Identity()
        if split_pos>=4:
            self.layer4=Identity()
        
    def _forward_impl(self, x):
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

def splitted_resnet101(split_pos):
    assert split_pos>=1 and split_pos<=5
    model=resnet101()
    if split_pos<5:
        model1=create_feature_extractor(model,{f"layer{i}":f"layer{i}" for i in range(split_pos)})

