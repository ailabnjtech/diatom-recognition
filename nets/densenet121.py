import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
from torchstat import stat
from collections import OrderedDict
from torch.hub import load_state_dict_from_url

model_urls = {
    'densenet121': 'https://download.pytorch.org/models/densenet121-a639ec97.pth',
}
# 构建DenseBlock中的内部结构
# 通过语法结构，把这个当成一个layer即可.
# bottleneck + DenseBlock == > DenseNet-B

class _DenseLayer(nn.Sequential):
    # num_input_features作为输入特征层的通道数， growth_rate增长率， bn_size输出的倍数一般都是4， drop_rate判断是都进行dropout层进行处理
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate):
        super(_DenseLayer, self).__init__()
        self.add_module('norm1', nn.BatchNorm2d(num_input_features))
        self.add_module('relu1', nn.ReLU(inplace=True))
        self.add_module('conv1',
                        nn.Conv2d(num_input_features, bn_size * growth_rate, kernel_size=1, stride=1, bias=False))

        self.add_module('norm2', nn.BatchNorm2d(bn_size * growth_rate))
        self.add_module('relu2', nn.ReLU(inplace=True))
        self.add_module('conv2',
                        nn.Conv2d(bn_size * growth_rate, growth_rate, kernel_size=3, stride=1, padding=1, bias=False))

        self.drop_rate = drop_rate

    def forward(self, x):
        new_features = super(_DenseLayer, self).forward(x)
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        return torch.cat([x, new_features], 1)


# 定义Denseblock模块
class _DenseBlock(nn.Sequential):
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(num_input_features + i * growth_rate, growth_rate, bn_size, drop_rate)
            self.add_module("denselayer %d" % (i + 1), layer)


# 定义Transition层
# 负责将Denseblock连接起来，一般都有0.5的维道（通道数）的压缩
class _Transition(nn.Sequential):
    def __init__(self, num_input_features, num_output_features):
        super(_Transition, self).__init__()
        self.add_module('norm', nn.BatchNorm2d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv2d(num_input_features, num_output_features, kernel_size=1, stride=1, bias=False))
        self.add_module('pool', nn.AvgPool2d(2, stride=2))


# 实现DenseNet网络
class DenseNet(nn.Module):
    def __init__(self, growth_rate=32, block_config=(6, 12, 24, 26), num_init_features=64, bn_size=4,
                 comparession_rate=0.5, drop_rate=0, num_classes=1000):
        super(DenseNet, self).__init__()
        # 前面 卷积层+最大池化
        self.features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(3, num_init_features, kernel_size=7, stride=2, padding=3, bias=False)),
            ('norm0', nn.BatchNorm2d(num_init_features)),
            ('relu0', nn.ReLU(inplace=True)),
            ('pool0', nn.MaxPool2d(3, stride=2, padding=1))

        ]))
        # Denseblock
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(num_layers, num_features, bn_size, growth_rate, drop_rate)

            self.features.add_module("denseblock%d" % (i + 1), block)
            num_features += num_layers * growth_rate  # 确定一个DenseBlock输出的通道数

            if i != len(block_config) - 1:  # 判断是不是最后一个Denseblock
                transition = _Transition(num_features, int(num_features * comparession_rate))
                self.features.add_module("transition%d" % (i + 1), transition)
                num_features = int(num_features * comparession_rate)  # 为下一个DenseBlock的输出做准备

        # Final bn+ReLu
        self.features.add_module('norm5', nn.BatchNorm2d(num_features))
        self.features.add_module('relu5', nn.ReLU(inplace=True))

        # classification layer
        self.classifier = nn.Linear(num_features, num_classes)

        # params initalization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        features = self.features(x)
        out = F.avg_pool2d(features, 7, stride=1).view(features.size(0), -1)
        out = self.classifier(out)
        return out
    def freeze_backbone(self):
        for param in self.features.parameters():
            param.requires_grad = False

    def Unfreeze_backbone(self):
        for param in self.features.parameters():
            param.requires_grad = True



def densenet121(pretrained=False,progress=True, num_classes=1000, **kwargs):
    """DenseNet121"""
    model = DenseNet(num_init_features=64, growth_rate=32, block_config=(6, 12, 24, 16),
                     **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls['densenet121'], model_dir='./model_data',
                                              progress=progress)
        model.load_state_dict(state_dict,strict=False)
    if num_classes!=1000:
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)

    return model


# if __name__ == '__main__':
#     # 输出模型的结构
#     dense = densenet121()
#
#     # 输出模型每层的输出
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     dense = dense.to(device)
#     # summary(dense,input_size=(3,416,416), batch_size = 3)
#
#     # 每层模型的输入输出(这个能同时显示 输入输出)
#     # stat(dense, (3,416,416))import torch
