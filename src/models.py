import logging
import torch
import torch.nn as nn

from torchvision import models
from log_util import log_trainable_parameters


def build_model(model_name, num_classes, device, img_size, use_batch_norm, use_dropout, dropout_rate, freeze_features, unfreeze_last_n_layers):
    feature_params = None

    model_name = model_name.lower()
    if model_name == "vgg16":
        model = VGG16(num_classes, img_size, use_batch_norm, use_dropout, dropout_rate)
    elif model_name == "vgg16_pretreinada":
        model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)        
        model.classifier[6] = nn.Linear(4096, num_classes)

        feature_params = model.features.parameters()
    elif model_name == "resnet50":
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        model.fc = nn.Linear(model.fc.in_features, num_classes)

        feature_params = model.parameters()

    elif model_name == "densenet121":
        model = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)

        feature_params = model.features.parameters()

    else:
        raise ValueError("Modelo não suportado")

    if freeze_features and unfreeze_last_n_layers and feature_params is not None:
        feature_params = list(feature_params)
        layers_n = min(unfreeze_last_n_layers, len(feature_params))
        for param in feature_params:
          param.requires_grad = False
    
        for param in feature_params[-layers_n:]:
          param.requires_grad = True

    logging.info(model)

    log_trainable_parameters(model)

    return model.to(device)



class VGG16(nn.Module):
    def __init__(self, num_classes=10, img_size=32, use_batch_norm=False, use_dropout=False, dropout_rate=0.5):
        use_batch_norm = use_batch_norm
        use_dropout = use_dropout
        dropout_rate = dropout_rate
        
        super(VGG16, self).__init__()

        def conv_block(in_channels, out_channels, num_convs):
            layers = []
            for _ in range(num_convs):
                layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
                if use_batch_norm:
                    layers.append(nn.BatchNorm2d(out_channels))
                layers.append(nn.ReLU(inplace=True))                
                in_channels = out_channels

            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            return layers

        self.features = nn.Sequential(
            *conv_block(3, 64, 2),
            *conv_block(64, 128, 2),
            *conv_block(128, 256, 3),
            *conv_block(256, 512, 3),
            *conv_block(512, 512, 3)
        )
        in_features = 512 * 7 * 7
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))        
        self.classifier = nn.Sequential(
            nn.Linear(in_features, 4096),            
            nn.ReLU(True),            
            nn.Dropout(dropout_rate) if use_dropout else nn.Identity(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(dropout_rate) if use_dropout else nn.Identity(),
            nn.Linear(4096, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

