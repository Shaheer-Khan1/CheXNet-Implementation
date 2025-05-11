import torch
import torch.nn as nn
import torchvision.models as models

class CheXNetWithSegmentation(nn.Module):
    def __init__(self, num_classes=14):
        super(CheXNetWithSegmentation, self).__init__()
        # Load the base DenseNet model
        self.densenet = models.densenet121(pretrained=True)
        
        # Classification head
        self.classifier = nn.Linear(1024, num_classes)
        
        # Segmentation head with skip connections
        self.segmentation_head = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
        # Upsampling layers
        self.upsample = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        )
        
    def forward(self, x):
        # Get features from DenseNet
        features = self.densenet.features(x)
        
        # Classification
        out = torch.relu(features)
        out = torch.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)
        classification_output = self.classifier(out)
        
        # Segmentation
        seg_features = self.segmentation_head(features)
        segmentation_output = self.upsample(seg_features)
        
        return classification_output, segmentation_output 