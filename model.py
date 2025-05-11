import torch
import torch.nn as nn
import torchvision.models as models

class CheXNetWithSegmentation(nn.Module):
    def __init__(self, num_classes=14):
        super(CheXNetWithSegmentation, self).__init__()
        # Load pre-trained DenseNet for classification
        self.densenet = models.densenet121(pretrained=True)
        num_ftrs = self.densenet.classifier.in_features
        self.densenet.classifier = nn.Linear(num_ftrs, num_classes)
        
        # Add U-Net for segmentation
        self.unet = UNet(in_channels=1, out_channels=1)
        
    def forward(self, x):
        # Classification
        cls_output = self.densenet(x)
        
        # Segmentation
        seg_output = self.unet(x)
        
        return cls_output, seg_output

# Option 2: Train a new model with segmentation head
class CheXNetSegmentation(nn.Module):
    def __init__(self, num_classes=14):
        super(CheXNetSegmentation, self).__init__()
        # Encoder (DenseNet)
        self.encoder = models.densenet121(pretrained=True).features
        
        # Decoder for segmentation
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
        # Classification head
        self.classifier = nn.Linear(1024, num_classes)
        
    def forward(self, x):
        # Get features
        features = self.encoder(x)
        
        # Classification
        out = torch.relu(features)
        out = torch.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)
        cls_output = self.classifier(out)
        
        # Segmentation
        seg_output = self.decoder(features)
        
        return cls_output, seg_output 