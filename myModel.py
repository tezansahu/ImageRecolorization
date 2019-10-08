import torch
import torch.nn as nn
import torch.nn.functional as F

class ColorRestorationNet(nn.Module):

    def __init__(self, img_dim = (256, 256)):
        super(ColorRestorationNet, self).__init__()

        # Low-level Features Network
        self.low_conv1 = nn.Conv2d(1, 8, 3, stride=2, padding=1)
        self.low_conv2 = nn.Conv2d(8, 16, 3, stride=1, padding=1)
        self.low_conv3 = nn.Conv2d(16, 16, 3, stride=2, padding=1)
        self.low_conv4 = nn.Conv2d(16, 32, 3, stride=1, padding=1)
        self.low_conv5 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.low_conv6 = nn.Conv2d(32, 64, 3, stride=1, padding=1)

        # Global Features Network
        self.global_conv1 = nn.Conv2d(64, 64, 3, stride=2, padding=1)
        self.global_conv2 = nn.Conv2d(64, 64, 3, stride=1, padding=1)
        self.global_conv3 = nn.Conv2d(64, 64, 3, stride=2, padding=1)
        self.global_conv4 = nn.Conv2d(64, 64, 3, stride=1, padding=1)
        self.global_fc1 = nn.Linear(int(img_dim[0]*img_dim[1]/16), 128)
        # self.global_fc1 = nn.Linear(img_dim[0]*img_dim[1]*32, 16)
        self.global_fc2 = nn.Linear(128, 64)
        self.global_fc3 = nn.Linear(64, 32)

        # Mid-level Features Network
        self.mid_conv1 = nn.Conv2d(64, 64, 3, stride=1, padding=1)
        self.mid_conv2 = nn.Conv2d(64, 32, 3, stride=1, padding=1)

        # Fusion Layer
        self.fusion = nn.Linear(64, 32)

        # Colorization Network
        self.color_conv1 = nn.Conv2d(32, 16, 3, stride=1, padding=1)
        self.color_conv2 = nn.Conv2d(16, 8, 3, stride=1, padding=1)
        self.color_conv3 = nn.Conv2d(8, 8, 3, stride=1, padding=1)
        self.color_conv4 = nn.Conv2d(8, 4, 3, stride=1, padding=1)

        # Output Layer
        self.output_conv = nn.Conv2d(4, 2, 3, stride=1, padding=1)

    def forward(self, input):

        # Low-level Features
        low_lev_feat = F.relu(self.low_conv1(input))
        low_lev_feat = F.relu(self.low_conv2(low_lev_feat))
        low_lev_feat = F.relu(self.low_conv3(low_lev_feat))
        low_lev_feat = F.relu(self.low_conv4(low_lev_feat))
        low_lev_feat = F.relu(self.low_conv5(low_lev_feat))
        low_lev_feat = F.relu(self.low_conv6(low_lev_feat))

        # Global Features
        glob_feat = F.relu(self.global_conv1(low_lev_feat))
        glob_feat = F.relu(self.global_conv2(glob_feat))
        glob_feat = F.relu(self.global_conv3(glob_feat))
        glob_feat = F.relu(self.global_conv4(glob_feat))
        glob_feat = glob_feat.view(-1, self.num_flat_features(glob_feat))
        glob_feat = F.relu(self.global_fc1(glob_feat))
        glob_feat = F.relu(self.global_fc2(glob_feat))
        glob_feat = F.relu(self.global_fc3(glob_feat))

        # Mid-level Features
        mid_lev_feat = F.relu(self.mid_conv1(low_lev_feat))
        mid_lev_feat = F.relu(self.mid_conv2(mid_lev_feat))

        # Fusion of Mid-level features and global features
        fused_feat = self.fuseFeatures(glob_feat, mid_lev_feat)

        # Colorization 
        color_feat = F.relu(self.color_conv1(fused_feat))
        color_feat = F.interpolate(color_feat, scale_factor=2, mode='nearest') # Upsampling
        color_feat = F.relu(self.color_conv2(color_feat))
        color_feat = F.relu(self.color_conv3(color_feat))
        color_feat = F.interpolate(color_feat, scale_factor=2, mode='nearest') # Upsampling
        color_feat = F.relu(self.color_conv4(color_feat))

        output = F.interpolate(torch.sigmoid(self.output_conv(color_feat)), scale_factor=2, mode='nearest') # Upsampling

        return output

    def fuseFeatures(self, global_features, mid_level_features):
        req_dim = mid_level_features.size()[2]
        global_features = global_features.repeat(req_dim, req_dim, 1).permute(2, 0, 1)
        fused_features = torch.cat((mid_level_features[0], global_features), dim = 0)
        fused_features = self.fusion(fused_features.permute(2, 1, 0))
        fused_features = fused_features.permute(2, 1, 0).unsqueeze(0)
        return fused_features

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1;
        for s in size:
            num_features = num_features * s
        return num_features
        
