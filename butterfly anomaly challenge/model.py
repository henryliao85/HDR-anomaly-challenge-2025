import os
import numpy as np
import torch
from torch import nn
from PIL import Image

import torchvision.transforms as T

# Remove any reference to cv2
# import cv2  # <--- Removed!

from open_clip import create_model

##################################################
#  U-Net model definition (Consistent with Training)
##################################################
class DoubleConv(nn.Module):
    """(Conv2d => BN => ReLU) * 2"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.block(x)

class UNet256(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super().__init__()
        # Encoder
        self.down1 = DoubleConv(in_channels, 64)
        self.down2 = DoubleConv(64, 128)
        self.down3 = DoubleConv(128, 256)
        self.down4 = DoubleConv(256, 512)
        
        # Bridge
        self.bridge = DoubleConv(512, 1024)
        
        # Decoder
        self.up1 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.conv_up1 = DoubleConv(1024, 512)
        
        self.up2 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.conv_up2 = DoubleConv(512, 256)
        
        self.up3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.conv_up3 = DoubleConv(256, 128)
        
        self.up4 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv_up4 = DoubleConv(128, 64)
        
        # Output (remove sigmoid for BCEWithLogitsLoss)
        self.out = nn.Conv2d(64, out_channels, 1)

    def forward(self, x):
        # Encoder
        c1 = self.down1(x)       # 256x256 -> 256x256
        p1 = nn.MaxPool2d(2)(c1) # 256x256 -> 128x128
        
        c2 = self.down2(p1)      # 128x128 -> 128x128
        p2 = nn.MaxPool2d(2)(c2) # 128x128 -> 64x64
        
        c3 = self.down3(p2)      # 64x64 -> 64x64
        p3 = nn.MaxPool2d(2)(c3) # 64x64 -> 32x32
        
        c4 = self.down4(p3)      # 32x32 -> 32x32
        p4 = nn.MaxPool2d(2)(c4) # 32x32 -> 16x16
        
        # Bridge
        b  = self.bridge(p4)     # 16x16 -> 16x16
        
        # Decoder
        u1 = self.up1(b)         # 16x16 -> 32x32
        u1 = torch.cat([u1, c4], dim=1)
        u1 = self.conv_up1(u1)   # 1024 -> 512
        
        u2 = self.up2(u1)        # 32x32 -> 64x64
        u2 = torch.cat([u2, c3], dim=1)
        u2 = self.conv_up2(u2)   # 512 -> 256
        
        u3 = self.up3(u2)        # 64x64 -> 128x128
        u3 = torch.cat([u3, c2], dim=1)
        u3 = self.conv_up3(u3)   # 256 -> 128
        
        u4 = self.up4(u3)        # 128x128 -> 256x256
        u4 = torch.cat([u4, c1], dim=1)
        u4 = self.conv_up4(u4)   # 128 -> 64
        
        return self.out(u4)  # Raw logits for BCEWithLogitsLoss

##################################################
# Classification Head
##################################################
class ClassificationHead(nn.Module):
    def __init__(self, in_dim, num_classes):
        super().__init__()
        self.fc = nn.Linear(in_dim, num_classes)

    def forward(self, x):
        return self.fc(x)

##################################################
# Model Class (Modified for Consistency)
#    Now using ONLY NumPy for background removal.
##################################################
class Model:
    def __init__(self):
        self.device = None
        self.model = None
        self.preprocess_img = None
        self.classifier_head = None
        
        # We'll store the U-Net for segmentation
        self.unet_model = None

    def load(self):
        # 1. Setup device
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        ft_path = os.path.join(os.path.dirname(__file__), f"fine_tuned_bioclip_select_wings.pth")
        cl_head_path = os.path.join(os.path.dirname(__file__), f"cl_head_select_wings.pth")
        # 2. Load BioCLIP weights


        self.model = create_model("hf-hub:imageomics/bioclip", output_dict=True, require_pretrained=True)
        self.model.load_state_dict(torch.load(ft_path, map_location=self.device))
        self.model = self.model.to(self.device)
        self.model.eval()  # Set to evaluation mode

        self.classifier_head = ClassificationHead(512, 15).to(self.device)
        self.classifier_head.load_state_dict(torch.load(cl_head_path, map_location=self.device))
        self.classifier_head.eval()  # Set to evaluation mode

        # 3. Image transform for classification
        self.preprocess_img = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
        ])

        # 4. Load your trained U-Net for wings (binary seg)
        self.unet_model = UNet256(in_channels=3, out_channels=1).to(self.device)
        unet_ckpt = os.path.join(os.path.dirname(__file__), f"best_unet_model.pth")
        self.unet_model.load_state_dict(torch.load(unet_ckpt, map_location=self.device))
        self.unet_model.eval()

    def predict_mask(self, pil_img):
        """
        1) Convert PIL -> np array (RGB)
        2) Resize to (256,256) to match U-Net input
        3) U-Net => binary mask (0=bg,1=wing)
        4) White background outside wings
        5) Return final PIL image (256×256)
        """
        # Convert to numpy (H,W,3)
        orig_np = np.array(pil_img.convert("RGB"))

        # We'll resize to 256x256 for the segmentation model
        resized_pil = pil_img.resize((256,256), Image.BICUBIC)

        # Convert to tensor with normalization (Consistent with Training)
        transform_seg = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], 
                        std=[0.229, 0.224, 0.225])
        ])
        image_tensor = transform_seg(resized_pil).unsqueeze(0).to(self.device)  # [1,3,256,256]

        # Forward pass through U-Net
        with torch.no_grad():
            mask_pred = self.unet_model(image_tensor)  # [1,1,256,256], raw logits

        # Apply sigmoid to obtain probabilities
        mask_prob = torch.sigmoid(mask_pred).squeeze().cpu().numpy()  # [256,256]

        # Binarize the mask (multiply by 255 to mimic typical 0/255 usage)
        binary_mask = (mask_prob > 0.5).astype(np.uint8) * 255  # [256,256]

        # Apply mask to the original resized image
        final_pil = self.remove_background_color_based(np.array(resized_pil), binary_mask)

        return final_pil

    def remove_bg(self, pil_img):
        """
        Same as predict_mask, but as a separate function if desired.
        """
        # Convert to numpy (H,W,3)
        orig_np = np.array(pil_img.convert("RGB"))

        # Resize to 256x256 for the segmentation model
        resized_pil = pil_img.resize((256,256), Image.BICUBIC)

        # Convert to tensor with normalization (Consistent with Training)
        transform_seg = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], 
                        std=[0.229, 0.224, 0.225])
        ])
        image_tensor = transform_seg(resized_pil).unsqueeze(0).to(self.device)  # [1,3,256,256]

        with torch.no_grad():
            mask_pred = self.unet_model(image_tensor)  # [1,1,256,256], raw logits

        mask_prob = torch.sigmoid(mask_pred).squeeze().cpu().numpy()  # [256,256]
        binary_mask = (mask_prob > 0.5).astype(np.uint8) * 255  # [256,256]

        final_pil = self.remove_background_color_based(np.array(resized_pil), binary_mask)
        return final_pil

    def remove_background_color_based(self, image_np, mask_np):
        """
        Removes background using NumPy masking (no cv2).
          - keep wing pixels from `image_np`
          - fill background with white
        Inputs:
            image_np: (H, W, 3)
            mask_np: (H, W) in {0, 255}
        Returns:
            PIL Image of shape (H, W, 3)
        """
        # Convert the 0/255 mask into a boolean mask
        # True where foreground should be kept, False where background
        mask_bool = (mask_np == 255)

        # Create the foreground by zeroing out background areas
        masked_foreground = image_np.copy()
        masked_foreground[~mask_bool] = 0

        # Create a white background of the same size
        white_bg = np.full(image_np.shape, 255, dtype=np.uint8)
        # Zero out the foreground area in the background
        white_bg[mask_bool] = 0

        # Combine the two
        final_image = masked_foreground + white_bg
        return Image.fromarray(final_image)

    def predict(self, datapoint):
        """
        1) Use U-Net to remove background (keep wings on white)
        2) Preprocess for classification
        3) BioCLIP + classifier => get probability distribution
        4) Return final 'hybrid' score
        """
        with torch.no_grad():
            # 1) U-Net background removal
            wings_pil = self.remove_bg(datapoint)

            # 2) Preprocess for classification
            tensor_img = self.preprocess_img(wings_pil).to(self.device).unsqueeze(0)

            # 3) Forward pass in BioCLIP
            out_dict = self.model(tensor_img)
            features = out_dict['image_features']  # shape (1,512)
            logits = self.classifier_head(features) # shape (1,15)
            probs = torch.softmax(logits, dim=1).cpu().numpy()  # shape (1,15)

            # 4) Hybrid probability
            score = self.get_hybrid_prob(probs)  # shape(1,)
        
        return float(score)

    def get_hybrid_prob(self, probs):
        """
        Your existing logic:
         - sort normal classes
         - compare with last class
        """
        # Sort all but the last class, descending
        cl_probs = np.sort(probs[:, :-1], axis=1)[:, ::-1]
        # The last class probability
        last_cl = probs[:, -1]

        # Hybrid probability: max of (last_cl) and (1.0 - (top1 - top2))
        # top1 => cl_probs[:,0], top2 => cl_probs[:,1]
        hybrid_probs = np.maximum(last_cl, 1.0 - (cl_probs[:, 0] - cl_probs[:, 1]))
        return hybrid_probs[0]  # Return scalar
