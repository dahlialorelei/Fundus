try:
    import os
    from PIL import Image
    from torchvision.transforms import Compose, ToTensor, Resize
    import torch
    import torch.nn as nn
    import torchvision
    from torch.optim import lr_scheduler
    import torch.nn.functional as F
    from torchvision.transforms import Compose, RandomHorizontalFlip, Grayscale, Resize, RandomCrop, ToTensor
    from torch.utils.data import RandomSampler, DataLoader
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import sklearn
    import random
    import pickle
    from datetime import datetime
    import sys

except Exception as e:
    print(f"Error: {e}")
    print("Please use python 3.11 and install the required libraries using requirements.txt")
    exit(1)

RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed_all(RANDOM_SEED)

if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))  # If CUDA is available

# Define hyperparameters
LEARNING_RATE = 0.00001
BATCH_SIZE = 42
NUM_EPOCHS = 30
DATA_DIR = "Fundus_PartialDataSet"
RES_X = 640
RES_Y = 400
MASK_X = 32
MASK_Y = 20
DO_TEST = True
TIMING = False
SAVE_PICKLE = True
DO_PREPROCESS = True

# Class that applies the Sobel filter to an image
class SobelFilter(nn.Module):
    def __init__(self):
        super(SobelFilter, self).__init__()
        # Define Sobel kernels
        sobel_kernel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        sobel_kernel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        
        self.register_buffer('sobel_kernel_x', sobel_kernel_x)
        self.register_buffer('sobel_kernel_y', sobel_kernel_y)

    def forward(self, x):
        sobel_kernel_x = self.sobel_kernel_x.to(x.device)
        sobel_kernel_y = self.sobel_kernel_y.to(x.device)

        # Apply reflect padding to the input to avoid border effects
        x_padded = F.pad(x, (1, 1, 1, 1), mode='reflect')

        # Apply Sobel filter in x and y directions
        grad_x = F.conv2d(x_padded, sobel_kernel_x, padding=0)
        grad_y = F.conv2d(x_padded, sobel_kernel_y, padding=0)

        # Combine gradients
        grad = torch.sqrt(grad_x ** 2 + grad_y ** 2)
        sobel_output = grad * 8;
        
        # Change any values greater than 1 to 1
        clamped_sobel_output = torch.clamp(sobel_output, 0, 1)
        
        return clamped_sobel_output


# Class that preprocesses the raw dataset
# Loads images and masks, applies transformations, and returns them as tensors
class FundusDataset_Preprocess(torch.utils.data.Dataset):
    def __init__(self, data_dir, device, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.sobel_filter = SobelFilter()
        self.device = device

        # Find all images and masks
        self.image_files = {
            f: f for f in os.listdir(data_dir) if '_output_' in f and f.endswith('.png')
        }
        self.mask_files = {
            f: f for f in os.listdir(data_dir) if '_mask_' in f and f.endswith('.png')
        }

        # Match images with their corresponding masks based on the shared prefix and suffix
        self.common_keys = []
        for img_name in self.image_files.keys():
            prefix_suffix = img_name.split('_output_')
            if len(prefix_suffix) == 2:
                mask_name = f"{prefix_suffix[0]}_mask_{prefix_suffix[1]}"
                if mask_name in self.mask_files:
                    self.common_keys.append((img_name, mask_name))


    def __len__(self):
        return len(self.common_keys)

    def __getitem__(self, idx):
        img_name, mask_name = self.common_keys[idx]
        img_path = os.path.join(self.data_dir, img_name)
        mask_path = os.path.join(self.data_dir, mask_name)

        image = Image.open(img_path).convert('RGBA')
        original_image = np.array(image)

        mask = Image.open(mask_path).convert('L')

        # Resize the image
        transformed_image = self.transform(image)

        # Normalize the image
        image_array = np.array(transformed_image, dtype=np.float32)  # Convert to NumPy array for processing
        image_min, image_max = image_array.min(), image_array.max()
        normalized_image = (image_array - image_min) / (image_max - image_min + 1e-8)
        normalized_image = torch.tensor(normalized_image, dtype=torch.float32).permute(2, 0, 1)  # Convert back to Tensor

        normalized_image_device = normalized_image.to(self.device)

        # Convert Sobel image to greyscale
        greyscale_image_device = normalized_image_device.mean(dim=0, keepdim=True)

        # Apply Sobel filter
        sobel_image_device = self.sobel_filter(greyscale_image_device.unsqueeze(0)).squeeze(0)  # Add batch dimension for Sobel filter

        sobel_image = sobel_image_device.cpu()
        
        # Apply resizing and transforms to mask
        mask = Resize((MASK_Y, MASK_X))(mask) 
        mask = ToTensor()(mask)
        mask = (mask > 0.5).float()

        return sobel_image, mask, self.common_keys[idx]
    

def main_preprocess(in_data_dir, out_data_dir, device):
    print("Preprocessing images in", in_data_dir)
    print("Outputting to", out_data_dir)

    # Create output directory if it doesn't exist
    if not os.path.exists(out_data_dir):
        os.makedirs(out_data_dir)

    # Transformations for the dataset
    transform = Compose([
        Resize((RES_Y, RES_X))
    ])
    dataset = FundusDataset_Preprocess(in_data_dir, device, transform)

    for i in range(len(dataset)):
        # Save the Sobel image and mask
        sobel_image, mask, names = dataset[i]
        sobel_image = sobel_image.cpu().numpy()

        # Check if the output file already exists
        output_name = os.path.join(out_data_dir, f"sobel_{names[0]}")
        if os.path.exists(output_name):
            continue

        mask = mask.cpu().numpy()

        # Transpose the array to (640, 400, 1)
        sobel_image = (sobel_image * 255).astype(np.uint8).squeeze()
        mask = (mask * 255).astype(np.uint8).squeeze()
        
        sobel_image = Image.fromarray(sobel_image)
        mask = Image.fromarray(mask)

        sobel_image.save(output_name)
        mask.save(os.path.join(out_data_dir, f"{names[1]}"))


# Class that preprocesses the sobel transformed dataset
# Loads images and masks, applies transformations, and returns them as tensors
class FundusDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, device, include_orig=False):
        self.data_dir = data_dir
        self.device = device
        self.include_orig = include_orig

        # Find all images and masks
        self.image_files = {
            f: f for f in os.listdir(data_dir) if '_output_' in f and 'sobel_' in f and f.endswith('.png')
        }
        
        # Only use part of the data
        #self.image_files = dict(list(self.image_files.items())[0:300])

        print("Loading ", len(self.image_files), " images")

        self.mask_files = {
            f: f for f in os.listdir(data_dir) if '_mask_' in f and f.endswith('.png')
        }
        if include_orig:
            self.orig_image_files = {
                f: f for f in os.listdir(data_dir) if '_output_' in f and 'sobel_' not in f and f.endswith('.png')
            }

        # Match images with their corresponding masks based on the shared prefix and suffix
        self.common_keys = []
        for img_name in self.image_files.keys():
            prefix_suffix = img_name.split('_output_')
            if len(prefix_suffix) == 2:
                mask_name = f"{prefix_suffix[0][6:]}_mask_{prefix_suffix[1]}"
                if mask_name not in self.mask_files:
                    print("Cannot find mask ",mask_name, " for ", img_name)
                    exit(1)
                if include_orig:
                    orig_name = img_name[6:]
                    if orig_name in self.orig_image_files:
                        self.common_keys.append((img_name, mask_name, orig_name))
                    else:
                        print("Cannot find original image ", orig_name, " for ", img_name)
                        exit(1)
                else:
                    self.common_keys.append((img_name, mask_name))

        # Calculate the proportion of ones in each mask
        # Used for balancing the dataset during training
        self.one_proportions = []
        for keys in self.common_keys:
            if self.include_orig:
                _, mask_name, _ = keys
            else:
                _, mask_name = keys
            mask_path = os.path.join(self.data_dir, mask_name)
            mask = Image.open(mask_path).convert('L')
            mask = Resize((RES_Y, RES_X))(mask) # Resize the mask to match the image
            mask = ToTensor()(mask)
            proportion = (mask > 0.5).float().mean().item()
            self.one_proportions.append(proportion)

        self.one_proportions = torch.tensor(self.one_proportions)

    def __len__(self):
        return len(self.common_keys)

    def __getitem__(self, idx):
        # Load the image and mask
        if (self.include_orig):
            img_name, mask_name, orig_name = self.common_keys[idx]
            orig_path = os.path.join(self.data_dir, orig_name)
        else:
            img_name, mask_name = self.common_keys[idx]

        img_path = os.path.join(self.data_dir, img_name)
        mask_path = os.path.join(self.data_dir, mask_name)

        # Convert the image to grayscale
        image = Image.open(img_path).convert('L')
        np_image = np.array(image)
        image_t = (torch.tensor(np_image, dtype=torch.uint8).float() / 255.0).unsqueeze(0) # Scale pixel values to [0, 1]

        # Resize and binarize the mask
        orig_mask = Image.open(mask_path).convert('L') # Convert to grayscale
        mask = Resize((RES_Y, RES_X))(orig_mask) # Resize the mask to match the image
        mask = ToTensor()(mask)
        mask = (mask > 0.5).float() # Binarize the mask

        if (self.include_orig):
            # Save and return the original image for visualization
            # Used in Test.py
            orig_image = Image.open(orig_path).convert('RGBA')
            np_orig_image = np.array(orig_image)
            orig_image_t = torch.tensor(np_orig_image, dtype=torch.uint8).permute(2, 0, 1)
            orig_mask_t = ToTensor()(orig_mask)
            return image_t, orig_mask_t, orig_image_t

        # Return the image and mask tensors
        return image_t, mask

class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, features=[64, 128, 256, 512]):
        super(UNet, self).__init__()

        # Encoder (Downsampling)
        self.encoders = nn.ModuleList()
        for feature in features:
            self.encoders.append(self.double_conv(in_channels, feature))
            in_channels = feature
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Bottleneck
        self.bottleneck = self.double_conv(features[-1], features[-1] * 2)

        # Decoder (Upsampling + Skip connections)
        self.upconvs = nn.ModuleList()
        self.decoders = nn.ModuleList()
        for feature in reversed(features):
            self.upconvs.append(nn.ConvTranspose2d(feature * 2, feature, kernel_size=2, stride=2))
            self.decoders.append(self.double_conv(feature * 2, feature))

        # Final convolution (output to match mask dimensions)
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []

        # Encoder path
        for encoder in self.encoders:
            x = encoder(x)
            skip_connections.append(x)
            x = self.pool(x)

        # Bottleneck
        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]  # Reverse skip connections

        # Decoder path
        for i in range(len(self.upconvs)):
            x = self.upconvs[i](x)
            skip_connection = skip_connections[i]
            if x.shape != skip_connection.shape:  # Handle mismatched shapes due to padding
                x = F.interpolate(x, size=skip_connection.shape[2:], mode="bilinear", align_corners=False)
            x = torch.cat((skip_connection, x), dim=1)
            x = self.decoders[i](x)

        # Final output
        x = self.final_conv(x)
        x = F.interpolate(x, size=(MASK_Y, MASK_X), mode="bilinear", align_corners=False) # Resize to match the original mask dimensions
        return x

    @staticmethod
    def double_conv(in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

def train(model, train_loader, loss_function, optimizer, device):
    log("Training...")
    count = 0
    model.train()
    running_loss = 0.0
    for images, masks in train_loader:
        images, masks = images.to(device), masks.to(device)
        
        # Forward pass
        if (TIMING):
            starttime = datetime.now()
            print("starting model at ", starttime.strftime("%Y%m%d-%H%M%S"))

        outputs = model(images)
        outputs = torch.clamp(outputs, min=-10, max=10)  # Prevent extreme logits

        if (TIMING):
            endtime = datetime.now() 
            print("model done at ", endtime.strftime("%Y%m%d-%H%M%S") + " took ", endtime - starttime)
        
        masks_resized = F.interpolate(masks, size=(MASK_Y, MASK_X), mode="bilinear", align_corners=False)
        loss = loss_function(outputs, masks_resized)
        
        # Backward pass and optimization
        optimizer.zero_grad()

        if (TIMING):
            starttime = datetime.now()
            print("starting backward at ", starttime.strftime("%Y%m%d-%H%M%S"))

        loss.backward()

        if (TIMING):
            endtime = datetime.now()
            print("backward done at ", endtime.strftime("%Y%m%d-%H%M%S") + " took ", endtime - starttime)
        
        optimizer.step()

        running_loss += loss.item()
        count += 1
        log(f"Batch {count}, Loss: {loss.item():.4f}")

    return running_loss / len(train_loader)

# Define a helper function for validation
def validate(model, val_loader, loss_function, device):
    log("Validating...")
    model.eval()
    val_loss = 0.0
    total_iou = 0.0
    total_dice = 0.0
    total_pixel_accuracy = 0.0
    total_fpr = 0.0
    count = 0

    with torch.no_grad():
        for images, masks in val_loader:
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            outputs = torch.clamp(outputs, min=-10, max=10)  # Prevent extreme logits
            masks = F.interpolate(masks, size=(MASK_Y, MASK_X), mode="bilinear", align_corners=False)

            loss = loss_function(outputs, masks)

            # Calculate metrics
            val_loss += loss.item()
            total_iou += iou_score(outputs, masks).item()
            total_dice += dice_score(outputs, masks).item()
            total_pixel_accuracy += pixel_accuracy(outputs, masks).item()
            total_fpr += false_positive_rate(outputs, masks)
           
            count += 1
            log(f"Batch {count}, Loss: {loss.item():.4f}")
        
    num_batches = len(val_loader)
    return (
        # Average metrics
        val_loss / num_batches,
        total_iou / num_batches,
        total_dice / num_batches,
        total_pixel_accuracy / num_batches,
        total_fpr / num_batches, 
    )

# IOU calculates the intersection over union of the predicted mask and the target mask
def iou_score(preds_, targets, threshold=0.5):
    preds = (preds_ > threshold).float()
    intersection = (preds * targets).sum()
    union = preds.sum() + targets.sum() - intersection
    return intersection / union

# DSC score calculated as twice the intersection divided by the area of the predicted and target masks
def dice_score(preds_, targets, threshold=0.5):
    preds = (preds_ > threshold).float()
    intersection = (preds * targets).sum()
    return (2. * intersection) / (preds.sum() + targets.sum())

# Pixel accuracy calculates the proportion of correctly predicted pixels
def pixel_accuracy(preds_, targets, threshold=0.5):
    preds = (preds_ > threshold).float()
    correct = (preds == targets).sum()
    total = targets.numel()
    return correct / total

# False positive rate calculates the proportion of pixels incorrectly predicted as foreground
def false_positive_rate(predictions_, targets):
    # Binarize predictions at a threshold of 0.5
    predictions = (predictions_ > 0.5).float()
    
    # Flatten tensors for computation
    predictions = predictions.view(-1)
    targets = targets.view(-1)
    
    # Calculate False Positives (FP) and True Negatives (TN)
    fp = ((predictions == 1) & (targets == 0)).sum().item()
    tn = ((predictions == 0) & (targets == 0)).sum().item()
    
    # Avoid division by zero
    if fp + tn == 0:
        return 0.0
    
    return fp / (fp + tn)


datetimestr = datetime.now().strftime("%Y%m%d-%H%M%S")

# Focal Loss: https://arxiv.org/pdf/1708.02002.pdf
# Mitigates the problem of class imbalance
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)  # Probabilities
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss
        return F_loss.mean()

# Dice Loss
# Measures the overlap between the predicted and target masks
class DiceLoss(nn.Module):
    def forward(self, inputs, targets, smooth=1):
        inputs = torch.sigmoid(inputs)  # Convert logits to probabilities
        intersection = (inputs * targets).sum()
        dice = (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)
        return 1 - dice

# Focal Dice Loss: Combination of Focal Loss and Dice Loss
# Used to mitigate class imbalance and measure overlap between masks
class FocalDiceLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2):
        super(FocalDiceLoss, self).__init__()
        self.focal = FocalLoss(alpha, gamma)
        self.dice = DiceLoss()

    def forward(self, inputs, targets):
        return 0.5 * self.focal(inputs, targets) + 0.5 * self.dice(inputs, targets)

# Function to log messages to the console and a file
def log(*args):
    print(*args)
    with open('logs/log_' + datetimestr + '.txt', 'a') as f:
        print(*args, file=f)

# Main function
def main(learning_rate, num_epochs, data_dir, continue_from, device):

    log("Current Timestamp =", datetimestr)
    log("Running ", sys.argv[0])

    log("  LEARNING_RATE", LEARNING_RATE)
    log("  NUM_EPOCHS", NUM_EPOCHS)
    log("  DATA_DIR", DATA_DIR)
    log("  RES_X", RES_X)
    log("  RES_Y", RES_Y)
    log("  MASK_X", MASK_X)
    log("  MASK_Y", MASK_Y)
    log("  DO_TEST", DO_TEST)
    log("  RANDOM_SEED", RANDOM_SEED)
    log("  BATCH_SIZE", BATCH_SIZE)
    log("  continue_from", continue_from)
    log("  CUDA available", torch.cuda.is_available())

    # Make logs and models directories if they don't exist
    if not os.path.exists('logs'):
        os.makedirs('logs')
    if not os.path.exists('models'):
        os.makedirs('models')

    # Load the dataset
    dataset = FundusDataset(data_dir, device)
    train_size = int(0.7 * len(dataset))
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size], generator=torch.Generator().manual_seed(RANDOM_SEED))

    # Print the sizes of the training and validation datasets
    log(f"Training dataset size: {len(train_dataset)}")
    log(f"Validation dataset size: {len(val_dataset)}")

    # Data loaders
    train_weights = dataset.one_proportions[train_dataset.indices] # Using the one_proportions tensor to balance the dataset
    sampler = torch.utils.data.WeightedRandomSampler(train_weights, len(train_weights))
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    def init_weights(m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            #nn.init.xavier_uniform_(m.weight)
            nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    # Initialize model, loss function, and optimizer
    base_model_name = "models/model_file"
    if (continue_from):
        log("Continuing training...")
        base_model_name = "models/" + continue_from
        # if name ends with _torch, load with torch
        if continue_from.endswith('_torch'):
            model = torch.load(base_model_name, map_location=torch.device(device))
        else:
            model = pickle.load(open(base_model_name, 'rb')).to(device)
        log (f"Continuing training from {continue_from}")
    else:
        model = UNet(in_channels=1, out_channels=1).to(device)
        model.apply(init_weights)
        log (f"Starting new training")


    loss_function = FocalDiceLoss() # Using Focal Dice Loss
    
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5) # Using Adam optimizer
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2) # Reduce learning rate on plateau

    # Training and validation
    train_losses, val_losses, iou_scores, dice_scores, pixel_accuracies, fprs = [], [], [], [], [], [] 

    for epoch in range(num_epochs):
        train_loss = train(model, train_loader, loss_function, optimizer, device)
        val_loss, val_iou, val_dice, val_pixel, fpr = validate(model, val_loader, loss_function, device)
        scheduler.step(train_loss)

        # Save metrics
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        iou_scores.append(val_iou)
        dice_scores.append(val_dice)
        pixel_accuracies.append(val_pixel)
        fprs.append(fpr)

        '''
        if(SAVE_PICKLE):
            pickle.dump(model, open(base_model_name + '_' + datetimestr + "_" + str(epoch+1), 'wb')) # Save the model 
        torch.save(model, base_model_name + '_' + datetimestr + "_" + str(epoch+1) + '_torch') # Save the model
        '''

        # Print loss, IoU, Dice, Pixel Accuracy, and FPR score after every epoch
        log(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, IoU: {val_iou:.4f}, Dice: {val_dice:.4f}, Pixel Accuracy: {val_pixel:.4f}, FPR: {fpr:.4f}')
        
        # Show current time
        log(f"Current Time = {datetime.now().strftime('%H:%M:%S')}")

    if(SAVE_PICKLE):
        pickle.dump(model, open(base_model_name + '_' + datetimestr + '_final', 'wb')) # Save the model 
    torch.save(model, base_model_name + '_' + datetimestr + "_final" + '_torch') # Save the model

    # Plot training and validation loss
    plt.figure(figsize=(16, 4))
    plt.subplot(1, 4, 1)
    plt.plot(range(1, num_epochs + 1), train_losses, label='Training Loss')
    plt.plot(range(1, num_epochs + 1), val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    
    plt.subplot(1, 4, 2)
    plt.plot(range(1, num_epochs + 1), iou_scores, label='IoU Score')
    plt.plot(range(1, num_epochs + 1), dice_scores, label='Dice Score')
    plt.xlabel('Epochs')
    plt.ylabel('Score')
    plt.legend()
    plt.title('IoU and Dice Scores')

    plt.subplot(1, 4, 3)
    plt.plot(range(1, num_epochs + 1), pixel_accuracies, label='Pixel Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Pixel Accuracy')

    plt.subplot(1, 4, 4)
    plt.plot(range(1, num_epochs + 1), fprs, label='FPR Score')
    plt.xlabel('Epochs')
    plt.ylabel('Score')
    plt.legend()
    plt.title('FPR Score')
    
    plt.tight_layout()

    os.makedirs('graphs', exist_ok=True)

    # Save the figure to a file
    basename = 'graphs/training_validation_metrics'
    plt_filename = f'{basename}_{datetimestr}.png'
    plt.savefig(plt_filename)
    print(f"Plot saved as {plt_filename}")
    #plt.show()


    # Testing the model
    if (DO_TEST):
        model.eval()
        correct = 0
        total = 0
        val_loss = 0.0
        total_iou = 0.0 
        total_dice = 0.0
        total_fpr = 0.0
        total_pixel_accuracy = 0.0

        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(device), masks.to(device)
                outputs = model(images)
                masks = F.interpolate(masks, size=(MASK_Y, MASK_X), mode="bilinear", align_corners=False) # Resize masks to match the output
                
                outputs = torch.sigmoid(outputs)  # Apply sigmoid to convert logits to probabilities
                loss = loss_function(outputs, masks)

                # Compute metrics
                val_loss += loss.item()
                total_iou += iou_score(outputs, masks).item()
                total_dice += dice_score(outputs, masks).item()
                total_pixel_accuracy += pixel_accuracy(outputs, masks).item()
                total_fpr += false_positive_rate(outputs, masks)
                predictions = (outputs > 0.5).float()
                correct += (predictions == masks).sum().item()
                total += masks.numel()

        # Calculate average loss, IoU, Dice, Pixel Accuracy, and FPR
        val_loss /= len(val_loader)
        total_iou /= len(val_loader)
        total_dice /= len(val_loader)
        total_pixel_accuracy /= len(val_loader)
        total_fpr /= len(val_loader)

        log(f'Test Loss: {val_loss:.4f}, IoU: {total_iou:.4f}, Dice: {total_dice:.4f}, Pixel Accuracy: {total_pixel_accuracy:.4f}, FPR: {total_fpr:.4f}')


def print_syntax_train():
    print("Usage: python Training.py [data_dir]")
    print("   data_dir: The directory containing the test data")
    print("   Example: python Test.py Fundus_PartialDataSet")

if __name__ == "__main__":
    python_version = sys.version.split()[0]
    python_major_version = python_version.split('.')[0]
    python_minor_version = python_version.split('.')[1]

    if python_major_version != '3' or python_minor_version not in ['10','11']:
        print(f'Your python version is: {python_version}')
        print("Please use Python 3.11")
        exit(1)

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log(f'Using device: {device}')

    # Create models and logs directories if they don't already exist
    if not os.path.exists('models'):
        os.makedirs('models')
    if not os.path.exists('logs'):
        os.makedirs('logs')

    # If a data directory is specified on the command line, use it
    if len(sys.argv) > 1:
        DATA_DIR = sys.argv[1]
    
    # If the data directory doesn't exist
    if not os.path.exists(DATA_DIR):
        print(f"Data directory {DATA_DIR} does not exist")
        print_syntax_train()
        exit(1)
    
    # Preprocess the dataset
    if DO_PREPROCESS:
        main_preprocess(DATA_DIR, DATA_DIR, device)

    # Train the model
    main(LEARNING_RATE, NUM_EPOCHS, DATA_DIR, False, device)