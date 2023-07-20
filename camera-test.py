from jetcam.jetcam.usb_camera import USBCamera


# for USB Camera (Logitech C270 webcam), uncomment the following line
camera = USBCamera(width=224, height=224, capture_device=0) # confirm the capture_device number

# camera.running = True
print("camera created")

import torchvision.transforms as transforms
from dataset import ImageClassificationDataset

TASK = 'movement'

CATEGORIES = ['wlkfr', 'sprfr', 'rclk', 'lclk']

DATASETS = ['A', 'B']

TRANSFORMS = transforms.Compose([
    transforms.ColorJitter(0.2, 0.2, 0.2, 0.2),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

print("{} task with {} categories defined".format(TASK, CATEGORIES))

import torch
import torchvision

MODEL_PATH = ""

device = torch.device('cuda')

# RESNET 18
model = torchvision.models.resnet18(pretrained=True)
model.fc = torch.nn.Linear(512, len(CATEGORIES))
    
model = model.to(device)

model.load_state_dict(torch.load(MODEL_PATH))

# display(model_widget)
print("model configured and model_widget created")