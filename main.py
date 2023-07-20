# -*- coding: utf-8 -*-
import os
import datetime
import time
import random
import cv2
import torch
import torchvision
import PIL
import pyautogui
import numpy as np
import torchvision.transforms as transforms
from pywinauto import keyboard
from utils import preprocess
import torch.nn.functional as F

def walk():
    keyboard.send_keys("{w down}")
    time.sleep(2)
    keyboard.send_keys("{w up}")
    
def sprint():
    print("sprint")
    keyboard.send_keys("{w down}")
    keyboard.send_keys("{r down}")
    time.sleep(2)
    keyboard.send_keys("{w up}")
    keyboard.send_keys("{r up}")

def right_click():
    pyautogui.mouseDown(button='right')
    time.sleep(3)
    pyautogui.mouseUp(button='right')

def left_click():
    pyautogui.mouseDown(button='left')
    time.sleep(3)
    pyautogui.mouseUp(button='left')
    print("Left clicked")

def none():
    print("N/A")

TASK = "tabs"

CATEGORIES = ['wlkfr', 'sprfr', 'rclk', 'lclk']

DATASETS = ["A"]

TRANSFORMS = transforms.Compose(
    [
        transforms.ColorJitter(0.2, 0.2, 0.2, 0.2),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)

print("{} task with {} categories defined".format(TASK, CATEGORIES))

# ================ Load Models ============================
CATEGORY_LEN = 4
MODEL_PATH = "models/my_model.pth"

device = torch.device("cuda")  # .device(0)

# RESNET 18
model = torchvision.models.resnet18(pretrained=True)
model.fc = torch.nn.Linear(512, CATEGORY_LEN)
model.load_state_dict(torch.load(MODEL_PATH))
model = model.to(device)
model.eval()

print("model configured, using ", MODEL_PATH)

# ================ Live Execution ==========================
CATEGORIES = ['wlkfr', 'sprfr', 'rclk', 'lclk']


def live(model, camera):
    # print("enter here")
    image = camera.value
    # print("camera read: ", image.shape)
    preprocessed = preprocess(image)
    # print("preprocessed image: ", preprocessed.shape)
    output = model(preprocessed)
    # print("output shape:", output.shape)
    output = F.softmax(output, dim=1)
    output = output.detach()
    output = output.cpu()
    output = output.numpy()
    output = output.flatten()
    # print("output: ", output)
    category_index = output.argmax()
    # print("category index:", category_index)
    prediction_value = CATEGORIES[category_index]
    # print("detected:", prediction_value, ", output:", [(CATEGORIES[i], prediction) for i, prediction in enumerate(output)])

    if output[category_index] > 0.70:
        print("Executing commands ", prediction_value)
        if prediction_value == "wlkfr":
            walk()
        elif prediction_value == "sprfr":
            sprint()
        elif prediction_value == "rclk":
            right_click()
        elif prediction_value == "lclk":
            left_click()
        else:
            none()
    # time.sleep(10)
    return prediction_value

t_end = datetime.datetime.now() + datetime.timedelta(seconds=100)
print("end time: ", t_end)


class FakeCamera:
    pass


fake_camera = FakeCamera()

DATA_DIR = "images/movement_A/"

images = []
for category in CATEGORIES:
     files = [DATA_DIR + category + "/" + i for i in os.listdir(DATA_DIR + category) if i.endswith(".jpg")]
     random.shuffle(files)
     images += files

#DATA_DIR = "images/"

# images = ["lclk-1.jpg", "rclk-1.jpg", "sprfr-1.jpg", "wlkfr-1.jpg"]
#images = ["rclk-1.jpg"]
    
error_count = 0
for image in images:
    print("image:", image)
    #image_path = os.path.join(DATA_DIR, image)

    #fake_camera.value = np.asarray(PIL.Image.open(image_path))
    fake_camera.value = np.asarray(PIL.Image.open(image))

    print("Start executing command...")
    prediction_value = live(model, fake_camera)
    # if prediction_value not in image:
    #     error_count += 1
    # print("detect: ", prediction_value)
    print("=" * 20)
# print(1.0 - error_count / len(images))


# display(live_execution_widget)
print("detection ends")

# ================ Exiting ============================
import os

os._exit(00)