Minecraft Player


This project allows for code to play Minecraft for you. It uses image recognition to identify which function you would like to use which includes walking, sprinting, right-clicking, and left-clicking. This allows for a fun new way to play the game and can help with things like afk farms or even just trying to play through the game with an added difficulty. 

You need pytorch, torch vision, and pyautogui installed for this. Make sure that you have images prepared with Jupyter so that they can be analized. You first make the def fuctions of your commands in order to bind them to keys. Then you want to create lists with your task, commands, and datasets. After that, you want to establish your resnet 18 with torch. Then you will want to create another def function for all your outputs, and an if statement sending your images to the key bind def functions. 

All you need to do is run the code with "python .\main.py" and then switch to minecraft so that it plays. 

