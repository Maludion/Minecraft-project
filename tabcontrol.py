import pyautogui

screenWidth, screenHeight = pyautogui.size()
currentMouseX, currentMouseY = pyautogui.position()

pyautogui.click(button='right')
pyautogui.click(button='left')

pyautogui.press('w')
pyautogui.press('r')
