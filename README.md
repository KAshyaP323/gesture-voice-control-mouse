# gesture-voice-control-mouse

Mouse control using hand gestures (MediaPipe) and voice commands (TensorFlow) with a PyQt5 interface.

---

## 📌 Project Overview

This project provides a desktop application that allows controlling a computer using:

- **Hand gestures** (camera + MediaPipe + OpenCV)
- **Voice commands** (MFCC features + CNN model)

The GUI front-end and runtime are implemented in **test.py**.  
The voice model is trained using **train.py**.

---

## 🎯 Use-Cases
- Accessibility (hands-free computer control)  
- Presentation control  
- Media control and desktop automation  
- Prototyping gesture- and voice-driven UIs  

---

## ✨ Features

### 🖐️ Hand Gesture Control  
Real-time hand tracking and gesture recognition using MediaPipe + OpenCV.

Supports:
- Move mouse cursor  
- Left click, right click, double click  
- Scroll up/down  
- Volume up/down  
- Screenshot  
- Win+Tab  
- Copy/paste  
- Drag and drop  
- **Open Palm → Block all actions (safety override)**

### 🎤 Voice Command Control  
Using microphone input → MFCC extraction (librosa) → TensorFlow CNN model.

Supports:
- Cursor movement  
- Clicks (left/right/double)  
- Volume control  
- Screenshot  
- Scroll  
- Play/pause  
- Win+S, Win+Tab  
- Stop/Exit

### 🖥️ GUI (PyQt5)
- Live camera feed  
- Gesture info overlay  
- Status indicators  
- Start/Stop mode buttons  
- Voice command feedback  

---

## 📂 Repository Layout:
-├── test.py # Main GUI application
-├── train.py # Model training script
-├── requirements.txt # Recommended Python dependencies
-├── README.md # This file
-└── model/ # Saved model(s)



---

## 🎮 Controls & Gesture Reference

### ✋ Hand Gestures

| Gesture | Action |
|--------|--------|
| Index finger only | Move cursor |
| Pinky | Left click |
| Index + Middle | Right click |
| Middle + Ring | Double click |
| Middle | Scroll up |
| Ring | Scroll down |
| Ring + Pinky | Volume up |
| Middle + Pinky | Volume down |
| Middle + Ring + Pinky | Screenshot |
| Index (hold) | Drag (mouseDown) |
| Index + Thumb | Release drag (mouseUp) |
| **Open Palm** | **Block all actions** |

---

### 🎤 Voice Commands

| Command | Action |
|---------|--------|
| up / down / left / right | Move cursor |
| zero | Left click |
| one | Right click |
| two | Double click |
| three | Volume up |
| four | Volume down |
| five | Win+Tab |
| six | Screenshot |
| seven | Scroll up |
| eight | Scroll down |
| nine | Win+S |
| go | Press Enter |
| stop | Play/pause |
| off | Exit application |




