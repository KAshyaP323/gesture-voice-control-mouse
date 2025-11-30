# gesture-voice-control-mouse
Mouse control using hand gestures (MediaPipe) and voice commands (TensorFlow) with a PyQt5 interface.

Project overview

This project provides a desktop application that allows controlling a computer using hand gestures (camera + MediaPipe) and voice commands (MFCC features + CNN model). The GUI front-end and runtime are implemented in test.py.

Use-cases:
Accessibility (hands-free control)
Presentation control
Media control and desktop automation
Prototyping gesture- and voice-driven UIs


Features
Real-time hand tracking and gestures via MediaPipe + OpenCV.
Voice command capture using microphone, MFCC extraction (librosa), and a trained TensorFlow model.
GUI implemented in PyQt5 for mode switching, status, and video display.
Common commands: cursor movement, left/right/double click, scroll, volume controls, screenshot, Win+Tab, play/pause, and more.
Safety feature: open-palm blocks all actions.
Training script uses SpeechCommands dataset (or local dataset) to build a speech classifier.


Repository layout
.
├── test.py              # Main GUI application 
├── requirements.txt     # (recommended) list of Python packages
├── README.md            # This file  
└── model/               # Saved model(s) 


Controls & gestures reference

Hand Gestures 
Index finger only — Move mouse cursor
Pinky — Left click
Index + Middle — Right click
Middle + Ring — Double click
Middle — Scroll up
Ring — Scroll down
Ring + Pinky — Volume up
Middle + Pinky — Volume down
Middle + Ring + Pinky — Screenshot
Index only (hold) — MouseDown (drag)
Index + Thumb — MouseUp (release)
Open Palm — Block all actions (safety override)

Voice commands
up, down, left, right — move cursor
zero — left click
one — right click
two — double click
three — volume up
four — volume down
five — Win+Tab
six — screenshot
seven — scroll up
eight — scroll down
nine — Win+S
go — press Enter
stop — play/pause
off — stop application
