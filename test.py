""" Visit readme.txt for help """

import os
os.environ["TF_ENABLE_ONEDNN_OPTS"]="0"
import sys
import cv2
import mediapipe as mp
import time
import numpy as np
import pyautogui
import math
import librosa
import sounddevice as sd
import tensorflow as tf
from PyQt5.QtWidgets import (QApplication, QMainWindow, QLabel, QVBoxLayout, QWidget, 
                            QPushButton, QHBoxLayout, QFrame, QSizePolicy, QMessageBox)
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QImage, QPixmap, QFont
import subprocess

# Prevent PyAutoGUI from causing issues when moving mouse
pyautogui.FAILSAFE = False

# Constants for voice recognition
SR = 16000  # Sampling rate
DURATION = 2  # Seconds
NUM_MFCC = 40
LABELS = ["up", "down", "left", "right", "zero", "one", "two", "three", "four", "five",
          "six", "seven", "eight", "nine", "go", "stop", "off", "unknown", "undefined"]

class HandDetector():
    def __init__(self, mode=False, maxHands=1, detectionCon=0.8, trackCon=0.8):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands,min_detection_confidence=self.detectionCon,min_tracking_confidence=self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils
        self.tipIds = [4, 8, 12, 16, 20]

    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks[:1]:  # Track only one hand
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)
        return cv2.flip(img, 1)

    def findPosition(self, img, handNo=0, draw=True):
        self.lmList = []
        if self.results.multi_hand_landmarks:
            if len(self.results.multi_hand_landmarks) > handNo:
                myHand = self.results.multi_hand_landmarks[handNo]
                for id, lm in enumerate(myHand.landmark):
                    h, w, _ = img.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    self.lmList.append([id, cx, cy])
                    
        return self.lmList

    def fingersStraight(self):
        def distance(p1, p2):
            return math.sqrt((p1[1] - p2[1]) ** 2 + (p1[2] - p2[2]) ** 2)

        fingers = []
        if len(self.lmList) != 0:
            # Check for thumb position - checking if it's folded or not
            if len(self.lmList) >= 5:  # Ensure we have thumb landmarks
                thumb_tip = self.lmList[4]
                thumb_ip = self.lmList[3]
                # Check if thumb is extended
                thumb_extended = thumb_tip[1] < thumb_ip[1]
                fingers.append(1 if thumb_extended else 0)  # 1 if extended, 0 if folded
            else:
                fingers.append(0)  # Default if we can't determine
                
            # Check other fingers
            for tip, base, _ in [[8, 5, 0], [12, 9, 0], [16, 13, 0], [20, 17, 0]]:
                ratio = distance(self.lmList[tip], self.lmList[base]) / distance(self.lmList[base], self.lmList[0])
                fingers.append(1 if ratio > 0.8 else 0)
        return fingers

    def isOpenPalm(self, fingers):
        # Check if all fingers are extended (open palm)
        return sum(fingers) >= 4  # Consider it an open palm if at least 4 fingers are up

class CombinedApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()
        
        # Camera setup
        self.camera_id = 0
        self.cap = None
        
        # Hand detector setup
        self.detector = HandDetector(detectionCon=0.7, trackCon=0.7)
        
        # Mouse control variables
        self.screen_w, self.screen_h = pyautogui.size()
        self.smoothening = 5
        self.plocX, self.plocY = 0, 0
        
        # Gesture recognition variables
        self.gesture_cooldown = 0
        self.cooldown_time = 20
        self.scroll_cooldown = 5
        self.holding_click = False  # Track whether mouse click is being held

        # App state
        self.running = False
        self.open_palm_detected = False
        
        # Voice model
        try:
            self.voice_model = tf.keras.models.load_model(r"C:\mini\speech_commandcomp_model.keras")
            self.voice_model_loaded = True
        except Exception as e:
            self.voice_model_loaded = False
            print(f"Failed to load voice model: {e}")
        
        # Voice recognition state
        self.is_recording = False
        self.last_command = "None"
        
        # Setup timer for video capture
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        
        # Last detected gesture for display
        self.last_gesture = "None"
        self.last_gesture_time = time.time()
        
        # App mode (0 = hand gestures, 1 = voice commands)
        self.app_mode = 0

    def initUI(self):
        # Main window setup
        self.setWindowTitle('Gesture & Voice Control App')
        self.setGeometry(100, 100, 800, 600)
        
        # Create main layout
        main_layout = QVBoxLayout()
        
        # Heading
        heading_label = QLabel("Gesture & Voice Command Control")
        heading_label.setFont(QFont('Arial', 14, QFont.Bold))
        heading_label.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(heading_label)
        
        # Video display
        self.video_frame = QFrame()
        self.video_frame.setFrameShape(QFrame.Box)
        self.video_frame.setLineWidth(2)
        self.video_frame.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        
        video_layout = QVBoxLayout()
        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setMinimumSize(640, 480)
        video_layout.addWidget(self.video_label)
        self.video_frame.setLayout(video_layout)
        
        main_layout.addWidget(self.video_frame)
        
        # Status display
        self.status_label = QLabel("Status: Ready")
        self.status_label.setFont(QFont('Arial', 10))
        self.status_label.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(self.status_label)
        
        # Button layout
        button_layout = QHBoxLayout()
        
        # Hand Gesture button
        self.hand_btn = QPushButton("Start Camera (Hand Gestures)")
        self.hand_btn.clicked.connect(self.start_hand_gesture)
        button_layout.addWidget(self.hand_btn)
        
        # Voice Command button
        self.voice_btn = QPushButton("Record (Voice Command)")
        self.voice_btn.clicked.connect(self.start_voice_command)
        button_layout.addWidget(self.voice_btn)
        
        # Stop button
        self.stop_btn = QPushButton("Stop")
        self.stop_btn.clicked.connect(self.stop_all)
        self.stop_btn.setEnabled(False)
        button_layout.addWidget(self.stop_btn)
        
        main_layout.addLayout(button_layout)
        
        # Set central widget
        central_widget = QWidget()
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)
        
        # Show the UI
        self.show()

    def start_hand_gesture(self):
        self.stop_all()  # Stop any ongoing operations
        
        # Initialize camera
        self.cap = cv2.VideoCapture(self.camera_id)
        if not self.cap.isOpened():
            QMessageBox.critical(self, "Error", "Could not open camera!")
            return
        
        self.app_mode = 0
        self.running = True
        self.timer.start(30)  # Update every 30ms (approx. 33 FPS)
        
        self.hand_btn.setEnabled(False)
        self.voice_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        
        self.status_label.setText("Status: Hand Gesture Mode Active")

    def start_voice_command(self):
        if not self.voice_model_loaded:
            QMessageBox.warning(self, "Warning", "Voice model not loaded correctly. Please check the model path.")
            return
    
        self.stop_all()  # Stop any ongoing operations
    
    # No camera initialization for voice mode
    
        self.app_mode = 1
        self.running = True
    # Don't start the timer that updates camera frames
    
        self.hand_btn.setEnabled(False)
        self.voice_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
    
        self.status_label.setText("Status: Voice Command Mode - Press Stop when done")
    
    # Start recording in a non-blocking way
        self.record_and_process_voice()

    def stop_all(self):
        if self.running:
            self.running = False
            self.timer.stop()
            
            # Release the camera
            if self.cap and self.cap.isOpened():
                self.cap.release()
            
            self.hand_btn.setEnabled(True)
            self.voice_btn.setEnabled(True)
            self.stop_btn.setEnabled(False)
            
            self.status_label.setText("Status: Ready")
            
            # Clear display
            self.video_label.clear()

    def update_frame(self):
        if self.app_mode == 0:  # Hand gesture mode
            success, img = self.cap.read()
            if not success:
                self.stop_all()
                return
            self.process_hand_gesture(img)
        else:  # Voice command mode
                # In voice mode, we don't need to update frames
            # Just show a static image or clear the display
            img = np.zeros((480, 640, 3), dtype=np.uint8)  # Black image
            self.display_voice_feedback(img)

    def process_hand_gesture(self, img):
        # Process the image for hand detection
        img = self.detector.findHands(img, draw=True)
        lmList = self.detector.findPosition(img, draw=True)
        
        # Process hand detection results
        if lmList:
            fingers = self.detector.fingersStraight()
            
            # Add finger status to image
            if len(lmList) >= 5:
                cv2.putText(img, f'Fingers: {fingers}', 
                           (10, 30), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 1)
            
            # Check for open palm - BLOCKS ALL ACTIONS
            self.open_palm_detected = self.detector.isOpenPalm(fingers)
            
            # Gesture cooldown handling
            if self.gesture_cooldown > 0:
                self.gesture_cooldown -= 1
                
            # If open palm is detected, block all actions and display message
            if self.open_palm_detected:
                self.last_gesture = "Open Palm - All Actions Blocked"
                
            # Only process gestures if NOT open palm
            elif not self.open_palm_detected:
                # Mouse movement with index finger
                if fingers[1] == 1 and sum(fingers[2:]) == 0:
                    x, y = lmList[8][1], lmList[8][2]
                    x3 = np.interp(x, (50, 640 - 50), (0, self.screen_w))
                    y3 = np.interp(y, (50, 480 - 50), (0, self.screen_h))
                    clocX = self.plocX + (x3 - self.plocX) / self.smoothening
                    clocY = self.plocY + (y3 - self.plocY) / self.smoothening
                    pyautogui.moveTo(self.screen_w - clocX, clocY)
                    self.plocX, self.plocY = clocX, clocY
                
                # Process gestures
                # INDEX+PINKY: Win+Tab (Task View)
                if fingers[1] == 1 and fingers[4] == 1 and fingers[2] == 0 and fingers[3] == 0 and self.gesture_cooldown == 0:
                    pyautogui.hotkey('win', 'tab')
                    self.gesture_cooldown = self.cooldown_time
                    self.last_gesture = "Win+Tab (Task View)"
                
                # Only MIDDLE finger: Scroll Up
                elif fingers[2] == 1 and fingers[1] == 0 and fingers[3] == 0 and fingers[4] == 0 and fingers[0] == 1 and  self.gesture_cooldown == 0:
                    pyautogui.scroll(70)
                    self.gesture_cooldown = self.scroll_cooldown
                    self.last_gesture = "Scroll Up"
                
                # Only RING finger: Scroll Down
                elif fingers[3] == 1 and fingers[1] == 0 and fingers[2] == 0 and fingers[4] == 0 and self.gesture_cooldown == 0:
                    pyautogui.scroll(-70)
                    self.gesture_cooldown = self.scroll_cooldown
                    self.last_gesture = "Scroll Down"
                
                # Only PINKY finger: Left Click
                elif fingers[4] == 1 and fingers[1] == 0 and fingers[2] == 0 and fingers[3] == 0 and fingers[0] == 1 and self.gesture_cooldown == 0:
                    pyautogui.click()
                    self.gesture_cooldown = self.cooldown_time
                    self.last_gesture = "Left Click"
                
                # Only INDEX+MIDDLE: Right Click
                elif fingers[1] == 1 and fingers[2] == 1 and fingers[3] == 0 and fingers[4] == 0 and self.gesture_cooldown == 0:
                    pyautogui.rightClick()
                    self.gesture_cooldown = self.cooldown_time
                    self.last_gesture = "Right Click"
                
                # Only MIDDLE+RING: Double Click
                elif fingers[2] == 1 and fingers[3] == 1 and fingers[1] == 0 and fingers[4] == 0 and self.gesture_cooldown == 0:
                    pyautogui.doubleClick()
                    self.gesture_cooldown = self.cooldown_time
                    self.last_gesture = "Double Click"
                
                # Only RING+PINKY: Volume Up
                elif fingers[3] == 1 and fingers[4] == 1 and fingers[1] == 0 and fingers[2] == 0 and self.gesture_cooldown == 0:
                    for _ in range(5):   
                        pyautogui.press('volumeup')
                    self.gesture_cooldown = self.cooldown_time
                    self.last_gesture = "Volume Up"
                
                # Only MIDDLE+PINKY: Volume Down
                elif fingers[2] == 1 and fingers[4] == 1 and fingers[1] == 0 and fingers[3] == 0 and self.gesture_cooldown == 0:
                    for _ in range(5):
                        pyautogui.press('volumedown')
                    self.gesture_cooldown = self.cooldown_time
                    self.last_gesture = "Volume Down"
                
                # Only MIDDLE+RING+PINKY: Screenshot
                elif fingers[2] == 1 and fingers[3] == 1 and fingers[4] == 1 and fingers[1] == 0 and fingers[0] == 0 and self.gesture_cooldown == 0:
                    pyautogui.hotkey('win', 'printscreen')
                    self.gesture_cooldown = self.cooldown_time
                    self.last_gesture = "Screenshot"

                elif fingers[0] == 0 and fingers[1] == 1 and fingers[2] == 0 and fingers[3] == 0 and fingers[4] == 0 and self.gesture_cooldown == 0 and not self.holding_click:
                    pyautogui.mouseDown()  # Hold mouse click (drag)
                    self.holding_click = True
                    self.gesture_cooldown = self.cooldown_time
                    print("Mouse Click Held (Drag)")

                # Thumb Down: Release Mouse Click
                elif fingers[0] == 1 and fingers[1] == 1 and fingers[2] == 0 and fingers[3] == 0 and fingers[4] == 0 and self.holding_click and self.gesture_cooldown == 0:
                    pyautogui.mouseUp()  # Release mouse click (drop)
                    self.holding_click = False
                    self.gesture_cooldown = self.cooldown_time
                    print("Mouse Click Released")    
                elif fingers[0] == 0 and fingers[4] == 1 and fingers[1] == 0 and fingers[2] == 0 and fingers[3] == 0 and self.gesture_cooldown == 0:
                    pyautogui.hotkey('ctrl', 'c')
                    self.gesture_cooldown = self.cooldown_time
                    self.last_gesture = "copy"
                elif fingers[0] == 0 and fingers[4] == 0 and fingers[1] == 0 and fingers[2] == 1 and fingers[3] == 0 and self.gesture_cooldown == 0:
                    pyautogui.hotkey('ctrl', 'v')
                    self.gesture_cooldown = self.cooldown_time
                    self.last_gesture = "paste"    
            # Add simple instruction guide
            cv2.putText(img, "Open Palm = BLOCK ALL ACTIONS",(10, 90), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 1)
            cv2.putText(img, "I+P: Win+Tab, M: Scroll Up, R: Scroll Down",(10, 110), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 1)
            cv2.putText(img, "P: Left Click, I+M: Right Click, M+R: Double Click",(10, 130), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 1)
            cv2.putText(img, "R+P: Volume Up, M+P: Volume Down, M+R+P: Screenshot",(10, 150), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 1)
            cv2.putText(img, "T+P: copy, T+M: paste",(10, 170), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 1)
        # Display the image
        self.display_image(img)

    def display_voice_feedback(self, img):
        # We're now using a static image for feedback
        img = np.zeros((480, 640, 3), dtype=np.uint8)  # Black background
    
        # Add a recording indicator and last command
        cv2.putText(img, "VOICE COMMAND MODE", (160, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
        if self.is_recording:
            cv2.putText(img, " Recording...", (200, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    
        cv2.putText(img, f"Last Command: {self.last_command}", (160, 180), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)   
        cv2.putText(img, "Available Commands:",(160, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
        cv2.putText(img, "up, down, left, right - Cursor Movement",(160, 280), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
        cv2.putText(img, "zero - Left Click, one - Right Click",(160, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
        cv2.putText(img, "two - Double Click, three - Volume Up",(160, 320), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
        cv2.putText(img, "four - Volume Down, five - Win+Tab",(160, 340), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
        cv2.putText(img, "six - Screenshot, seven - Scroll Up, off-recording off",(160, 360), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
        cv2.putText(img, "eight - Scroll Down, nine - Win+S",(160, 380), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
        cv2.putText(img, "go - Press Enter, stop - Pause/play",(160, 400), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)

    
        # Display the image
        self.display_image(img)


    def display_image(self, img):
        # Convert image to Qt format for display
        h, w, ch = img.shape
        bytesPerLine = ch * w
        convertToQtFormat = QImage(img.data, w, h, bytesPerLine, QImage.Format_RGB888)
        convertToQtFormat = convertToQtFormat.rgbSwapped()
        pixmap = QPixmap.fromImage(convertToQtFormat)
        self.video_label.setPixmap(pixmap)

    def record_and_process_voice(self):
        if not self.running or self.app_mode != 1:
            return
        
        try:
            self.is_recording = True
            # Update UI
            self.status_label.setText("Status: Recording audio...")
            # Update the voice feedback display
            img = np.zeros((480, 640, 3), dtype=np.uint8)
            self.display_voice_feedback(img)
            QApplication.processEvents()  # Ensure UI updates
        
            # Record audio
            audio = self.record_audio()
        
            # Process and predict
            self.is_recording = False
            self.status_label.setText("Status: Processing audio...")
            # Update the voice feedback display again
            self.display_voice_feedback(img)
            QApplication.processEvents()  # Ensure UI updates
        
            # Process audio and get prediction
            command = self.process_voice_command(audio)
            self.last_command = command
        
            # Execute the command
            self.execute_voice_command(command)
        
            # Update display with latest command
            self.display_voice_feedback(img)
        
            # Schedule next recording if still in voice mode
            if self.running and self.app_mode == 1:
                QTimer.singleShot(500, self.record_and_process_voice)  # Wait 500ms before recording again
        
        except Exception as e:
            self.is_recording = False
            self.status_label.setText(f"Status: Error - {str(e)}")
        
            # Try to continue if still in voice mode
            if self.running and self.app_mode == 1:
                QTimer.singleShot(2000, self.record_and_process_voice)

    def record_audio(self, duration=DURATION, sr=SR):
        """Records audio for a given duration and sampling rate."""
        audio = sd.rec(int(duration * sr), samplerate=sr, channels=1, dtype="float32")
        sd.wait()  # Wait for recording to finish
        audio = np.squeeze(audio)  # Convert from (N,1) to (N,)
        return audio

    def process_voice_command(self, audio):
        """Processes recorded audio and returns the predicted command."""
        # Convert audio to MFCC features
        mfccs = librosa.feature.mfcc(y=audio, sr=SR, n_mfcc=NUM_MFCC)
        mfccs = librosa.util.fix_length(mfccs, size=40, axis=1)  # Fix to (40,40)
        mfccs = np.expand_dims(mfccs, axis=-1)  # Add channel dimension
        mfccs = np.expand_dims(mfccs, axis=0)  # Add batch dimension
    
        # Predict
        prediction = self.voice_model.predict(mfccs)
        predicted_index = np.argmax(prediction)
        confidence = prediction[0][predicted_index]
    
        # Store confidence for display
        self.last_confidence = confidence
    
        # Check if confidence is below threshold
        if confidence < 0.7:
            return "unknown"
        else:
            predicted_label = LABELS[predicted_index]
            return predicted_label

    def execute_voice_command(self, command):
        """Executes actions based on voice commands."""
        # Update status with confidence value
        confidence_percent = round(self.last_confidence * 100, 1) if hasattr(self, 'last_confidence') else 0
        self.status_label.setText(f"Status: Voice Command Detected: {command} (Confidence: {confidence_percent}%)")

        if command == "up":
            pyautogui.move(0, -100)  # Move mouse up by 100 pixels
        elif command == "down":
            pyautogui.move(0, 100)  # Move mouse down by 100 pixels
        elif command == "left":
            pyautogui.move(-100, 0)  # Move mouse left by 100 pixels
        elif command == "right":
            pyautogui.move(100, 0)  # Move mouse right by 100 pixels
        elif command == "stop":
            pyautogui.press("playpause")  # Play/Pause media
        elif command == "zero":
            pyautogui.click()  # Left click
        elif command == "one":
            pyautogui.rightClick()  # Right click
        elif command == "two":
            pyautogui.doubleClick()  # Double click
        elif command == "three":
           for _ in range(5):  # Repeat 5 times (each press increases by 2)
                pyautogui.press("volumeup")
        elif command == "four":
            for _ in range(5):
                pyautogui.press("volumedown")  # Volume down
        elif command == "five":
            pyautogui.hotkey("win", "tab")  # Win+Tab
        elif command == "six":
            pyautogui.hotkey('win', 'printscreen')  # Screenshot (Windows)
        elif command == "seven":
            pyautogui.scroll(100)  # Scroll up
        elif command == "eight":
            pyautogui.scroll(-100)  # Scroll down
        elif command == "go":
            pyautogui.press("enter")  # Press Enter
        elif command == "off":
            self.stop_all()  # Turn off 
        elif command == "nine":
            pyautogui.hotkey("win", "s")  # Example additional command
        

    def closeEvent(self, event):
        # Clean up resources when closing the app
        self.stop_all()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    ex = CombinedApp()
    sys.exit(app.exec_())
    