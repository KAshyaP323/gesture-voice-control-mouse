"""
Gesture + Voice Controlled Virtual Mouse
Updated for the 19-class CNN exported by the Google Colab notebook.

Place these files in the SAME directory:
    virtual_mouse_app.py
    voice_command_inference.py
    voice_command_cnn.keras
    preprocessing_config.json
    class_labels.json

Install the desktop dependencies in your local application environment:
    pip install opencv-python mediapipe pyautogui sounddevice PyQt5 numpy scipy tensorflow

The CNN is trained and exported in Google Colab. This application only loads
the exported model and performs local inference; it does not train the model.
"""

import os
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import sys
import time
import math
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import pyautogui
import sounddevice as sd

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QLabel, QVBoxLayout, QWidget,
    QPushButton, QHBoxLayout, QFrame, QSizePolicy, QMessageBox
)
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QImage, QPixmap, QFont

# Import the exact preprocessing + inference implementation exported by Colab.
from voice_command_inference import VoiceCommandPredictor


# ============================================================
# PATHS AND VOICE MODEL CONFIGURATION
# ============================================================

APP_DIR = Path(__file__).resolve().parent

MODEL_PATH = APP_DIR / "voice_command_cnn.keras"
CONFIG_PATH = APP_DIR / "preprocessing_config.json"
LABELS_PATH = APP_DIR / "class_labels.json"

SAMPLE_RATE = 16000
AUDIO_DURATION = 2.0
AUDIO_CHANNELS = 1

TARGET_COMMANDS = {
    "up", "down", "left", "right", "stop",
    "zero", "one", "two", "three", "four",
    "five", "six", "seven", "eight", "nine",
    "go", "off"
}

NON_ACTION_CLASSES = {"unknown", "silence"}

# Voice command policy:
# Execute every threshold-approved command prediction.
# Repeated commands are allowed on consecutive recording cycles.
REQUIRE_CONSECUTIVE_PREDICTIONS = 1
VOICE_REARM_ON_UNKNOWN_OR_SILENCE = False

# PyAutoGUI safety configuration.
pyautogui.FAILSAFE = True
pyautogui.PAUSE = 0.05


# ============================================================
# HAND GESTURE DETECTOR
# ============================================================

class HandDetector:

    def __init__(
        self,
        mode=False,
        maxHands=1,
        detectionCon=0.8,
        trackCon=0.8
    ):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(
            static_image_mode=self.mode,
            max_num_hands=self.maxHands,
            min_detection_confidence=self.detectionCon,
            min_tracking_confidence=self.trackCon
        )

        self.mpDraw = mp.solutions.drawing_utils
        self.tipIds = [4, 8, 12, 16, 20]
        self.results = None
        self.lmList = []

    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks[:1]:
                if draw:
                    self.mpDraw.draw_landmarks(
                        img, handLms, self.mpHands.HAND_CONNECTIONS
                    )

        return cv2.flip(img, 1)

    def findPosition(self, img, handNo=0, draw=True):
        self.lmList = []

        if self.results and self.results.multi_hand_landmarks:
            if len(self.results.multi_hand_landmarks) > handNo:
                myHand = self.results.multi_hand_landmarks[handNo]

                for landmark_id, lm in enumerate(myHand.landmark):
                    h, w, _ = img.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    self.lmList.append([landmark_id, cx, cy])

        return self.lmList

    def fingersStraight(self):

        def distance(p1, p2):
            return math.sqrt(
                (p1[1] - p2[1]) ** 2 +
                (p1[2] - p2[2]) ** 2
            )

        fingers = []

        if len(self.lmList) >= 21:

            # Thumb extension (same orientation rule as the original project).
            thumb_tip = self.lmList[4]
            thumb_ip = self.lmList[3]
            thumb_extended = thumb_tip[1] < thumb_ip[1]
            fingers.append(1 if thumb_extended else 0)

            # Index, middle, ring, pinky.
            for tip, base in [(8, 5), (12, 9), (16, 13), (20, 17)]:
                reference = distance(self.lmList[base], self.lmList[0])

                if reference == 0:
                    fingers.append(0)
                    continue

                ratio = (
                    distance(self.lmList[tip], self.lmList[base])
                    / reference
                )
                fingers.append(1 if ratio > 0.8 else 0)

        return fingers

    def isOpenPalm(self, fingers):
        return len(fingers) == 5 and sum(fingers) >= 4

    def close(self):
        try:
            self.hands.close()
        except Exception:
            pass


# ============================================================
# MAIN APPLICATION
# ============================================================

class CombinedApp(QMainWindow):

    def __init__(self):
        super().__init__()

        self.initUI()

        # Camera.
        self.camera_id = 0
        self.cap = None

        # Hand detector.
        self.detector = HandDetector(
            detectionCon=0.7,
            trackCon=0.7
        )

        # Mouse movement.
        self.screen_w, self.screen_h = pyautogui.size()
        self.smoothening = 5
        self.plocX, self.plocY = 0, 0

        # Gesture cooldowns.
        self.gesture_cooldown = 0
        self.cooldown_time = 20
        self.scroll_cooldown = 5
        self.holding_click = False

        # Application state.
        self.running = False
        self.app_mode = 0
        self.open_palm_detected = False

        # Load the Colab-trained voice model and matching preprocessing.
        self.voice_predictor = None
        self.voice_model_loaded = False
        self.voice_model_error = None
        self.load_voice_model()

        # Voice state.
        self.is_recording = False
        self.last_command = "None"
        self.last_confidence = 0.0
        self.last_raw_prediction = "None"

        self.previous_voice_prediction = None
        self.voice_prediction_streak = 0
        self.voice_armed = True

        # Display state.
        self.last_gesture = "None"
        self.last_gesture_time = time.time()

        # Video timer.
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)

    # --------------------------------------------------------
    # UI
    # --------------------------------------------------------

    def initUI(self):

        self.setWindowTitle("Gesture & Voice Control App")
        self.setGeometry(80, 40, 1200, 900)

        main_layout = QVBoxLayout()

        heading_label = QLabel(
            "Gesture & Voice Command Control"
        )
        heading_label.setFont(QFont("Arial", 14, QFont.Bold))
        heading_label.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(heading_label)

        self.video_frame = QFrame()
        self.video_frame.setFrameShape(QFrame.Box)
        self.video_frame.setLineWidth(2)
        self.video_frame.setSizePolicy(
            QSizePolicy.Expanding,
            QSizePolicy.Expanding
        )

        video_layout = QVBoxLayout()

        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setMinimumSize(1000, 700)

        video_layout.addWidget(self.video_label)
        self.video_frame.setLayout(video_layout)
        main_layout.addWidget(self.video_frame)

        self.status_label = QLabel("Status: Ready")
        self.status_label.setFont(QFont("Arial", 10))
        self.status_label.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(self.status_label)

        button_layout = QHBoxLayout()

        self.hand_btn = QPushButton(
            "Start Camera (Hand Gestures)"
        )
        self.hand_btn.clicked.connect(
            self.start_hand_gesture
        )
        button_layout.addWidget(self.hand_btn)

        self.voice_btn = QPushButton(
            "Start Voice Commands"
        )
        self.voice_btn.clicked.connect(
            self.start_voice_command
        )
        button_layout.addWidget(self.voice_btn)

        self.stop_btn = QPushButton("Stop")
        self.stop_btn.clicked.connect(self.stop_all)
        self.stop_btn.setEnabled(False)
        button_layout.addWidget(self.stop_btn)

        main_layout.addLayout(button_layout)

        central_widget = QWidget()
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)

        self.show()

    # --------------------------------------------------------
    # LOAD CNN MODEL
    # --------------------------------------------------------

    def load_voice_model(self):

        required_files = [
            MODEL_PATH,
            CONFIG_PATH,
            LABELS_PATH
        ]

        missing = [
            str(path.name)
            for path in required_files
            if not path.exists()
        ]

        if missing:
            self.voice_model_error = (
                "Missing model files: " + ", ".join(missing)
            )
            print(self.voice_model_error)
            return

        try:
            self.voice_predictor = VoiceCommandPredictor(
                model_path=str(MODEL_PATH),
                config_path=str(CONFIG_PATH)
            )

            self.voice_model_loaded = True

            print("Voice CNN loaded successfully.")
            print("Confidence threshold:",
                  self.voice_predictor.threshold)
            print("Classes:", self.voice_predictor.labels)

        except Exception as exc:
            self.voice_model_loaded = False
            self.voice_model_error = str(exc)
            print("Failed to load voice CNN:", exc)

    # --------------------------------------------------------
    # START HAND GESTURE MODE
    # --------------------------------------------------------

    def start_hand_gesture(self):

        self.stop_all()

        self.cap = cv2.VideoCapture(self.camera_id)

        if not self.cap.isOpened():
            QMessageBox.critical(
                self,
                "Camera Error",
                "Could not open camera!"
            )
            self.cap = None
            return

        self.app_mode = 0
        self.running = True

        self.timer.start(30)

        self.hand_btn.setEnabled(False)
        self.voice_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)

        self.status_label.setText(
            "Status: Hand Gesture Mode Active"
        )

    # --------------------------------------------------------
    # START VOICE MODE
    # --------------------------------------------------------

    def start_voice_command(self):

        if not self.voice_model_loaded:
            QMessageBox.warning(
                self,
                "Voice Model Error",
                "The CNN model could not be loaded.\n\n"
                f"{self.voice_model_error or 'Check model files.'}\n\n"
                "Keep voice_command_cnn.keras, "
                "preprocessing_config.json, "
                "class_labels.json and "
                "voice_command_inference.py beside this script."
            )
            return

        self.stop_all()

        self.app_mode = 1
        self.running = True

        self.previous_voice_prediction = None
        self.voice_prediction_streak = 0
        self.voice_armed = True

        self.hand_btn.setEnabled(False)
        self.voice_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)

        self.status_label.setText(
            "Status: Voice Mode Active"
        )

        self.last_command = "Waiting..."
        self.display_voice_feedback()

        # Start the first recording without blocking the button callback.
        QTimer.singleShot(
            100,
            self.record_and_process_voice
        )

    # --------------------------------------------------------
    # STOP ALL MODES
    # --------------------------------------------------------

    def stop_all(self):

        self.running = False
        self.timer.stop()

        if self.cap is not None:
            if self.cap.isOpened():
                self.cap.release()
            self.cap = None

        if self.holding_click:
            try:
                pyautogui.mouseUp()
            except Exception:
                pass
            self.holding_click = False

        self.is_recording = False

        self.hand_btn.setEnabled(True)
        self.voice_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)

        self.status_label.setText("Status: Ready")
        self.video_label.clear()

        self.previous_voice_prediction = None
        self.voice_prediction_streak = 0
        self.voice_armed = True

    # --------------------------------------------------------
    # CAMERA UPDATE
    # --------------------------------------------------------

    def update_frame(self):

        if not self.running or self.app_mode != 0:
            return

        if self.cap is None:
            self.stop_all()
            return

        success, img = self.cap.read()

        if not success:
            self.stop_all()
            return

        self.process_hand_gesture(img)

    # --------------------------------------------------------
    # HAND GESTURE PROCESSING
    # --------------------------------------------------------

    def process_hand_gesture(self, img):

        img = self.detector.findHands(img, draw=True)
        lmList = self.detector.findPosition(img)

        if lmList:

            fingers = self.detector.fingersStraight()

            if len(fingers) != 5:
                self.display_image(img)
                return

            cv2.putText(
                img,
                f"Fingers: {fingers}",
                (10, 30),
                cv2.FONT_HERSHEY_PLAIN,
                1,
                (255, 0, 0),
                1
            )

            self.open_palm_detected = (
                self.detector.isOpenPalm(fingers)
            )

            if self.gesture_cooldown > 0:
                self.gesture_cooldown -= 1

            if self.open_palm_detected:

                self.last_gesture = (
                    "Open Palm - All Actions Blocked"
                )

            else:

                # Cursor movement using index finger.
                if fingers[1] == 1 and sum(fingers[2:]) == 0:

                    x, y = lmList[8][1], lmList[8][2]

                    x3 = np.interp(
                        x, (50, 590), (0, self.screen_w)
                    )
                    y3 = np.interp(
                        y, (50, 430), (0, self.screen_h)
                    )

                    clocX = (
                        self.plocX +
                        (x3 - self.plocX) / self.smoothening
                    )
                    clocY = (
                        self.plocY +
                        (y3 - self.plocY) / self.smoothening
                    )

                    pyautogui.moveTo(
                        self.screen_w - clocX,
                        clocY
                    )

                    self.plocX, self.plocY = clocX, clocY

                # Win + Tab.
                if (
                    fingers[1] == 1 and fingers[4] == 1
                    and fingers[2] == 0 and fingers[3] == 0
                    and self.gesture_cooldown == 0
                ):
                    pyautogui.hotkey("win", "tab")
                    self.gesture_cooldown = self.cooldown_time
                    self.last_gesture = "Win+Tab"

                # Scroll up.
                elif (
                    fingers[2] == 1 and fingers[1] == 0
                    and fingers[3] == 0 and fingers[4] == 0
                    and fingers[0] == 1
                    and self.gesture_cooldown == 0
                ):
                    pyautogui.scroll(70)
                    self.gesture_cooldown = self.scroll_cooldown
                    self.last_gesture = "Scroll Up"

                # Scroll down.
                elif (
                    fingers[3] == 1 and fingers[1] == 0
                    and fingers[2] == 0 and fingers[4] == 0
                    and self.gesture_cooldown == 0
                ):
                    pyautogui.scroll(-70)
                    self.gesture_cooldown = self.scroll_cooldown
                    self.last_gesture = "Scroll Down"

                # Left click.
                elif (
                    fingers[4] == 1 and fingers[1] == 0
                    and fingers[2] == 0 and fingers[3] == 0
                    and fingers[0] == 1
                    and self.gesture_cooldown == 0
                ):
                    pyautogui.click()
                    self.gesture_cooldown = self.cooldown_time
                    self.last_gesture = "Left Click"

                # Right click.
                elif (
                    fingers[1] == 1 and fingers[2] == 1
                    and fingers[3] == 0 and fingers[4] == 0
                    and self.gesture_cooldown == 0
                ):
                    pyautogui.rightClick()
                    self.gesture_cooldown = self.cooldown_time
                    self.last_gesture = "Right Click"

                # Double click.
                elif (
                    fingers[2] == 1 and fingers[3] == 1
                    and fingers[1] == 0 and fingers[4] == 0
                    and self.gesture_cooldown == 0
                ):
                    pyautogui.doubleClick()
                    self.gesture_cooldown = self.cooldown_time
                    self.last_gesture = "Double Click"

                # Volume up.
                elif (
                    fingers[3] == 1 and fingers[4] == 1
                    and fingers[1] == 0 and fingers[2] == 0
                    and self.gesture_cooldown == 0
                ):
                    for _ in range(5):
                        pyautogui.press("volumeup")
                    self.gesture_cooldown = self.cooldown_time
                    self.last_gesture = "Volume Up"

                # Volume down.
                elif (
                    fingers[2] == 1 and fingers[4] == 1
                    and fingers[1] == 0 and fingers[3] == 0
                    and self.gesture_cooldown == 0
                ):
                    for _ in range(5):
                        pyautogui.press("volumedown")
                    self.gesture_cooldown = self.cooldown_time
                    self.last_gesture = "Volume Down"

                # Screenshot.
                elif (
                    fingers[2] == 1 and fingers[3] == 1
                    and fingers[4] == 1 and fingers[1] == 0
                    and fingers[0] == 0
                    and self.gesture_cooldown == 0
                ):
                    pyautogui.hotkey("win", "printscreen")
                    self.gesture_cooldown = self.cooldown_time
                    self.last_gesture = "Screenshot"

                # Drag start.
                elif (
                    fingers[0] == 0 and fingers[1] == 1
                    and fingers[2] == 0 and fingers[3] == 0
                    and fingers[4] == 0
                    and self.gesture_cooldown == 0
                    and not self.holding_click
                ):
                    pyautogui.mouseDown()
                    self.holding_click = True
                    self.gesture_cooldown = self.cooldown_time
                    self.last_gesture = "Drag Started"

                # Drag release.
                elif (
                    fingers[0] == 1 and fingers[1] == 1
                    and fingers[2] == 0 and fingers[3] == 0
                    and fingers[4] == 0
                    and self.holding_click
                    and self.gesture_cooldown == 0
                ):
                    pyautogui.mouseUp()
                    self.holding_click = False
                    self.gesture_cooldown = self.cooldown_time
                    self.last_gesture = "Drag Released"

                # Copy.
                elif (
                    fingers[0] == 0 and fingers[4] == 1
                    and fingers[1] == 0 and fingers[2] == 0
                    and fingers[3] == 0
                    and self.gesture_cooldown == 0
                ):
                    pyautogui.hotkey("ctrl", "c")
                    self.gesture_cooldown = self.cooldown_time
                    self.last_gesture = "Copy"

                # Paste.
                elif (
                    fingers[0] == 0 and fingers[4] == 0
                    and fingers[1] == 0 and fingers[2] == 1
                    and fingers[3] == 0
                    and self.gesture_cooldown == 0
                ):
                    pyautogui.hotkey("ctrl", "v")
                    self.gesture_cooldown = self.cooldown_time
                    self.last_gesture = "Paste"

            cv2.putText(
                img, "Open Palm = BLOCK ALL ACTIONS",
                (10, 90), cv2.FONT_HERSHEY_PLAIN,
                1, (0, 0, 255), 1
            )
            cv2.putText(
                img, "I+P: Win+Tab | M: Scroll Up | R: Scroll Down",
                (10, 110), cv2.FONT_HERSHEY_PLAIN,
                1, (0, 0, 255), 1
            )
            cv2.putText(
                img, "P: Left Click | I+M: Right Click | M+R: Double Click",
                (10, 130), cv2.FONT_HERSHEY_PLAIN,
                1, (0, 0, 255), 1
            )
            cv2.putText(
                img, "R+P: Volume Up | M+P: Volume Down",
                (10, 150), cv2.FONT_HERSHEY_PLAIN,
                1, (0, 0, 255), 1
            )
            cv2.putText(
                img, "M+R+P: Screenshot | T+P: Copy | T+M: Paste",
                (10, 170), cv2.FONT_HERSHEY_PLAIN,
                1, (0, 0, 255), 1
            )

        self.display_image(img)

    # --------------------------------------------------------
    # VOICE FEEDBACK DISPLAY
    # --------------------------------------------------------

    def display_voice_feedback(self):

        # Larger voice-mode dashboard with command names and their actions.
        img = np.zeros((700, 1100, 3), dtype=np.uint8)

        cv2.putText(
            img, "VOICE COMMAND MODE",
            (365, 45), cv2.FONT_HERSHEY_SIMPLEX,
            1.1, (255, 255, 255), 2
        )

        status = "RECORDING..." if self.is_recording else "LISTENING"
        cv2.putText(
            img, status,
            (480, 90), cv2.FONT_HERSHEY_SIMPLEX,
            0.75, (0, 255, 255), 2
        )

        cv2.putText(
            img, f"Last command: {self.last_command}",
            (55, 145), cv2.FONT_HERSHEY_SIMPLEX,
            0.75, (0, 255, 0), 2
        )
        cv2.putText(
            img, f"Confidence: {self.last_confidence:.2f}",
            (55, 185), cv2.FONT_HERSHEY_SIMPLEX,
            0.62, (255, 255, 255), 1
        )
        cv2.putText(
            img, f"Raw prediction: {self.last_raw_prediction}",
            (55, 220), cv2.FONT_HERSHEY_SIMPLEX,
            0.58, (200, 200, 200), 1
        )

        cv2.putText(
            img, "VOICE COMMANDS AND ACTIONS",
            (350, 275), cv2.FONT_HERSHEY_SIMPLEX,
            0.8, (255, 220, 120), 2
        )

        left_commands = [
            ("UP", "Move cursor upward"),
            ("DOWN", "Move cursor downward"),
            ("LEFT", "Move cursor left"),
            ("RIGHT", "Move cursor right"),
            ("ZERO", "Left mouse click"),
            ("ONE", "Right mouse click"),
            ("TWO", "Double mouse click"),
            ("THREE", "Increase system volume"),
            ("FOUR", "Decrease system volume"),
        ]

        right_commands = [
            ("FIVE", "Open Windows Task View (Win + Tab)"),
            ("SIX", "Take a screenshot"),
            ("SEVEN", "Scroll upward"),
            ("EIGHT", "Scroll downward"),
            ("NINE", "Open Windows Search (Win + S)"),
            ("GO", "Press Enter"),
            ("STOP", "Play / pause media"),
            ("OFF", "Stop voice mode; keep app open"),
        ]

        y_start = 330
        row_gap = 38

        for index, (command, purpose) in enumerate(left_commands):
            y = y_start + index * row_gap
            cv2.putText(
                img, command,
                (65, y), cv2.FONT_HERSHEY_SIMPLEX,
                0.62, (0, 220, 255), 2
            )
            cv2.putText(
                img, purpose,
                (220, y), cv2.FONT_HERSHEY_SIMPLEX,
                0.53, (245, 245, 245), 1
            )

        for index, (command, purpose) in enumerate(right_commands):
            y = y_start + index * row_gap
            cv2.putText(
                img, command,
                (600, y), cv2.FONT_HERSHEY_SIMPLEX,
                0.62, (0, 220, 255), 2
            )
            cv2.putText(
                img, purpose,
                (755, y), cv2.FONT_HERSHEY_SIMPLEX,
                0.48, (245, 245, 245), 1
            )

        cv2.putText(
            img,
            "UNKNOWN / SILENCE: No action is executed",
            (330, 680), cv2.FONT_HERSHEY_SIMPLEX,
            0.58, (0, 200, 255), 1
        )

        self.display_image(img)

    # --------------------------------------------------------
    # DISPLAY CAMERA IMAGE
    # --------------------------------------------------------

    def display_image(self, img):

        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        h, w, ch = rgb.shape
        bytes_per_line = ch * w

        qimage = QImage(
            rgb.data,
            w,
            h,
            bytes_per_line,
            QImage.Format_RGB888
        )

        pixmap = QPixmap.fromImage(qimage)
        fitted_pixmap = pixmap.scaled(
            self.video_label.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )
        self.video_label.setPixmap(fitted_pixmap)

    # --------------------------------------------------------
    # RECORD AND CLASSIFY VOICE
    # --------------------------------------------------------

    def record_and_process_voice(self):

        if not self.running or self.app_mode != 1:
            return

        try:

            self.is_recording = True
            self.status_label.setText(
                "Status: Recording 2-second audio..."
            )
            self.display_voice_feedback()
            QApplication.processEvents()

            audio = sd.rec(
                int(AUDIO_DURATION * SAMPLE_RATE),
                samplerate=SAMPLE_RATE,
                channels=AUDIO_CHANNELS,
                dtype="float32"
            )

            sd.wait()

            audio = np.squeeze(audio)

            self.is_recording = False

            if not self.running or self.app_mode != 1:
                return

            self.status_label.setText(
                "Status: Classifying audio..."
            )

            result = self.voice_predictor.predict_audio(
                audio,
                sample_rate=SAMPLE_RATE
            )

            command = result["label"]

            self.last_command = command
            self.last_confidence = result["command_confidence"]
            self.last_raw_prediction = result["raw_prediction"]

            self.process_voice_safety_gate(
                command,
                result
            )

            self.display_voice_feedback()

            if self.running and self.app_mode == 1:
                QTimer.singleShot(
                    150,
                    self.record_and_process_voice
                )

        except Exception as exc:

            self.is_recording = False

            self.status_label.setText(
                f"Status: Voice Error - {str(exc)}"
            )

            print("Voice processing error:", exc)

            if self.running and self.app_mode == 1:
                QTimer.singleShot(
                    1500,
                    self.record_and_process_voice
                )

    # --------------------------------------------------------
    # VOICE COMMAND ACCEPTANCE GATE
    # --------------------------------------------------------

    def process_voice_safety_gate(self, command, result):

        # Unsupported/uncertain speech and silence never execute actions.
        if command in NON_ACTION_CLASSES:

            self.previous_voice_prediction = None
            self.voice_prediction_streak = 0

            if VOICE_REARM_ON_UNKNOWN_OR_SILENCE:
                self.voice_armed = True

            self.status_label.setText(
                f"Status: {command.upper()} - No action"
            )
            return

        if command not in TARGET_COMMANDS:
            return

        if command == self.previous_voice_prediction:
            self.voice_prediction_streak += 1
        else:
            self.previous_voice_prediction = command
            self.voice_prediction_streak = 1

        if self.voice_prediction_streak < REQUIRE_CONSECUTIVE_PREDICTIONS:

            self.status_label.setText(
                f"Status: Confirming '{command}' "
                f"({self.voice_prediction_streak}/"
                f"{REQUIRE_CONSECUTIVE_PREDICTIONS})"
            )
            return

        # Do not disarm after execution: consecutive repeated commands
        # are allowed on subsequent recording cycles.
        self.execute_voice_command(command)

    # --------------------------------------------------------
    # EXECUTE VOICE COMMAND
    # --------------------------------------------------------

    def execute_voice_command(self, command):

        confidence_percent = round(
            self.last_confidence * 100, 1
        )

        self.status_label.setText(
            f"Status: Executed '{command}' "
            f"(Command confidence: {confidence_percent}%)"
        )

        print(
            f"VOICE COMMAND: {command} | "
            f"Confidence: {confidence_percent}%"
        )

        try:

            if command == "up":
                pyautogui.move(0, -100)

            elif command == "down":
                pyautogui.move(0, 100)

            elif command == "left":
                pyautogui.move(-100, 0)

            elif command == "right":
                pyautogui.move(100, 0)

            elif command == "stop":
                pyautogui.press("playpause")

            elif command == "zero":
                pyautogui.click()

            elif command == "one":
                pyautogui.rightClick()

            elif command == "two":
                pyautogui.doubleClick()

            elif command == "three":
                for _ in range(5):
                    pyautogui.press("volumeup")

            elif command == "four":
                for _ in range(5):
                    pyautogui.press("volumedown")

            elif command == "five":
                pyautogui.hotkey("win", "tab")

            elif command == "six":
                pyautogui.hotkey("win", "printscreen")

            elif command == "seven":
                pyautogui.scroll(100)

            elif command == "eight":
                pyautogui.scroll(-100)

            elif command == "go":
                pyautogui.press("enter")

            elif command == "nine":
                pyautogui.hotkey("win", "s")

            elif command == "off":
                self.stop_all()

        except pyautogui.FailSafeException:

            self.stop_all()

            QMessageBox.warning(
                self,
                "PyAutoGUI Safety",
                "PyAutoGUI fail-safe activated. "
                "The application has stopped."
            )

        except Exception as exc:

            self.status_label.setText(
                f"Action Error: {str(exc)}"
            )
            print("Command execution error:", exc)

    # --------------------------------------------------------
    # CLEANUP
    # --------------------------------------------------------

    def closeEvent(self, event):

        self.stop_all()

        try:
            self.detector.close()
        except Exception:
            pass

        event.accept()


# ============================================================
# APPLICATION ENTRY POINT
# ============================================================

if __name__ == "__main__":

    app = QApplication(sys.argv)

    window = CombinedApp()

    sys.exit(app.exec_())
