# Virtual Mouse V2 – Hand Gesture and Voice Controlled Mouse

A Python-based virtual mouse application that allows users to control computer operations using hand gestures and voice commands. The application integrates computer vision, machine learning, speech recognition and desktop automation to provide an alternative method of human-computer interaction.

**GitHub Repository:** [gesture-voice-control-mouse](https://github.com/KAshyaP323/gesture-voice-control-mouse)

---

## Features

### 1. Hand Gesture Control
- Real-time hand detection and tracking using MediaPipe.
- Cursor movement based on hand and finger positions.
- Gesture-based mouse operations.
- Supports predefined gestures for mouse and keyboard interactions.

### 2. Voice Command Control
- Speech recognition using a custom-trained Convolutional Neural Network (CNN).
- Audio preprocessing and MFCC feature extraction.
- Supports 17 predefined voice commands.
- Includes Unknown and Silence classes to handle unrecognized speech and background audio.
- Continuous voice command recognition.
- Allows repeated execution of the same command without requiring silence between recordings.
- Displays predicted commands and confidence information.

### 3. Graphical User Interface
- Developed using PyQt5.
- Separate modes for hand gesture and voice control.
- Live camera feed display.
- Voice recognition dashboard.
- Start and stop controls for operating modes.

---

## Technologies Used

| Technology | Purpose |
|---|---|
| Python 3.12 | Core programming language |
| OpenCV | Camera access and image processing |
| MediaPipe | Hand detection and tracking |
| TensorFlow / Keras | CNN-based speech recognition |
| NumPy | Numerical computations |
| SciPy | Signal processing support |
| SoundDevice | Audio recording |
| PyAutoGUI | Mouse and keyboard automation |
| PyQt5 | Graphical user interface |

---

## Voice Command Mapping

The voice recognition system supports the following commands:

| Voice Command | Function |
|---|---|
| UP | Move cursor upward |
| DOWN | Move cursor downward |
| LEFT | Move cursor left |
| RIGHT | Move cursor right |
| ZERO | Left mouse click |
| ONE | Right mouse click |
| TWO | Double mouse click |
| THREE | Increase system volume |
| FOUR | Decrease system volume |
| FIVE | Open Windows Task View |
| SIX | Take screenshot |
| SEVEN | Scroll upward |
| EIGHT | Scroll downward |
| NINE | Open Windows Search |
| GO | Press Enter |
| STOP | Play/Pause media |
| OFF | Stop voice control mode |
| UNKNOWN | No action |
| SILENCE | No action |

---

## Project Structure

```text
gesture-voice-control-mouse/
│
├── .gitignore
├── README.md
├── requirements.txt
│
├── gesture_voice_mouse.py
├── voice_command_inference.py
│
├── voice_command_cnn.keras
├── class_labels.json
└── preprocessing_config.json
```

### Important Files

| File | Description |
|---|---|
| `gesture_voice_mouse.py` | Main application containing the GUI and mouse control functionality. |
| `voice_command_inference.py` | Handles audio preprocessing and CNN model inference. |
| `voice_command_cnn.keras` | Trained CNN voice recognition model. |
| `class_labels.json` | Contains the model's output class labels. |
| `preprocessing_config.json` | Stores audio preprocessing configuration. |
| `requirements.txt` | Lists the required Python dependencies. |
| `.gitignore` | Excludes virtual environments, cache files and unnecessary local files. |

---

## Installation and Setup

### Prerequisites

- Python 3.12
- Webcam or USB camera.
- Microphone.
- Windows operating system recommended for the predefined Windows-specific keyboard shortcuts.

### Step 1: Clone the Repository

Open a terminal and execute:

```bash
git clone https://github.com/KAshyaP323/gesture-voice-control-mouse.git
```

Navigate to the project directory:

```bash
cd gesture-voice-control-mouse
```

### Step 2: Create a Virtual Environment

Create a Python virtual environment:

```bash
python -m venv .venv
```

Activate the environment.

**Windows:**

```bash
.venv\Scripts\activate
```

**Linux:**

```bash
source .venv/bin/activate
```

### Step 3: Install Dependencies

Upgrade pip:

```bash
python -m pip install --upgrade pip
```

Install the required packages:

```bash
pip install -r requirements.txt
```

### Step 4: Verify Model Files

Ensure the following model files are available in the project root:

```text
voice_command_cnn.keras
class_labels.json
preprocessing_config.json
```

The inference module should reference these files:

```python
MODEL_PATH = APP_DIR / "voice_command_cnn.keras"
CONFIG_PATH = APP_DIR / "preprocessing_config.json"
LABELS_PATH = APP_DIR / "class_labels.json"
```

All three files are required for voice command recognition.

### Step 5: Run the Application

Start the application:

```bash
python gesture_voice_mouse.py
```

---

## How to Use

1. Launch the application using the Python execution command.
2. Select the desired control mode from the graphical interface.
3. For hand gesture control, start the camera and perform the supported gestures.
4. For voice control, activate the voice recording mode.
5. Speak one of the supported voice commands.
6. The application processes the audio and displays the predicted command and confidence.
7. Continue speaking commands to perform mouse and keyboard operations.
8. Press Stop or say `OFF` to stop voice control without closing the application.

---

## Machine Learning Model

The voice recognition component uses a custom-trained Convolutional Neural Network (CNN) based on the **Google Speech Commands V2 dataset**.

### Model Configuration

| Parameter | Configuration |
|---|---|
| Model architecture | Convolutional Neural Network (CNN) |
| Dataset | Google Speech Commands V2 |
| Audio sampling rate | 16 kHz |
| Feature extraction | Mel-Frequency Cepstral Coefficients (MFCC) |
| Number of MFCC coefficients | 40 |
| Inference input duration | 1 second |
| Classification | 17 voice commands + Unknown + Silence |

### Audio Recognition Pipeline

The voice recognition process follows these stages:

1. Audio recording through the microphone.
2. Audio preprocessing and normalization.
3. MFCC feature extraction.
4. Feature input to the trained CNN model.
5. Voice command classification.
6. Confidence evaluation and command execution.

The model checkpoint with the lowest validation loss is selected for deployment.

---

## Troubleshooting

| Issue | Suggested Solution |
|---|---|
| Camera not detected | Check camera permissions and verify the camera index. |
| Microphone not detected | Check microphone permissions and input device settings. |
| Voice commands not recognized | Check pronunciation, microphone quality and background noise. |
| Model loading error | Verify that the model and JSON configuration files are present. |
| Low prediction confidence | Check audio quality and the model confidence threshold. |
| Missing Python packages | Activate the virtual environment and reinstall dependencies. |
| Cursor movement issues | Check camera tracking and mouse control settings. |

---

## Future Enhancements

- Improve speech recognition accuracy across different accents.
- Introduce additional customizable voice commands.
- Improve real-time prediction performance.
- Add support for customizable hand gestures.
- Extend compatibility across operating systems.
- Optimize the CNN model for lightweight deployment.

---

## License

No open-source license has currently been specified for this project.

---

**Virtual Mouse V2**

*Enhancing human-computer interaction through computer vision and voice recognition.*