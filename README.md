# TinyML ESP32 Early Exit Inference Project

This project demonstrates an end-to-end TinyML workflow for deploying a machine learning model with an **early exit strategy** on an ESP32 microcontroller. The early exit architecture allows the system to save power and reduce latency by terminating inference early for simpler inputs.

## Complete Project Workflow and Introduction

This project is built around the concept of optimizing Neural Network inference on edge hardware (ESP32) utilizing an Early Exit architecture. Deep learning models often process all inputs through every layer of the network, which consumes a consistent, dense amount of power and time. However, many inputs are "easy" to classify or process and don't require the full depth of the network.

**How it Works:**
1. **Model Training & Compression:** The Python pipeline trains a neural network that has intermediate output layers (early exits). The model is then compressed and converted to the TensorFlow Lite for Microcontrollers (`.tflite`) format.
2. **Threshold Tuning:** A separate tuning script analyzes the confidence levels of the early exits. If the confidence of an intermediate exit exceeds a determined threshold, the model can halt inference early instead of computing the remaining layers.
3. **Model Conversion:** The `.tflite` models are converted into standard C arrays (`.h` files) so they can be compiled directly into the ESP32 firmware.
4. **Hardware Deployment:** On the ESP32 side, the firmware handles taking readings from sensors, preprocessing the data, and passing it to the TensorFlow Lite Micro interpreter.
5. **Inference & Power Saving:** While executing the model, if the early exit criteria are met, the processing stops immediately. The firmware then captures power metrics and computational latency to demonstrate the power savings compared to running a full, unoptimized network block.

## Project Structure

- esp32/: Contains the C++ source code for the ESP32 deployment.
  - Includes TFlite Micro inference, sensor capture, and power monitoring.
  - Deployable via PlatformIO (esp32/platformio.ini).
- python/: Scripts for training the model and exporting it to TinyML formats.
  - 	rain_and_compress.py: Trains the early exit neural network and compresses it.
  - early_exit_tuner.py: Tunes the threshold for the early exit to balance accuracy and power.
  - convert_tflite_to_header.py: Converts the .tflite model into C headers.
- models_export/: Generated TensorFlow Lite models (.tflite) and C headers (.h).
- dataset/: Datasets for training.
- docs/: Project documentation, diagrams, and logs.

## Getting Started

### 1. Model Training & Export (Python)
`ash
pip install -r python/requirements.txt
python python/train_and_compress.py
python python/early_exit_tuner.py
python python/convert_tflite_to_header.py
`

### 2. ESP32 Deployment (PlatformIO)
1. Navigate to the esp32/ directory or open it in VS Code with PlatformIO.
2. Build and upload to your ESP32.
3. Observe serial monitor for model inference and power savings via early exit.
