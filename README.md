AI-Powered ECG Monitoring System
Project Overview
This project describes the design and implementation of an edge-integrated Electrocardiogram (ECG) monitoring system. The system provides real-time detection of cardiac abnormalities, specifically focusing on Bradycardia and Tachycardia. By utilizing a 1D Convolutional Neural Network (1D-CNN), the system classifies heart rhythms into three categories based on a 10-second sliding window of cardiac data.

System Architecture
The system is organized into three specialized layers to ensure modularity and efficient data flow.

Layer 1: Embedded Acquisition Layer
The hardware interface responsible for capturing and digitizing the cardiac signal.

AD8232 ECG Sensor: A single-lead sensor using three electrode pads (RA, LA, RL) for differential input. It includes integrated signal conditioning and a Right Leg Drive (RLD) circuit to minimize common-mode noise and power line interference.

Arduino Uno Microcontroller: Serves as the data acquisition unit. It samples the analog signal at a strict rate of 250 Hz using hardware timer interrupts to ensure timing precision and avoid jitter.

Output: A digital stream of 10-bit ADC values transmitted via USB Serial.

Layer 2: Communication and Control Layer
The bridge between the embedded hardware and the computational host.

USB Serial Connection: Provides the physical link for data transmission.

Python Serial Library (pyserial): Manages asynchronous, non-blocking communication on the host side.

Bidirectional Feedback: This layer handles the transmission of raw ECG data to the laptop and sends encoded diagnostic commands (e.g., "Brady," "Normal," "Tachy") back to the Arduino to trigger physical alerts (LED/LCD).

Layer 3: Host Processing and AI Inference Layer
The high-performance computational core residing on a laptop.

Digital Signal Processing (DSP) Pipeline:

Bandpass Filtering: A third-order Butterworth filter (0.5 Hz - 40.0 Hz) removes baseline wander and high-frequency noise.

Resampling: Signals are resampled from 250 Hz (2500 samples) to 360 Hz (3600 samples) using linear interpolation to match the AI model's input requirements.

Normalization: Applied via Z-score standardization.

AI Inference Engine: A 1D-CNN trained using the MIT-BIH Arrhythmia Database. The model processes the 3600-sample segment to output probability distributions for the three cardiac classes.

User Interface: Real-time visualization of the ECG waveform via Matplotlib and status updates via LED and LCD indicators.

Methodology
1. Data Understanding and Preparation
The system utilizes the MIT-BIH Arrhythmia Database. Continuous signals were segmented into 10-second windows. Data augmentation and oversampling were employed to balance the classes and improve model generalization across underrepresented abnormalities.

2. Model Development
Architecture: 1D-CNN selected for its effectiveness in time-series pattern recognition.

Activation: Exponential Linear Unit (ELU) functions.

Training: Developed using the PyTorch framework in a Google Colab environment and exported as an ecg_bpm_classifier.h5 file.

3. Implementation and Integration
Hardware Connection: * AD8232 OUTPUT to Arduino A0.

Electrodes placed at RA (Right chest), LA (Left chest), and RL (Right abdomen/RLD).

Software Environment: Developed using Arduino IDE (C++ firmware) and Visual Studio Code (Python processing scripts).

Performance and Evaluation
Accuracy: The 1D-CNN demonstrated an internal validation accuracy exceeding 90%.

Latency: The end-to-end system latency, measured from buffer triggering to prediction, averaged 2.5 seconds (±0.3s), meeting the real-time requirement of <3.0 seconds.

Robustness: The DSP pipeline effectively suppresses artifacts, and a Signal Quality Check is implemented to prevent false predictions during periods of extreme noise or lead-off.

Installation and Usage
Prerequisites
Arduino IDE for firmware deployment.

Python 3.x with the following libraries:

tensorflow / keras

pyserial

numpy

scipy

matplotlib

Deployment
Upload the provided Arduino C++ sketch to the Arduino Uno.

Ensure the AD8232 electrodes are correctly placed on the user.

Connect the Arduino to the laptop via USB.

Execute the Python application in the VS Code environment to begin real-time monitoring.

Author
Mandisa Shandu
Student Number: 230522076
