AI-Powered ECG Monitoring System

Project Description

This repository contains the source code and documentation for an edge-integrated Electrocardiogram (ECG) monitoring system. Developed as part of the Industrial Computing Design Project (FISA) at the Cape Peninsula University of Technology, the system is designed to provide real-time cardiac health tracking using a Raspberry Pi 5.

The primary goal of the project is to classify heart rates into three distinct categories—Abnormally Low, Normal, and Abnormally High—using a custom 1D Convolutional Neural Network (CNN). By processing signals locally at the edge, the system offers an affordable solution for detecting conditions such as bradycardia and tachycardia without the need for high-latency cloud computing.

Hardware Specifications

The hardware architecture focuses on precision data acquisition and local processing.

Component

Function

Raspberry Pi 5

Central processing unit for edge inference and system logic

AD8232

Analog front-end for ECG signal conditioning and acquisition

ADS1115

16-bit Analog-to-Digital Converter (ADC) for I2C communication

16x2 I2C LCD

Visual interface for displaying real-time BPM and health status

LED Indicators

Physical alerts (Green: Normal status; Red: Heart rate abnormality)

Pin Configuration

The system utilizes the I2C protocol for communication between the ADC, the LCD, and the Raspberry Pi.

SDA: GPIO 2 (Pin 3)

SCL: GPIO 3 (Pin 5)

Green LED: GPIO 17 (Pin 11)

Red LED: GPIO 27 (Pin 13)

Machine Learning Methodology

To ensure academic rigor and model accuracy, the system follows a structured machine learning pipeline.

1. Data Acquisition and Automatic Labeling

The model is trained on the MIT-BIH Arrhythmia Database. To handle large-scale datasets efficiently, a rule-based algorithm was implemented for automatic ground-truth generation:

Signal Segmentation: Raw signals are divided into 10-second windows.

Peak Detection: R-peaks are identified using scipy.signal.find_peaks.

Label Assignment:

0 (Low): < 60 BPM

1 (Normal): 60 - 100 BPM

2 (High): > 100 BPM

2. CNN Architecture

A lightweight 1D-CNN was designed to maintain high performance on the Raspberry Pi's hardware:

Two Conv1D layers with 32 and 64 filters respectively.

Batch Normalization and MaxPooling for feature stabilization and reduction.

Dropout layers (0.3 - 0.4) to prevent overfitting during training.

A Softmax output layer for 3-class probability distribution.

Installation and Deployment

Prerequisites

Raspberry Pi OS (64-bit recommended)

Python 3.9+

I2C interface enabled via raspi-config

Software Setup

Clone the repository:

git clone [https://github.com/DisaTheBA/AIpowered-ECG-monitor)
cd ai-ecg-monitor


Install the required Python libraries:

pip install tensorflow wfdb scipy pandas matplotlib seaborn scikit-learn


Execute the training script (optional):

python ecg_bpm_classifier.py


Author

Mandisa Shandu
