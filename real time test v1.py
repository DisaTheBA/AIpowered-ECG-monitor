import serial
import numpy as np
import scipy.signal
from collections import deque
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# --- Configuration ---
SERIAL_PORT = 'COM3'  # !! IMPORTANT: Change this to your Arduino's serial port
BAUD_RATE = 115200
SAMPLING_RATE_ARDUINO = 250  # Must match the Arduino sketch
WINDOW_SECONDS = 10
BUFFER_SAMPLES = SAMPLING_RATE_ARDUINO * WINDOW_SECONDS # 2500 samples

# Model's expected input
SAMPLING_RATE_MODEL = 360
MODEL_SAMPLES = SAMPLING_RATE_MODEL * WINDOW_SECONDS # 3600 samples
MODEL_PATH = 'ecg_bpm_classifier.h5'
INPUT_SHAPE = (MODEL_SAMPLES, 1)

# Plotting
PLOT_SECONDS = 5 # How many seconds to display on the plot
PLOT_SAMPLES = SAMPLING_RATE_ARDUINO * PLOT_SECONDS # 1250 samples

# Global variables
data_buffer = deque(maxlen=BUFFER_SAMPLES)
plot_buffer = deque(maxlen=PLOT_SAMPLES)
current_prediction = "Initializing..."
class_names = ['Bradycardia (<60)', 'Normal (60-100)', 'Tachycardia (>100)']

# --- 1. Load Model ---
print(f"Loading model from {MODEL_PATH}...")
model = tf.keras.models.load_model(MODEL_PATH)
model.summary()

# --- 2. Signal Processing Functions (Must be identical to training) ---
def apply_bandpass_filter(signal, fs=SAMPLING_RATE_MODEL, lowcut=0.5, highcut=40.0):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = scipy.signal.butter(1, [low, high], btype='band')
    filtered_signal = scipy.signal.filtfilt(b, a, signal)
    return filtered_signal

def normalize_signal(signal):
    mean = np.mean(signal)
    std = np.std(signal)
    if std == 0:
        return signal - mean
    return (signal - mean) / std

# --- 3. Real-Time Prediction Function ---
def predict_from_buffer():
    global current_prediction

    # Only predict if the buffer is full
    if len(data_buffer) < BUFFER_SAMPLES:
        return

    # 1. Get snapshot of data
    signal = np.array(data_buffer)

    # 2. Resample from 250Hz -> 360Hz
    resampled_signal = scipy.signal.resample(signal, MODEL_SAMPLES)

    # 3. Apply Preprocessing
    filtered_signal = apply_bandpass_filter(resampled_signal, fs=SAMPLING_RATE_MODEL)
    normalized_signal = normalize_signal(filtered_signal)

    # 4. Reshape for model
    model_input = normalized_signal.reshape(1, MODEL_SAMPLES, 1)

    # 5. Predict
    try:
        pred_probs = model.predict(model_input, verbose=0)
        pred_class = np.argmax(pred_probs)
        current_prediction = f"Prediction: {class_names[pred_class]}"
        print(f"Prediction: {class_names[pred_class]} | Probs: {pred_probs}")
    except Exception as e:
        print(f"Prediction error: {e}")

# --- 4. Serial and Plotting Setup ---
print(f"Attempting to connect to {SERIAL_PORT} at {BAUD_RATE}...")
try:
    ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=0.1)
    print("Serial connection successful.")
except Exception as e:
    print(f"Error: Could not open serial port {SERIAL_PORT}.")
    print("Please check your Arduino connection and port name.")
    exit()

# Initialize plot
fig, ax = plt.subplots(figsize=(10, 6))
ax.set_ylim(0, 1024) # Arduino ADC is 0-1023
ax.set_xlim(0, PLOT_SAMPLES)
line, = ax.plot([], [])
ax.set_xlabel('Samples')
ax.set_ylabel('ADC Value')
title = ax.set_title('Live ECG Signal - Initializing...')
plot_buffer.extend([0] * PLOT_SAMPLES) # Fill plot buffer with zeros

# Counter to predict every N updates (e.g., every 1 second)
update_counter = 0
PREDICT_EVERY_N_UPDATES = 25 # Predict every 25 frames (approx 1 second)

# --- 5. Animation Loop ---
def update(frame):
    global update_counter
    try:
        # Read all available lines from serial
        while ser.in_waiting > 0:
            line_str = ser.readline().decode('utf-8').strip()
            if line_str:
                try:
                    value = int(line_str)
                    if 0 <= value <= 1023:
                        data_buffer.append(value)
                        plot_buffer.append(value)
                except ValueError:
                    pass # Ignore corrupted lines

        # Update plot data
        line.set_data(range(len(plot_buffer)), plot_buffer)

        # Update title with prediction
        title.set_text(f'Live ECG Signal | {current_prediction}')

        # Run prediction periodically
        update_counter += 1
        if update_counter >= PREDICT_EVERY_N_UPDATES:
            update_counter = 0
            predict_from_buffer()

        return line, title

    except Exception as e:
        print(f"Error in update loop: {e}")
        return line, title

# Start the animation
ani = FuncAnimation(fig, update, blit=True, interval=20, cache_frame_data=False)
plt.show()

# Clean up
ser.close()
print("Serial connection closed.")