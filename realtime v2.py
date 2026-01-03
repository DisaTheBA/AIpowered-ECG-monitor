import serial
import numpy as np
import scipy.signal
from collections import deque
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# --- Enhanced Configuration for Demo ---
SERIAL_PORT = 'COM6'
BAUD_RATE = 115200
SAMPLING_RATE_ARDUINO = 250
WINDOW_SECONDS = 10
BUFFER_SAMPLES = SAMPLING_RATE_ARDUINO * WINDOW_SECONDS

# Model configuration
SAMPLING_RATE_MODEL = 360
MODEL_SAMPLES = SAMPLING_RATE_MODEL * WINDOW_SECONDS
MODEL_PATH = 'ecg_bpm_classifier.h5'

# Demo-specific settings
PLOT_SECONDS = 8  # Show more seconds for better visualization
PLOT_SAMPLES = SAMPLING_RATE_ARDUINO * PLOT_SECONDS
PREDICTION_INTERVAL = 2.0  # Predict every 2 seconds for demo clarity

# Global variables
data_buffer = deque(maxlen=BUFFER_SAMPLES)
plot_buffer = deque(maxlen=PLOT_SAMPLES)
current_prediction = "Initializing..."
confidence = 0.0
class_names = ['Bradycardia (<60 BPM)', 'Normal (60-100 BPM)', 'Tachycardia (>100 BPM)']
class_colors = ['blue', 'green', 'red']  # Colors for each class

# --- Load Model ---
print("🫀 Loading ECG Classification Model...")
model = tf.keras.models.load_model(MODEL_PATH)
print("✅ Model loaded successfully!")

# --- Signal Processing Functions ---
def apply_bandpass_filter(signal, fs=SAMPLING_RATE_MODEL, lowcut=0.5, highcut=40.0):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = scipy.signal.butter(1, [low, high], btype='band')
    return scipy.signal.filtfilt(b, a, signal)

def normalize_signal(signal):
    mean = np.mean(signal)
    std = np.std(signal)
    return (signal - mean) / std if std != 0 else signal - mean

def predict_from_buffer():
    global current_prediction, confidence

    if len(data_buffer) < BUFFER_SAMPLES:
        return

    # Process signal
    signal = np.array(data_buffer)
    resampled_signal = scipy.signal.resample(signal, MODEL_SAMPLES)
    filtered_signal = apply_bandpass_filter(resampled_signal)
    normalized_signal = normalize_signal(filtered_signal)
    model_input = normalized_signal.reshape(1, MODEL_SAMPLES, 1)

    # Predict
    try:
        pred_probs = model.predict(model_input, verbose=0)
        pred_class = np.argmax(pred_probs)
        confidence = np.max(pred_probs) * 100

        current_prediction = class_names[pred_class]
        print(f"🎯 Prediction: {class_names[pred_class]} | Confidence: {confidence:.1f}%")

    except Exception as e:
        print(f"❌ Prediction error: {e}")

# --- Enhanced Plot Setup ---
plt.rcParams['font.size'] = 12
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), height_ratios=[3, 1])
fig.suptitle('Real-time ECG Classification System', fontsize=16, fontweight='bold')

# ECG Plot
ax1.set_ylim(0, 1024)
ax1.set_xlim(0, PLOT_SAMPLES)
ax1.set_ylabel('ECG Amplitude (ADC Value)')
ax1.set_xlabel('Time (samples)')
ax1.grid(True, alpha=0.3)
ecg_line, = ax1.plot([], [], 'b-', linewidth=1, label='Live ECG Signal')

# Prediction display
ax2.set_xlim(0, 1)
ax2.set_ylim(0, 1)
ax2.axis('off')
prediction_text = ax2.text(0.5, 0.7, 'Initializing...',
                          ha='center', va='center', fontsize=16, fontweight='bold')
confidence_text = ax2.text(0.5, 0.3, 'Confidence: --%',
                          ha='center', va='center', fontsize=14)

plt.tight_layout()

# --- Serial Setup ---
print(f"🔌 Connecting to Arduino on {SERIAL_PORT}...")
try:
    ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=0.1)
    print("✅ Serial connection established!")
except Exception as e:
    print(f"❌ Serial error: {e}")
    print("💡 TIP: Make sure Arduino IDE is closed and port is correct")
    exit()

# Initialize buffers
plot_buffer.extend([512] * PLOT_SAMPLES)

# Demo timing
update_counter = 0
PREDICT_EVERY = int(PREDICTION_INTERVAL * 50)  # 50 FPS

# --- Enhanced Animation Loop ---
def update(frame):
    global update_counter

    try:
        # Read serial data
        while ser.in_waiting > 0:
            line_str = ser.readline().decode('utf-8').strip()
            if line_str:
                try:
                    value = int(line_str)
                    if 0 <= value <= 1023:
                        data_buffer.append(value)
                        plot_buffer.append(value)
                except ValueError:
                    pass

        # Update ECG plot
        ecg_line.set_data(range(len(plot_buffer)), plot_buffer)

        # Update prediction display with color coding
        pred_class = 1  # Default to normal
        if "Bradycardia" in current_prediction:
            pred_class = 0
        elif "Tachycardia" in current_prediction:
            pred_class = 2

        prediction_text.set_text(f'Diagnosis: {current_prediction}')
        prediction_text.set_color(class_colors[pred_class])
        confidence_text.set_text(f'Confidence: {confidence:.1f}%')

        # Periodic prediction
        update_counter += 1
        if update_counter >= PREDICT_EVERY:
            update_counter = 0
            predict_from_buffer()

        return ecg_line, prediction_text, confidence_text

    except Exception as e:
        print(f"⚠️ Update error: {e}")
        return ecg_line, prediction_text, confidence_text

print("\nStarting real-time ECG monitoring...")
print("The system will classify heart rhythm every 2 seconds")
print("Close the plot window to stop the application\n")

# Start animation
ani = FuncAnimation(fig, update, blit=True, interval=20, cache_frame_data=False)
plt.show()

# Cleanup
ser.close()
print(" Application stopped. Serial connection closed.")