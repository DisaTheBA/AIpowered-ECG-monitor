import serial
import numpy as np
import scipy.signal
from collections import deque
import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import time
import os
import traceback

#     Display backend    
matplotlib.use('TkAgg')
plt.rcParams['path.simplify'] = True
plt.rcParams['path.simplify_threshold'] = 1.0

#     Configuration    
SERIAL_PORT = 'COM6'   # ← verify your Arduino port
BAUD_RATE = 115200
SAMPLING_RATE_ARDUINO = 250
WINDOW_SECONDS = 10
BUFFER_SAMPLES = SAMPLING_RATE_ARDUINO * WINDOW_SECONDS

#     Model configuration    
SAMPLING_RATE_MODEL = 360
MODEL_SAMPLES = SAMPLING_RATE_MODEL * WINDOW_SECONDS
MODEL_PATH = 'ecg_bpm_classifier.h5'   # ← ensure this file is in your project folder

#     Plot & Prediction settings    
PLOT_SECONDS = 8
PLOT_SAMPLES = SAMPLING_RATE_ARDUINO * PLOT_SECONDS
PREDICTION_INTERVAL = 3.0   # Predict every 3 seconds

ADC_CENTER = 512

#     Globals    
data_buffer = deque(maxlen=BUFFER_SAMPLES)
plot_buffer = deque(maxlen=PLOT_SAMPLES)
raw_buffer = deque(maxlen=PLOT_SAMPLES)
current_prediction = "Warming up..."
confidence = 0.0
bpm_estimate = 0.0
# FIXED: Range separator for Normal BPM
class_names = ['Bradycardia (<60 BPM)', 'Normal (60-100 BPM)', 'Tachycardia (>100 BPM)']
class_colors = ['blue', 'green', 'red']

recent_preds = deque(maxlen=5)
recent_confidences = deque(maxlen=5)
recent_bpms = deque(maxlen=5)

signal_stats = {'mean': 0, 'std': 0, 'min': 0, 'max': 0, 'range': 0}
ser = None

#     Model loading    
print("Loading ECG Classification Model...")
if not os.path.exists(MODEL_PATH):
    print("Model file not found. Please check path and try again.")
    exit()

try:
    # Set run-time options to prevent potential issues with multi-threading on some systems
    model = tf.keras.models.load_model(MODEL_PATH)
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

#     Signal processing    
def apply_bandpass_filter(signal, fs=SAMPLING_RATE_MODEL, lowcut=0.5, highcut=40.0):
    nyquist = 0.5 * fs
    low, high = lowcut / nyquist, highcut / nyquist
    b, a = scipy.signal.butter(3, [low, high], btype='band')
    try:
        return scipy.signal.filtfilt(b, a, signal)
    except Exception as e:
        print(f"Filter error: {e}")
        return signal

def normalize_signal(signal):
    p5, p95 = np.percentile(signal, [5, 95])
    clipped = np.clip(signal, p5, p95)
    mean, std = np.mean(clipped), np.std(clipped)
    if std < 0.01:
        # FIXED: Added subtraction operator (-)
        return signal - mean
    # FIXED: Added subtraction operator (-)
    normalized = (clipped - mean) / std
    # FIXED: Added negative sign to -5
    return np.clip(normalized, -5, 5)

def estimate_bpm_from_signal(signal, fs=SAMPLING_RATE_MODEL):
    try:
        mean, std = np.mean(signal), np.std(signal)
        # Using a conservative peak detection threshold
        peaks, _ = scipy.signal.find_peaks(signal, height=mean + 0.3 * std,
                                             distance=int(fs * 0.3), prominence=0.3 * std)
        if len(peaks) < 2:
            return None
        rr_intervals = np.diff(peaks) / fs
        # FIXED: Added subtraction operator (-)
        valid_rr = rr_intervals[np.abs(rr_intervals - np.mean(rr_intervals)) < 2 * np.std(rr_intervals)]
        if len(valid_rr) == 0:
            return None
        bpm = 60.0 / np.mean(valid_rr)
        return bpm if 30 <= bpm <= 200 else None
    except Exception as e:
        print(f"BPM calculation error: {e}")
        return None

def check_signal_quality(signal):
    if len(signal) < 100:
        return False, "Insufficient data"
    if np.std(signal) < 0.01:
        # FIXED: Added space
        return False, "Signal too flat - check electrodes"
    # FIXED: Added subtraction operator (-)
    if np.max(signal) - np.min(signal) < 0.1:
        # FIXED: Added space
        return False, "Signal saturated - adjust electrodes"
    # FIXED: Replaced non-standard "1e 6" with standard "1e-6" for epsilon
    cv = np.std(signal) / (np.abs(np.mean(signal)) + 1e-6)
    if cv > 10:
        return False, "Excessive noise detected"
    return True, "OK"

#     Prediction function    
def predict_from_buffer():
    global current_prediction, confidence, bpm_estimate, signal_stats

    if len(data_buffer) < BUFFER_SAMPLES:
        current_prediction = f"Collecting data... ({len(data_buffer)}/{BUFFER_SAMPLES})"
        return

    try:
        raw_signal = np.array(data_buffer)
        signal_stats.update({
            'mean': np.mean(raw_signal),
            'std': np.std(raw_signal),
            'min': np.min(raw_signal),
            'max': np.max(raw_signal),
            # FIXED: Added subtraction operator (-)
            'range': np.max(raw_signal) - np.min(raw_signal)
        })

        is_good, quality_msg = check_signal_quality(raw_signal)
        if not is_good:
            current_prediction = quality_msg
            confidence = 0.0
            if ser and ser.is_open:
                # Using simpler instruction for the Arduino
                ser.write("WAIT\n".encode('utf-8'))
            return

        # Signal processing pipeline
        resampled = scipy.signal.resample(raw_signal, MODEL_SAMPLES)
        filtered = apply_bandpass_filter(resampled, fs=SAMPLING_RATE_MODEL)
        normalized = normalize_signal(filtered)

        # BPM Estimation
        bpm = estimate_bpm_from_signal(normalized, fs=SAMPLING_RATE_MODEL)
        if bpm:
            recent_bpms.append(bpm)
            bpm_estimate = np.mean(list(recent_bpms))

        # Model Prediction
        model_input = normalized.reshape(1, MODEL_SAMPLES, 1)
        pred_probs = model.predict(model_input, verbose=0)[0]
        pred_class = np.argmax(pred_probs)
        current_confidence = pred_probs[pred_class] * 100

        recent_preds.append(pred_class)
        recent_confidences.append(current_confidence)

        # Simple Consensus Smoothing
        if len(recent_preds) >= 3:
            # Weighting classes by confidence
            class_votes = {0: 0.0, 1: 0.0, 2: 0.0}
            for pred, conf in zip(recent_preds, recent_confidences):
                class_votes[pred] += conf
            smoothed_class = max(class_votes, key=class_votes.get)
            confidence = np.mean(list(recent_confidences))
        else:
            smoothed_class = pred_class
            confidence = current_confidence

        #     Send classification to Arduino    
        key_word = class_names[smoothed_class].split(' ')[0].upper() # Use uppercase for clarity
        if ser and ser.is_open:
            ser.write(f"{key_word}\n".encode('utf-8'))

        pred_text = class_names[smoothed_class]
        if bpm_estimate > 0:
            pred_text += f" | {bpm_estimate:.0f} BPM"
        current_prediction = pred_text

        print(f"\n{'='*60}")
        print(f"Time: {time.strftime('%H:%M:%S')}")
        print(f"Signal: Mean={signal_stats['mean']:.2f}, Std={signal_stats['std']:.2f}")
        if bpm_estimate > 0:
            print(f"Estimated BPM: {bpm_estimate:.1f}")
        print(f"Prediction: {class_names[smoothed_class]} | Confidence: {confidence:.1f}%")
        print(f"{'='*60}")

    except Exception as e:
        print(f"Prediction error: {e}")
        traceback.print_exc()

#     Plot setup    
plt.rcParams['font.size'] = 11
fig = plt.figure(figsize=(14, 9))
gs = fig.add_gridspec(3, 2, height_ratios=[2, 2, 1], hspace=0.3, wspace=0.3)

ax1 = fig.add_subplot(gs[0, :])
# Adjust Y limits for normalized data
ax1.set_ylim(-5, 5)
ax1.set_xlim(0, PLOT_SECONDS)
ax1.set_ylabel('Normalized ECG Amplitude')
ax1.set_title('Processed ECG Signal', fontweight='bold')
ax1.grid(True, alpha=0.3)
# Corrected line plot argument for 'b-' (blue solid line)
ecg_line, = ax1.plot([], [], 'b-', linewidth=1.2, label='Filtered ECG')
ax1.legend(loc='upper right')

ax2 = fig.add_subplot(gs[1, 0])
# Adjust Y limits for raw normalized (ADC - 512) / 512 data, range is approx [-1, 1]
ax2.set_ylim(-1.5, 1.5)
ax2.set_xlim(0, PLOT_SECONDS)
ax2.set_ylabel('Normalized Raw Amplitude')
ax2.set_title('Normalized Raw Arduino Signal', fontweight='bold')
ax2.grid(True, alpha=0.3)
# Corrected line plot argument for 'g-' (green solid line)
raw_line, = ax2.plot([], [], 'g-', linewidth=0.8, alpha=0.7)

ax3 = fig.add_subplot(gs[1, 1])
ax3.set_xlim(0, 1)
ax3.set_ylim(0, 1)
ax3.axis('off')
ax3.set_title('Signal Metrics', fontweight='bold')
quality_text = ax3.text(0.1, 0.7, 'Initializing...', fontsize=10, family='monospace', va='top')

ax4 = fig.add_subplot(gs[2, :])
ax4.set_xlim(0, 1)
ax4.set_ylim(0, 1)
ax4.axis('off')
prediction_text = ax4.text(0.5, 0.7, 'Initializing...',
                           ha='center', va='center', fontsize=18, fontweight='bold')
# FIXED: Added space before %
confidence_text = ax4.text(0.5, 0.3, 'Confidence: 0.0 %',
                           ha='center', va='center', fontsize=14)

fig.suptitle('Real Time ECG Classification System', fontsize=16, fontweight='bold')

#     Serial setup    
print(f"\nConnecting to ECG device on {SERIAL_PORT}...")
try:
    ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=0.1)
    time.sleep(2)
    ser.reset_input_buffer()
    print("Serial connection established!")
except Exception as e:
    print(f"Serial error: {e}")
    print("Close Arduino IDE and check COM port.")
    exit()

# Initialize buffers with zero data for initial plot
plot_buffer.extend([0.0] * PLOT_SAMPLES)
raw_buffer.extend([0.0] * PLOT_SAMPLES)
update_counter = 0
# Calculate how many frames to wait between predictions (interval / frame_rate)
PREDICT_EVERY = int(PREDICTION_INTERVAL * 50) # interval=20ms => 50 frames/sec

#     Update loop    
def update(frame):
    global update_counter, confidence
    try:
        lines_read = 0
        # Read up to 50 lines to keep up with the data stream (250 Hz)
        while ser.in_waiting > 0 and lines_read < 50:
            lines_read += 1
            try:
                line = ser.readline().decode('utf-8', errors='ignore').strip()
                value = float(line)
                
                # Basic bounds check for 10-bit ADC
                if 0 <= value <= 1023:
                    data_buffer.append(value)
                    # FIXED: Added subtraction operator (-) for normalization around 0
                    raw_norm = (value - ADC_CENTER) / ADC_CENTER 
                    raw_buffer.append(raw_norm)
                    # Use the raw_norm for plot_buffer since it's already normalized for display
                    plot_buffer.append(raw_norm) 
            except:
                # Ignore lines that cannot be converted to float
                continue

        time_sec = np.arange(len(plot_buffer)) / SAMPLING_RATE_ARDUINO
        # Use only the last PLOT_SAMPLES of the time array
        time_sec_plot = time_sec[-PLOT_SAMPLES:]
        
        # Redraw lines
        ecg_line.set_data(time_sec_plot, plot_buffer)
        raw_line.set_data(time_sec_plot, raw_buffer)

        # Update quality metrics text
        quality_text.set_text(
            f"Buffer: {len(data_buffer)}/{BUFFER_SAMPLES}\n"
            f"Mean: {signal_stats['mean']:.3f}\nStd: {signal_stats['std']:.3f}\n"
            f"Range: {signal_stats['range']:.3f}"
        )

        # Determine color based on current prediction
        pred_class = 1
        if "Bradycardia" in current_prediction: pred_class = 0
        elif "Tachycardia" in current_prediction: pred_class = 2

        # Update prediction text and confidence
        prediction_text.set_text(current_prediction)
        prediction_text.set_color(class_colors[pred_class])
        # FIXED: Ensure consistent formatting and handling of zero confidence
        confidence_text.set_text(f'Confidence: {confidence:.1f} %' if confidence > 0.1 else 'Confidence: -- %')

        # Trigger prediction periodically
        update_counter += 1
        if update_counter >= PREDICT_EVERY:
            update_counter = 0
            predict_from_buffer()

        # Update plot limits based on the current data view (optional, but helpful for drift)
        ax1.set_xlim(time_sec_plot[0], time_sec_plot[-1])
        ax2.set_xlim(time_sec_plot[0], time_sec_plot[-1])
        
        # Must return all artists that were modified for blit=True
        return ecg_line, raw_line, prediction_text, confidence_text, quality_text

    except Exception as e:
        print(f"Update loop error: {e}")
        # Return all artists even on error to prevent plot from freezing entirely
        return ecg_line, raw_line, prediction_text, confidence_text, quality_text

print("\n" + "="*60)
print("REAL TIME ECG MONITORING STARTED")
print("="*60)
print("1. Ensure electrodes are properly attached")
print("2. Remain still during measurement")
print("3. System classifies every 3 seconds")
print("4. Check your external device (Arduino/LCD) for real-time status")
print("="*60 + "\n")

# interval=20ms means 50 frames per second
ani = FuncAnimation(fig, update, blit=True, interval=20, cache_frame_data=False)
plt.show()

# Clean up
if ser and ser.is_open:
    ser.close()
    print("\nSerial connection closed.")
else:
    print("\nApplication stopped (Serial not connected).")
