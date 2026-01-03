import serial
import numpy as np
import scipy.signal
from collections import deque
import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import time

matplotlib.use('TkAgg')
plt.rcParams['path.simplify'] = True
plt.rcParams['path.simplify_threshold'] = 1.0

SERIAL_PORT = 'COM6'
BAUD_RATE = 115200
SAMPLING_RATE_ARDUINO = 250
WINDOW_SECONDS = 10
BUFFER_SAMPLES = SAMPLING_RATE_ARDUINO * WINDOW_SECONDS

SAMPLING_RATE_MODEL = 360
MODEL_SAMPLES = SAMPLING_RATE_MODEL * WINDOW_SECONDS
MODEL_PATH = 'ecg_bpm_classifier.h5'

PLOT_SECONDS = 8
PLOT_SAMPLES = SAMPLING_RATE_ARDUINO * PLOT_SECONDS
PREDICTION_INTERVAL = 3.0  # Increased for stability

ADC_MIN = 0
ADC_MAX = 1023
ADC_CENTER = 512

# Global variables
data_buffer = deque(maxlen=BUFFER_SAMPLES)
plot_buffer = deque(maxlen=PLOT_SAMPLES)
raw_buffer = deque(maxlen=PLOT_SAMPLES)  # NEW: Store raw values for debugging
current_prediction = "Warming up..."
confidence = 0.0
bpm_estimate = 0.0
class_names = ['Bradycardia (<60 BPM)', 'Normal (60-100 BPM)', 'Tachycardia (>100 BPM)']
class_colors = ['blue', 'green', 'red']

# prediction smoothing
recent_preds = deque(maxlen=5)
recent_confidences = deque(maxlen=5)
recent_bpms = deque(maxlen=5)

# Signal metrics
signal_stats = {
    'mean': 0,
    'std': 0,
    'min': 0,
    'max': 0,
    'range': 0
}

# Load my Model
print("Loading ECG Classification Model...")
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print("Model loaded successfully!")
except Exception as e:
    print(f"ERROR: Could not load model: {e}")
    exit()

# Signal Processing
def apply_bandpass_filter(signal, fs=SAMPLING_RATE_MODEL, lowcut=0.5, highcut=40.0):
    """Apply bandpass filter with proper parameters"""
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    
    # filtering
    b, a = scipy.signal.butter(3, [low, high], btype='band')
    
    # zero-phase filtering
    try:
        filtered = scipy.signal.filtfilt(b, a, signal)
        return filtered
    except Exception as e:
        print(f"Filter error: {e}")
        return signal

def normalize_signal(signal):
    """normalization with outlier handling"""
    # Remove outliers using percentile-based normalization
    p5, p95 = np.percentile(signal, [5, 95])
    clipped = np.clip(signal, p5, p95)
    
    mean = np.mean(clipped)
    std = np.std(clipped)
    
    if std < 0.01:  # Signal too flat
        return signal - mean
    
    normalized = (clipped - mean) / std
    
    # Clip extreme values
    normalized = np.clip(normalized, -5, 5)
    
    return normalized

def estimate_bpm_from_signal(signal, fs=SAMPLING_RATE_MODEL):
    """Calculate BPM using peak detection"""
    try:
        # Find peaks with adaptive threshold
        signal_mean = np.mean(signal)
        signal_std = np.std(signal)
        
        # Peaks should be above mean + 0.5*std
        peaks, properties = scipy.signal.find_peaks(
            signal,
            height=signal_mean + 0.3 * signal_std,
            distance=int(fs * 0.3),  # <-- FIXED: Allows up to ~200 BPM (0.3s between beats)
            prominence=0.3 * signal_std
        )
        
        if len(peaks) < 2:
            return None
        
        # Calculate RR intervals
        rr_intervals = np.diff(peaks) / fs  # in seconds
        
        # Remove outliers
        rr_mean = np.mean(rr_intervals)
        rr_std = np.std(rr_intervals)
        valid_rr = rr_intervals[np.abs(rr_intervals - rr_mean) < 2 * rr_std]
        
        if len(valid_rr) < 1:
            return None
        
        # Calculate BPM
        avg_rr = np.mean(valid_rr)
        bpm = 60.0 / avg_rr
        
        # heart rate range
        if 30 <= bpm <= 200:
            return bpm
        
        return None
        
    except Exception as e:
        print(f"BPM calculation error: {e}")
        return None

def check_signal_quality(signal):
    """signal quality check"""
    if len(signal) < 100:
        return False, "Insufficient data"
    
    # Check for flat signal
    if np.std(signal) < 0.01:
        return False, "Signal too flat - check electrodes"
    
    # Check for saturation
    signal_range = np.max(signal) - np.min(signal)
    if signal_range < 0.1:
        return False, "Signal saturated - adjust electrodes"
    
    # Check for excessive noise (coefficient of variation)
    cv = np.std(signal) / (np.abs(np.mean(signal)) + 1e-6)
    if cv > 10:
        return False, "Excessive noise detected"
    
    return True, "OK"

def predict_from_buffer():
    global current_prediction, confidence, bpm_estimate, signal_stats
    
    if len(data_buffer) < BUFFER_SAMPLES:
        current_prediction = f"Collecting data... ({len(data_buffer)}/{BUFFER_SAMPLES})"
        return

    try:
        # Get raw signal
        raw_signal = np.array(data_buffer)
        
        # Update signal statistics for display
        signal_stats['mean'] = np.mean(raw_signal)
        signal_stats['std'] = np.std(raw_signal)
        signal_stats['min'] = np.min(raw_signal)
        signal_stats['max'] = np.max(raw_signal)
        signal_stats['range'] = signal_stats['max'] - signal_stats['min']
        
        # Check signal quality FIRST
        is_good, quality_msg = check_signal_quality(raw_signal)
        if not is_good:
            current_prediction = quality_msg
            confidence = 0.0
            return
        
        # Signal processing pipeline 
        # 1. Resample to model's expected rate
        resampled = scipy.signal.resample(raw_signal, MODEL_SAMPLES)
        
        # 2. Apply bandpass filter
        filtered = apply_bandpass_filter(resampled, fs=SAMPLING_RATE_MODEL)
        
        # 3. Normalize
        normalized = normalize_signal(filtered)
        
        # 4. Estimate BPM directly from signal
        bpm = estimate_bpm_from_signal(normalized, fs=SAMPLING_RATE_MODEL)
        if bpm is not None:
            recent_bpms.append(bpm)
            bpm_estimate = np.mean(list(recent_bpms))
        
        # 5. Prepare for model
        model_input = normalized.reshape(1, MODEL_SAMPLES, 1)
        
        # Model prediction
        pred_probs = model.predict(model_input, verbose=0)[0]
        pred_class = np.argmax(pred_probs)
        current_confidence = pred_probs[pred_class] * 100
        
        # Prediction smoothing with confidence weighting
        recent_preds.append(pred_class)
        recent_confidences.append(current_confidence)
        
        if len(recent_preds) >= 3:
            # Use majority voting with confidence weighting
            class_votes = {0: 0, 1: 0, 2: 0}
            for pred, conf in zip(recent_preds, recent_confidences):
                class_votes[pred] += conf
            
            smoothed_class = max(class_votes, key=class_votes.get)
            confidence = np.mean(list(recent_confidences))
        else:
            smoothed_class = pred_class
            confidence = current_confidence
        
        # prediction string with BPM
        pred_text = class_names[smoothed_class]
        if bpm_estimate > 0:
            pred_text += f" | {bpm_estimate:.0f} BPM"
        
        current_prediction = pred_text
        
        # Print detailed diagnostics
        print(f"\n{'='*60}")
        print(f"Time: {time.strftime('%H:%M:%S')}")
        print(f"Signal Stats: Mean={signal_stats['mean']:.2f}, "
              f"Std={signal_stats['std']:.2f}, Range={signal_stats['range']:.2f}")
        if bpm_estimate > 0:
            print(f"Estimated BPM: {bpm_estimate:.1f}")
        print(f"Model Prediction: {class_names[smoothed_class]}")
        print(f"Confidence: {confidence:.1f}%")
        print(f"Class Probabilities: Bradycardia={pred_probs[0]*100:.1f}%, "
              f"Normal={pred_probs[1]*100:.1f}%, Tachycardia={pred_probs[2]*100:.1f}%")
        
    except Exception as e:
        print(f"Prediction error: {e}")
        import traceback
        traceback.print_exc()

# Plot Setup
plt.rcParams['font.size'] = 11
fig = plt.figure(figsize=(14, 9))
gs = fig.add_gridspec(3, 2, height_ratios=[2, 2, 1], hspace=0.3, wspace=0.3)

# Main ECG plot (processed signal)
ax1 = fig.add_subplot(gs[0, :])
ax1.set_ylim(-3, 3)
ax1.set_xlim(0, PLOT_SECONDS)
ax1.set_ylabel('Normalized ECG Amplitude')
ax1.set_xlabel('Time (seconds)')
ax1.set_title('Processed ECG Signal', fontweight='bold')
ax1.grid(True, alpha=0.3)
ecg_line, = ax1.plot([], [], 'b-', linewidth=1.2, label='Filtered ECG')
ax1.legend(loc='upper right')

# Raw signal plot
ax2 = fig.add_subplot(gs[1, 0])
ax2.set_ylim(-1, 1)
ax2.set_xlim(0, PLOT_SECONDS)
ax2.set_ylabel('Raw Amplitude')
ax2.set_xlabel('Time (seconds)')
ax2.set_title('Raw Arduino Signal', fontweight='bold')
ax2.grid(True, alpha=0.3)
raw_line, = ax2.plot([], [], 'g-', linewidth=0.8, alpha=0.7)

# Signal quality metrics
ax3 = fig.add_subplot(gs[1, 1])
ax3.set_xlim(0, 1)
ax3.set_ylim(0, 1)
ax3.axis('off')
ax3.set_title('Signal Quality', fontweight='bold')
quality_text = ax3.text(0.1, 0.7, 'Initializing...', fontsize=10, family='monospace')

# Prediction display
ax4 = fig.add_subplot(gs[2, :])
ax4.set_xlim(0, 1)
ax4.set_ylim(0, 1)
ax4.axis('off')
prediction_text = ax4.text(0.5, 0.7, 'Initializing...', 
                           ha='center', va='center', fontsize=18, fontweight='bold')
confidence_text = ax4.text(0.5, 0.3, 'Confidence: --%',
                           ha='center', va='center', fontsize=14)

fig.suptitle('Real-Time ECG Classification System', fontsize=16, fontweight='bold')

# Serial Setup
print(f"\nConnecting to ECG device on {SERIAL_PORT}...")
try:
    ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=0.1)
    time.sleep(2)  # Allow Arduino to reset
    ser.reset_input_buffer()  # Clear any startup noise
    print("Serial connection established!")
    print("Waiting for stable signal...\n")
except Exception as e:
    print(f"Serial error: {e}")
    print("TIPS:")
    print("  1. Close Arduino IDE Serial Monitor")
    print("  2. Check the COM port number in Device Manager")
    print("  3. Try unplugging and replugging the Arduino")
    exit()

# Initialize buffers with zeros
plot_buffer.extend([0.0] * PLOT_SAMPLES)
raw_buffer.extend([0.0] * PLOT_SAMPLES)

# Timing setup
update_counter = 0
PREDICT_EVERY = int(PREDICTION_INTERVAL * 50)  # Predict every N frames (50 fps)

# Update Loop
def update(frame):
    global update_counter, confidence
    global update_counter

    try:
        # Read all available serial data
        # Limit reads per frame to 50 to prevent lagging the animation
        lines_read = 0
        while ser.in_waiting > 0 and lines_read < 50:
            lines_read += 1
            try:
                line_bytes = ser.readline()
                if not line_bytes:
                    continue
                
                line_str = line_bytes.decode('utf-8', errors='ignore').strip()
                
                # Just try to convert. If it's not a valid float,
                # the except block will catch it.
                value = float(line_str)
                
                # Validate ADC range
                if not (0 <= value <= 1023):
                    continue
                
                # --- FIX #1: THE DOUBLE NORMALIZATION BUG ---
                # Store the RAW (0-1023) value in the model's buffer
                data_buffer.append(value)
                
                # Store normalized values for the plots
                raw_normalized = (value - ADC_CENTER) / ADC_CENTER
                raw_buffer.append(raw_normalized)
                
                processed_value = (value - ADC_CENTER) / 512.0
                plot_buffer.append(processed_value)
                
            except (UnicodeDecodeError, ValueError, TypeError):
                # Catches bad serial data, non-float strings, etc.
                continue
            except Exception as e:
                # Catches any other unexpected read error
                print(f"Serial read error: {e}")
                continue

        # Update ECG plots
        time_seconds = np.arange(len(plot_buffer)) / SAMPLING_RATE_ARDUINO
        ecg_line.set_data(time_seconds, plot_buffer)
        
        time_seconds_raw = np.arange(len(raw_buffer)) / SAMPLING_RATE_ARDUINO
        raw_line.set_data(time_seconds_raw, raw_buffer)

        # Update signal quality display
        quality_str = (
            f"Buffer: {len(data_buffer)}/{BUFFER_SAMPLES}\n"
            f"Mean: {signal_stats['mean']:.3f}\n"
            f"Std Dev: {signal_stats['std']:.3f}\n"
            f"Range: {signal_stats['range']:.3f}\n"
            f"Min/Max: {signal_stats['min']:.1f}/{signal_stats['max']:.1f}"
        )
        quality_text.set_text(quality_str)

        # Update prediction text & color
        pred_class = 1  # Default to normal
        if "Bradycardia" in current_prediction:
            pred_class = 0
        elif "Tachycardia" in current_prediction:
            pred_class = 2
        elif "check" in current_prediction.lower() or "noise" in current_prediction.lower() or "flat" in current_prediction.lower():
            pred_class = 1  # Neutral color for errors
            confidence = 0.0 # Reset confidence on error

        prediction_text.set_text(current_prediction)
        prediction_text.set_color(class_colors[pred_class])
        
        if confidence > 0:
            confidence_text.set_text(f'Confidence: {confidence:.1f}%')
        else:
            confidence_text.set_text('Confidence: --%')


        # Make predictions at intervals
        update_counter += 1
        if update_counter >= PREDICT_EVERY:
            update_counter = 0
            predict_from_buffer()

        # Return all artists that have been updated
        return ecg_line, raw_line, prediction_text, confidence_text, quality_text

    except Exception as e:
        print(f"FATAL Update loop error: {e}")
        import traceback
        traceback.print_exc()
        return ecg_line, raw_line, prediction_text, confidence_text, quality_text

print("\n" + "="*60)
print("REAL-TIME ECG MONITORING STARTED")
print("="*60)
print("Instructions:")
print("  1. Ensure electrodes are properly attached")
print("  2. Remain still during measurement")
print("  3. System will classify every 3 seconds")
print("  4. Close window to stop")
print("="*60 + "\n")

# Start animation
ani = FuncAnimation(fig, update, blit=True, interval=20, cache_frame_data=False)
plt.show()

# Cleanup
ser.close()
print("\n" + "="*60)
print("Application stopped. Serial connection closed.")
print("="*60)