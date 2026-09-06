import numpy as np
import scipy.signal as signal
from scipy.stats import kurtosis, skew
import matplotlib.pyplot as plt

# ==========================================
# 1. SIGNAL GENERATORS
# ==========================================
def generate_qpsk(num_samples, fs):
    sps = 8
    num_symbols = num_samples // sps
    bits = np.random.randint(0, 2, num_symbols * 2)
    symbols = ( (2*bits[0::2] - 1) + 1j*(2*bits[1::2] - 1) ) / np.sqrt(2)
    
    upsampled = np.zeros(num_symbols * sps, dtype=complex)
    upsampled[::sps] = symbols
    
    # RRC Pulse Shaping
    t = np.arange(-32, 33) / sps
    rrc = np.sinc(t) * np.cos(np.pi * 0.25 * t) / (1 - (2 * 0.25 * t)**2 + 1e-8)
    rrc /= np.sum(rrc)
    
    sig = np.convolve(upsampled, rrc, mode='same')
    return sig[:num_samples]

def generate_lfm_chirp(num_samples, fs):
    t = np.arange(num_samples) / fs
    f_start = -fs / 4
    f_end = fs / 4
    # Linear Frequency Modulation (LFM Chirp)
    chirp_phase = 2 * np.pi * (f_start * t + (f_end - f_start) / (2 * t[-1]) * t**2)
    return np.exp(1j * chirp_phase)

# ==========================================
# 2. FEATURE EXTRACTION ENGINE
# ==========================================
def extract_features(signal_iq):
    # Normalized Instantaneous Amplitude (Envelope)
    env = np.abs(signal_iq)
    env_norm = env / np.mean(env)
    
    # Feature 1: Variance of Normalized Inst. Amplitude
    gamma_max = np.var(env_norm)
    
    # Feature 2: Kurtosis of Amplitude
    kurt = kurtosis(env)
    
    # Feature 3: Standard Deviation of Inst. Frequency Derivative
    inst_phase = np.unwrap(np.angle(signal_iq))
    inst_freq = np.diff(inst_phase)
    std_freq = np.std(inst_freq)
    
    # Feature 4: Higher-Order Cumulant C40
    # C40 = E[x^4] - 3*(E[x^2])^2
    c40 = np.abs(np.mean(signal_iq**4) - 3 * (np.mean(signal_iq**2))**2)
    
    return [gamma_max, kurt, std_freq, c40]

# ==========================================
# 3. GENERATE TEST SIGNALS & EXTRACT FEATURES
# ==========================================
fs = 8000
num_samples = 1024
snr_db = 10

# Generate Raw Signals
qpsk_sig = generate_qpsk(num_samples, fs)
chirp_sig = generate_lfm_chirp(num_samples, fs)

# Add AWGN Noise
def add_noise(sig, snr):
    p_sig = np.mean(np.abs(sig)**2)
    p_noise = p_sig / (10**(snr/10))
    noise = np.sqrt(p_noise/2) * (np.random.randn(len(sig)) + 1j*np.random.randn(len(sig)))
    return sig + noise

qpsk_noisy = add_noise(qpsk_sig, snr_db)
chirp_noisy = add_noise(chirp_sig, snr_db)

# Extract Features
feat_qpsk = extract_features(qpsk_noisy)
feat_chirp = extract_features(chirp_noisy)

print("="*55)
print("             EXTRACTED FEATURE COMPARISON          ")
print("="*55)
print(f"Feature Metric             | QPSK (Comm) | LFM Chirp (Radar)")
print("-" * 55)
print(f"1. Var(Norm Envelope)      | {feat_qpsk[0]:.4f}      | {feat_chirp[0]:.4f}")
print(f"2. Envelope Kurtosis       | {feat_qpsk[1]:.4f}      | {feat_chirp[1]:.4f}")
print(f"3. Std(Inst. Frequency)    | {feat_qpsk[2]:.4f}      | {feat_chirp[2]:.4f}")
print(f"4. Higher-Order Cumulant C40| {feat_qpsk[3]:.4f}      | {feat_chirp[3]:.4f}")
print("="*55)

# ==========================================
# 4. TIME-FREQUENCY SPECTROGRAM VISUALIZATION
# ==========================================
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
f_q, t_q, Sxx_q = signal.spectrogram(qpsk_noisy, fs, return_onesided=False, nperseg=64)
plt.pcolormesh(t_q * 1000, np.fft.fftshift(f_q), np.fft.fftshift(10 * np.log10(Sxx_q), axes=0), shading='gouraud')
plt.title('Communication Signal: QPSK Spectrogram')
plt.xlabel('Time (ms)')
plt.ylabel('Frequency (Hz)')
plt.colorbar(label='Power (dB)')

plt.subplot(1, 2, 2)
f_c, t_c, Sxx_c = signal.spectrogram(chirp_noisy, fs, return_onesided=False, nperseg=64)
plt.pcolormesh(t_c * 1000, np.fft.fftshift(f_c), np.fft.fftshift(10 * np.log10(Sxx_c), axes=0), shading='gouraud')
plt.title('Radar Waveform: LFM Chirp Spectrogram')
plt.xlabel('Time (ms)')
plt.ylabel('Frequency (Hz)')
plt.colorbar(label='Power (dB)')

plt.tight_layout()
plt.show()