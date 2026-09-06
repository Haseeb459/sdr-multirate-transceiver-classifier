import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt

# ==========================================
# 1. PARAMETERS & SIGNAL GENERATION
# ==========================================
np.random.seed(42)

num_symbols = 500
symbol_rate = 1000                  # R_s = 1 ksps
sps_base = 4                        # Baseband samples per symbol
fs_base = symbol_rate * sps_base    # Baseband sampling rate = 4 kHz

# Interpolation / Decimation factor for DUC/DDC
M = 8                               # Upsampling factor (Baseband -> IF)
fs_if = fs_base * M                 # IF sampling rate = 32 kHz
f_if = 8000                         # Intermediate Frequency carrier = 8 kHz

# Generate QPSK Symbols
bits = np.random.randint(0, 2, num_symbols * 2)
I_bits = 2 * bits[0::2] - 1
Q_bits = 2 * bits[1::2] - 1
symbols = (I_bits + 1j * Q_bits) / np.sqrt(2)

# Baseband Upsampling (1 symbol -> sps_base samples)
baseband_upsampled = np.zeros(num_symbols * sps_base, dtype=complex)
baseband_upsampled[::sps_base] = symbols

# RRC Pulse Shaping Filter (Baseband)
num_taps = 65
beta = 0.25
t = np.arange(-num_taps // 2 + 1, num_taps // 2 + 1) / sps_base
rrc_taps = np.sinc(t) * np.cos(np.pi * beta * t) / (1 - (2 * beta * t)**2 + 1e-8)
rrc_taps /= np.sum(rrc_taps)

baseband_tx = np.convolve(baseband_upsampled, rrc_taps, mode='same')

# ==========================================
# 2. DIGITAL UP-CONVERTER (DUC) - TX
# ==========================================
# A. Interpolation: Zero-stuffing by factor M
duc_interpolated = np.zeros(len(baseband_tx) * M, dtype=complex)
duc_interpolated[::M] = baseband_tx

# B. Anti-Aliasing Low-Pass Filter at IF Rate
# Cutoff at Nyquist of baseband rate (fs_base / 2) normalized to IF Nyquist (fs_if / 2)
cutoff_duc = (fs_base / 2) / (fs_if / 2)
b_duc = signal.firwin(101, cutoff_duc, window='hamming') * M  # Multiply by M to compensate zero-stuffing gain loss
duc_filtered = signal.convolve(duc_interpolated, b_duc, mode='same')

# C. Digital NCO & Mixing to IF
t_if = np.arange(len(duc_filtered)) / fs_if
nco_tx = np.exp(1j * 2 * np.pi * f_if * t_if)

# Real Passband IF Transmission
tx_if_signal = np.real(duc_filtered * nco_tx)

# ==========================================
# 3. CHANNEL SIMULATION (AWGN)
# ==========================================
snr_db = 15
snr_linear = 10 ** (snr_db / 10)
signal_power = np.mean(tx_if_signal ** 2)
noise_power = signal_power / snr_linear
noise = np.sqrt(noise_power) * np.random.randn(len(tx_if_signal))

rx_if_signal = tx_if_signal + noise

# ==========================================
# 4. DIGITAL DOWN-CONVERTER (DDC) - RX
# ==========================================
# A. NCO Mixing to Baseband (Complex Conjugate mixing)
nco_rx = np.exp(-1j * 2 * np.pi * f_if * t_if)
rx_mixed = rx_if_signal * nco_rx

# B. Anti-Aliasing LPF prior to Decimation
cutoff_ddc = (fs_base / 2) / (fs_if / 2)
b_ddc = signal.firwin(101, cutoff_ddc, window='hamming')
rx_filtered = signal.convolve(rx_mixed, b_ddc, mode='same')

# C. Decimation (Downsampling by factor M)
rx_baseband = rx_filtered[::M]

# D. Receiver Matched Filtering
rx_matched = np.convolve(rx_baseband, rrc_taps, mode='same')

# ==========================================
# 5. VISUALIZATION OF DSP PIPELINE
# ==========================================
plt.figure(figsize=(12, 8))

# Spectrum 1: Original Baseband
plt.subplot(3, 1, 1)
freqs, psd = signal.periodogram(baseband_tx, fs_base, return_onesided=False)
plt.plot(np.fft.fftshift(freqs), 10 * np.log10(np.fft.fftshift(psd) + 1e-12), 'b')
plt.title('1. Baseband Complex Spectrum (Fs = 4 kHz, Centered at 0 Hz)')
plt.ylabel('Power (dB)')
plt.grid(True)
plt.ylim([-80, 10])

# Spectrum 2: DUC Transmitted IF Signal
plt.subplot(3, 1, 2)
freqs_if, psd_if = signal.periodogram(tx_if_signal, fs_if, return_onesided=False)
plt.plot(np.fft.fftshift(freqs_if), 10 * np.log10(np.fft.fftshift(psd_if) + 1e-12), 'r')
plt.title('2. DUC Real IF Output Spectrum (Fs = 32 kHz, Upconverted to 8 kHz Carrier)')
plt.ylabel('Power (dB)')
plt.grid(True)
plt.ylim([-80, 10])

# Spectrum 3: DDC Recovered Baseband Signal
plt.subplot(3, 1, 3)
freqs_rx, psd_rx = signal.periodogram(rx_matched, fs_base, return_onesided=False)
plt.plot(np.fft.fftshift(freqs_rx), 10 * np.log10(np.fft.fftshift(psd_rx) + 1e-12), 'g')
plt.title('3. DDC Recovered Baseband Spectrum (Downconverted & Decimated back to Fs = 4 kHz)')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Power (dB)')
plt.grid(True)
plt.ylim([-80, 10])

plt.tight_layout()
plt.show()