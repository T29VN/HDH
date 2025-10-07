import numpy as np
import matplotlib.pyplot as plt

def generate_data(M, d, doa_true_deg, wavelength, N, snr_db):
    ant_positions = np.arange(M) * d
    doa_true_rad = np.deg2rad(doa_true_deg)

    def steering_vector(doa_rad):
        k = 2 * np.pi / wavelength
        return np.exp(-1j * k * ant_positions * np.sin(doa_rad))

    a_true = steering_vector(doa_true_rad)
    source_signal = (np.random.randn(N) + 1j*np.random.randn(N)) / np.sqrt(2)
    signal_matrix = np.outer(a_true, source_signal)

    signal_power = np.mean(np.abs(source_signal)**2)
    noise_power = signal_power / (10**(snr_db / 10))
    noise_matrix = (np.random.randn(M, N) + 1j*np.random.randn(M, N)) * np.sqrt(noise_power/2)

    X = signal_matrix + noise_matrix
    return X, doa_true_deg
