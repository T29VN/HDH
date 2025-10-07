from data import generate_data
from estimation import music_doa_estimation
import matplotlib.pyplot as plt
import numpy as np

M, d, wavelength, N, snr_db = 10, 0.5, 1.0, 500, 15
doa_true_deg = 60.0

X, doa_true = generate_data(M, d, doa_true_deg, wavelength, N, snr_db)
doa_est, angles, ps = music_doa_estimation(X, wavelength, d)

print(f"DOA thật: {doa_true:.2f}°")
print(f"DOA ước lượng: {doa_est:.2f}°")

plt.plot(angles, 10*np.log10(ps/np.max(ps)))
plt.axvline(x=doa_true, color='r', linestyle='--')
plt.grid()
plt.show()
