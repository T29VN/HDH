import numpy as np

def music_doa_estimation(X, wavelength, d, n_signals=1):
    M, N = X.shape
    ant_positions = np.arange(M) * d

    R = (X @ X.conj().T) / N
    eigenvalues, eigenvectors = np.linalg.eig(R)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvectors = eigenvectors[:, idx]

    En = eigenvectors[:, n_signals:]

    search_angles_deg = np.linspace(-90, 90, 361)
    pseudospectrum = []
    for angle in np.deg2rad(search_angles_deg):
        a_scan = np.exp(-1j * 2*np.pi/wavelength * ant_positions * np.sin(angle))
        denom = a_scan.conj().T @ En @ En.conj().T @ a_scan
        pseudospectrum.append(1 / np.abs(denom))

    pseudospectrum = np.array(pseudospectrum)
    doa_estimated_deg = search_angles_deg[np.argmax(pseudospectrum)]
    return doa_estimated_deg, search_angles_deg, pseudospectrum
