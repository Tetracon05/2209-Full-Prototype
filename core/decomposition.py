"""
decomposition.py — Wrappers for signal decomposition algorithms:
  VMD  : Variational Mode Decomposition (vmdpy)
  EMD  : Empirical Mode Decomposition (PyEMD)
  EEMD : Ensemble EMD (PyEMD)
  CEEMDAN : Complete EEMD with Adaptive Noise (PyEMD)
"""

import numpy as np


def decompose(signal: np.ndarray,
              method: str = "EMD",
              n_components: int = 5,
              **kwargs) -> np.ndarray:
    """
    Decompose a 1-D signal into Intrinsic Mode Functions (IMFs).

    Parameters
    ----------
    signal       : 1-D numpy array (e.g. Active_Power column)
    method       : 'EMD', 'EEMD', 'CEEMDAN', or 'VMD'
    n_components : desired number of IMF components (used by VMD; EMD
                   variants produce variable counts and are trimmed/padded)
    **kwargs     : extra arguments forwarded to the underlying library

    Returns
    -------
    imfs : np.ndarray of shape (k, len(signal))
    """
    signal = np.asarray(signal, dtype=np.float64)

    if method == "VMD":
        return _vmd(signal, n_components, **kwargs)
    elif method == "EMD":
        return _emd(signal, n_components)
    elif method == "EEMD":
        return _eemd(signal, n_components)
    elif method == "CEEMDAN":
        return _ceemdan(signal, n_components)
    else:
        raise ValueError(f"Unknown decomposition method: {method}. "
                         "Choose from EMD, EEMD, CEEMDAN, VMD.")


# ------------------------------------------------------------------
# VMD
# ------------------------------------------------------------------
def _vmd(signal: np.ndarray, K: int, **kwargs) -> np.ndarray:
    """
    Variational Mode Decomposition via the vmdpy library.
    Falls back to simple FFT-based mock if vmdpy is unavailable.
    """
    try:
        from vmdpy import VMD  # noqa
        alpha  = kwargs.get("alpha", 2000)    # bandwidth constraint
        tau    = kwargs.get("tau", 0.0)       # noise tolerance
        DC     = kwargs.get("DC", 0)          # include DC part
        init   = kwargs.get("init", 1)        # initialize omegas uniformly
        tol    = kwargs.get("tol", 1e-7)      # convergence tolerance
        u, _, _ = VMD(signal, alpha, tau, K, DC, init, tol)
        return u  # shape (K, N)
    except ImportError:
        # Graceful fallback: split frequency bands with FFT
        return _fft_fallback(signal, K)


# ------------------------------------------------------------------
# EMD
# ------------------------------------------------------------------
def _emd(signal: np.ndarray, max_imf: int) -> np.ndarray:
    """Empirical Mode Decomposition using PyEMD."""
    try:
        from PyEMD import EMD
        emd = EMD()
        emd.MAX_ITERATION = 200
        imfs = emd.emd(signal, max_imf=max_imf)
        return _trim_pad(imfs, max_imf, len(signal))
    except ImportError:
        return _fft_fallback(signal, max_imf)


# ------------------------------------------------------------------
# EEMD
# ------------------------------------------------------------------
def _eemd(signal: np.ndarray, max_imf: int) -> np.ndarray:
    """Ensemble Empirical Mode Decomposition using PyEMD."""
    try:
        from PyEMD import EEMD
        eemd = EEMD()
        eemd.noise_seed(42)
        imfs = eemd.eemd(signal, max_imf=max_imf)
        return _trim_pad(imfs, max_imf, len(signal))
    except ImportError:
        return _fft_fallback(signal, max_imf)


# ------------------------------------------------------------------
# CEEMDAN
# ------------------------------------------------------------------
def _ceemdan(signal: np.ndarray, max_imf: int) -> np.ndarray:
    """Complete EEMD with Adaptive Noise using PyEMD."""
    try:
        from PyEMD import CEEMDAN
        ceemdan = CEEMDAN()
        ceemdan.noise_seed(42)
        imfs = ceemdan.ceemdan(signal, max_imf=max_imf)
        return _trim_pad(imfs, max_imf, len(signal))
    except ImportError:
        return _fft_fallback(signal, max_imf)


# ------------------------------------------------------------------
# Utilities
# ------------------------------------------------------------------
def _trim_pad(imfs: np.ndarray, n: int, length: int) -> np.ndarray:
    """
    Ensure the output is exactly (n, length):
    - Trim if more IMFs than requested
    - Pad with zeros if fewer
    """
    k = imfs.shape[0]
    result = np.zeros((n, length), dtype=np.float64)
    take = min(k, n)
    for i in range(take):
        arr = imfs[i]
        l = min(len(arr), length)
        result[i, :l] = arr[:l]
    return result


def _fft_fallback(signal: np.ndarray, K: int) -> np.ndarray:
    """
    Simple FFT-based band-splitting fallback used when decomposition
    libraries are not installed.  Divides the spectrum into K equal bands.
    """
    N = len(signal)
    freqs = np.fft.rfft(signal)
    n_freq = len(freqs)
    band_size = max(1, n_freq // K)
    imfs = np.zeros((K, N), dtype=np.float64)
    for i in range(K):
        band = np.zeros_like(freqs)
        lo = i * band_size
        hi = lo + band_size if i < K - 1 else n_freq
        band[lo:hi] = freqs[lo:hi]
        imfs[i] = np.fft.irfft(band, n=N)
    return imfs
