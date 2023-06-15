import torch
import torchaudio
import pywt
import numpy as np
from scipy.stats import skew, kurtosis, entropy
from typing import Union, List

def calculate_energy(
    coeffs: List
) -> List:
    """
    Calculate the energy from the wavelet coefficients.

    Args:
        coeffs (List): the wavelet's coefficients.

    Returns:
        List: the energy of each wavelet coefficient.
    """
    energies = [sum(np.multiply(coeff, coeff))/coeff.shape[0] for coeff in coeffs]
    return energies

def calculate_mean(
    coeffs: List
) -> List:
    """
    Calculate the mean from the wavelet coefficients.

    Args:
        coeffs (List): the wavelet's coefficients.

    Returns:
        List: the mean of each wavelet coefficient.
    """
    means = [np.mean(coeff) for coeff in coeffs]
    return means

def calculate_median(
    coeffs: List
) -> List:
    """
    Calculate the median from the wavelet coefficients.

    Args:
        coeffs (List): the wavelet's coefficients.

    Returns:
        List: the median of each wavelet coefficient.
    """
    medians = [np.median(coeff) for coeff in coeffs]
    return medians

def calculate_std(
    coeffs: List
) -> List:
    """
    Calculate the standard deviation from the wavelet coefficients.

    Args:
        coeffs (List): the wavelet's coefficients.

    Returns:
        List: the standard deviation of each wavelet coefficient.
    """
    stds = [np.std(coeff) for coeff in coeffs]
    return stds

def calculate_skew(
    coeffs: List
) -> List:
    """
    Calculate the skewness from the wavelet coefficients.

    Args:
        coeffs (List): the wavelet's coefficients.

    Returns:
        List: the skewness of each wavelet coefficient.
    """
    skews = [skew(coeff) for coeff in coeffs]
    return skews

def calculate_kurtosis(
    coeffs: List
) -> List:
    """
    Calculate the kurtosis from the wavelet coefficients.

    Args:
        coeffs (List): the wavelet's coefficients.

    Returns:
        List: the kurtosis of each wavelet coefficient.
    """
    k = [kurtosis(coeff) for coeff in coeffs]
    return k

def calculate_entropy(
    coeffs: List
) -> List:
    """
    Calculate the entropy from the wavelet coefficients.

    Args:
        coeffs (List): the wavelet's coefficients.

    Returns:
        List: the entropy of each wavelet coefficient.
    """
    entropies = [entropy(coeff) for coeff in coeffs]
    return entropies

def extract_wavelet_from_spectrogram(
    spectrogram: torch.Tensor,
    wavelet: str,
    maxlevel: int,
    type: str,
    mode: str
) -> torch.Tensor:
    """
    Extract the wavelet from the mel spectrogram.

    Args:
        spectrogram (torch.Tensor): the mel spectrogram.
        wavelet (str): the wavelet's name.
        maxlevel (int): the wavelet's max level.
        type (str): which wavelet to extract (dwt).
        mode (str): how the wavelet should be extracted.

    Returns:
        torch.Tensor: the extracted wavelet.
    """
    if type == "dwt":
        use_gpu = False
        
        if spectrogram.is_cuda:
            use_gpu = True
            spectrogram = spectrogram.cpu()
            
        coeffs = pywt.wavedec2(
            data=spectrogram,
            level=maxlevel,
            wavelet=wavelet, 
            mode=mode
        )
        
        arr, coeffs = pywt.coeffs_to_array(coeffs)
        arr = torch.from_numpy(arr)
        
        if use_gpu:
            arr = arr.cuda()
            
        return arr, coeffs
    else:
        raise NotImplementedError

def extract_wavelet_from_raw_audio(
    audio: torch.Tensor,
    wavelet: str,
    maxlevel: int,
    type: str,
    mode: str
) -> List:
    """
    Extract the wavelet from a raw audio waveform.

    Args:
        audio (torch.Tensor): the audio waveform.
        wavelet (str): the wavelet's name.
        maxlevel (int): the wavelet's max level.
        type (str): which wavelet type to extract (packet or dwt).
        mode (str): how the wavelet should be extracted.

    Returns:
        pd.DataFrame: the extracted wavelet packet.
    """
    if type == "packet":
        datas = []
            
        audio = audio.numpy()
        
        wp = pywt.WaveletPacket(
            data=audio,
            wavelet=wavelet,
            mode=mode,
            maxlevel=maxlevel
        )
        
        if wp.maxlevel > 0:
            nodes = [node.path for node in wp.get_level(maxlevel, "natural")]
            
            for node in nodes:
                data = wp[node].data
                datas.append(data)
                
        return datas
    
    elif type == "dwt":
        audio = audio.cpu()
        
        coeffs = pywt.wavedec(
            data=audio,
            wavelet=wavelet,
            level=maxlevel,
            mode=mode,
        )
        return coeffs
    
def extract_mfcc(
    audio: torch.Tensor,
    sample_rate: int,
    n_fft: int,
    hop_length: int,
    n_mfcc: int,
    f_min: int = 0,
    f_max: Union[int, None] = None
) -> torch.Tensor:
    """
    Extracts the MFCC of a given audio.
    
    Args:
        audio (np.ndarray): the audio's waveform.
        sample_rate (int): the audio's sample rate.
        n_fft (int): the number of fft.
        hop_length (int): the hop length.
        n_mels (int): the number of mels.
        f_min (int): the minimum frequency.
        f_max (int): the maximum frequency.
        
    Returns:
        torch.Tensor: the extracted MFCC.
    """
    transform = torchaudio.transforms.MFCC(
        sample_rate=sample_rate,
        n_mfcc=n_mfcc,
        melkwargs={
            "n_fft": n_fft,
            "hop_length": hop_length,
            "f_min": f_min,
            "f_max": f_max
        }
    )
    mel_spectrogram = transform(audio)
    return mel_spectrogram

def extract_melspectrogram(
    audio: torch.Tensor,
    sample_rate: int,
    n_fft: int,
    hop_length: int,
    n_mels: int
) -> torch.Tensor:
    """
    Extracts the mel spectrogram of a given audio.
    
    Args:
        audio (np.ndarray): the audio's waveform.
        sample_rate (int): the audio's sample rate.
        n_fft (int): the number of fft.
        hop_length (int): the hop length.
        n_mels (int): the number of mels.
        
    Returns:
        torch.Tensor: the extracted Mel Spectrogram.
    """
    transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels
    )
    mel_spectrogram = transform(audio)
    return mel_spectrogram