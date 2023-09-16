import torch
import torchaudio
import pywt
from typing import Union, List, Tuple


def extract_wavelet_from_spectrogram(
    spectrogram: torch.Tensor, wavelet: str, maxlevel: int, type: str, mode: str
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
            data=spectrogram, level=maxlevel, wavelet=wavelet, mode=mode
        )

        arr, coeffs = pywt.coeffs_to_array(coeffs)
        arr = torch.from_numpy(arr)

        if use_gpu:
            arr = arr.cuda()

        return arr, coeffs
    else:
        raise NotImplementedError


def extract_wavelet_from_raw_audio(
    audio: torch.Tensor, wavelet: str, maxlevel: int, type: str, mode: str
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
            data=audio, wavelet=wavelet, mode=mode, maxlevel=maxlevel
        )

        if wp.maxlevel > 0:
            nodes = [node.path for node in wp.get_level(maxlevel, "natural")]

            for node in nodes:
                data = wp[node].data
                datas.append(data)

        return datas

    elif type == "dwt":
        audio = audio.numpy()

        coeffs = pywt.wavedec(
            data=audio,
            wavelet=wavelet,
            level=maxlevel,
            mode=mode,
        )
        return coeffs
    else:
        raise NotImplementedError


def extract_mfcc(
    audio: torch.Tensor,
    sample_rate: int,
    n_fft: int,
    hop_length: int,
    n_mfcc: int,
    f_min: int = 0,
    f_max: Union[int, None] = None,
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
            "f_max": f_max,
        },
    )
    mel_spectrogram = transform(audio)
    return mel_spectrogram


def extract_melspectrogram(
    audio: torch.Tensor,
    sample_rate: int,
    n_fft: int,
    hop_length: int,
    n_mels: int,
    f_min: float = 0,
    f_max: Tuple[float, None] = None,
) -> torch.Tensor:
    """
    Extracts the mel spectrogram of a given audio.

    Args:
        audio (np.ndarray): the audio's waveform.
        sample_rate (int): the audio's sample rate.
        n_fft (int): the number of fft.
        hop_length (int): the hop length.
        n_mels (int): the number of mels.
        f_min (float): the minimum frequency. Default is 0.
        f_max (float): the maximum frequency. Default is None.

    Returns:
        torch.Tensor: the extracted Mel Spectrogram.
    """
    transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        f_min=f_min,
        f_max=f_max,
    )
    mel_spectrogram = transform(audio)
    return mel_spectrogram
