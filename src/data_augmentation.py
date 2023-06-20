import torch
import numpy as np
import torchaudio.transforms as T
from audiomentations import Compose, TimeMask, Trim, PitchShift, TimeStretch, TanhDistortion
from typing import Tuple
from scipy.signal import butter, sosfilt, sosfilt_zi
from src.utils import convert_frequency_to_mel, convert_mel_to_frequency

class AudioAugment:
    """
    Class responsible for applying data augmentation (time mask and trim) to the raw audio waveform.
    """
    def __init__(
        self,
        sample_rate: int,
        transformations: list,
        p: float = 1.0
    ) -> None:
        """
        Args:
            sample_rate (int): the audio sample rate.
            transformations (list): a list containing the transformations that will be applied ("time_mask" or "trim").
            p (float, optional): the probability of the audio augmentation being applied to the audio. Defaults to 1.0.
        """
        _valid_transformations = ["tanh_distortion", "pitch_shift", "time_strech", "time_mask", "trim"]
        
        assert 0 <= p <= 1.0
        assert all([t in _valid_transformations for t in transformations])
        assert sample_rate > 0
        
        self.p = p
        self.sample_rate = sample_rate
        self.transformations = transformations
        self.transformations_applied = []
            
    def __call__(
        self,
        audio: torch.Tensor
    ) -> torch.Tensor:
        """
        Applies the audio augmentation to the given audio's waveform.

        Args:
            waveform (torch.Tensor): the audio's waveform.

        Returns:
            torch.Tensor: the augmented audio's waveform.
        """
        audio = audio.squeeze(0).numpy()
                
        for transformation in self.transformations:
            if transformation == "time_mask":
                self.transformations_applied.append(TimeMask(
                    min_band_part=0.1,
                    max_band_part=0.15,
                    fade=True,
                    p=self.p
                ))
            elif transformation == "trim":
                self.transformations_applied.append(Trim(
                    top_db=10.0,
                    p=self.p
                ))
            elif transformation == "pitch_shift":
                self.transformations_applied.append(PitchShift(
                    p=self.p
                ))
            elif transformation == "time_stretch":
                self.transformations_applied.append(TimeStretch(
                    min_rate=0.8,
                    max_rate=1.25,
                    leave_length_unchanged=True,
                    p=self.p
                ))
            elif transformation == "tanh_distortion":
                self.transformations_applied.append(TanhDistortion(
                    min_distortion=0.01,
                    max_distortion=0.2,
                    p=self.p
                ))
        
        augment = Compose(self.transformations_applied)
        audio = augment(samples=audio, sample_rate=self.sample_rate)
        audio = torch.from_numpy(audio).to(dtype=torch.float32).unsqueeze(0)
        return audio        
    
class Denoiser:
    """
    Class responsible for denoising the audio's waveform (applies low pass filter and/or higher pass filter).
    """
    def __init__(
        self,
        filters: list,
        sample_rate: int,
        p: float = 1.0
    ) -> None:
        """
        Args:
            p (float): the probability of the denoiser filter being applied to the audio. Default is 1.
            filters (list): a list containing the filters that will be applied ("low_pass" or "high_pass").
            sample_rate (int): the audio's sample rate.
        """
        _valid_filters = ["low_pass", "high_pass"]
        
        assert 0 <= p <= 1.0
        assert all([f in _valid_filters for f in filters])
        
        self.p = p
        self.filters = filters
        self.sample_rate = sample_rate
        self.min_cutoff_freq = 0
        self.max_cutoff_freq = 0
        self.w = 0
    
    def __call__(
        self,
        waveform: torch.Tensor
    ) -> torch.Tensor:
        """
        Applies the denoising transformation to the given audio's waveform.

        Args:
            waveform (torch.Tensor): the audio's waveform.

        Returns:
            torch.Tensor: the denoised audio's waveform.
        """
        rand = torch.rand(1).item()

        if rand < self.p:
            for fil in self.filters:
                if fil == "low_pass":
                    self.min_cutoff_freq = 150
                    self.max_cutoff_freq = self.sample_rate // 2

                    # cutoff_mel = np.random.uniform(
                    #     low=convert_frequency_to_mel(self.min_cutoff_freq),
                    #     high=convert_frequency_to_mel(self.max_cutoff_freq),
                    # )
                    # self.w = convert_mel_to_frequency(cutoff_mel)
                    self.w = 400
                    
                    waveform = self._apply_filter(
                        waveform,
                        2,
                        self.w,
                        "low",
                        self.sample_rate
                    )
                elif fil == "high_pass":
                    self.min_cutoff_freq = 20
                    self.max_cutoff_freq = 2400

                    # cutoff_mel = np.random.uniform(
                    #     low=convert_frequency_to_mel(self.min_cutoff_freq),
                    #     high=convert_frequency_to_mel(self.max_cutoff_freq),
                    # )
                    # self.w = convert_mel_to_frequency(cutoff_mel)
                    self.w = 700

                    waveform = self._apply_filter(
                        waveform,
                        2,
                        self.w,
                        "high",
                        self.sample_rate
                    )
        
        return waveform
    
    def _apply_filter(
        self,
        waveform: torch.Tensor,
        order: int,
        w: float,
        filter_type: str,
        sample_rate: int,
        analog: bool = False
    ) -> torch.Tensor:
        """
        Applies the Butterworth's low/high pass filter.

        Args:
            waveform (torch.Tensor): the audio's waveform.
            order (int): the order of the filter.
            w (float): the critical frequency or frequencies (W is a scalar).
            filter_type (str): the type of filter ("low" or "high").
            sample_rate (int): the audio's sample rate.
            analog (bool, optional): when True, return an analog filter, otherwise a digital filter is returned. Defaults to False.

        Returns:
            torch.Tensor: the denoised audio's waveform.
        """
        assert 0 < w <= sample_rate // 2
        
        signal = waveform.squeeze(0).numpy()
        sos = butter(
            order,
            w,
            filter_type,
            analog=analog,
            output="sos",
            fs=sample_rate
        )
        
        processed_samples, _ = sosfilt(
            sos, signal, zi=sosfilt_zi(sos) * signal[0]
        )
        processed_samples = processed_samples.astype(np.float32)
        processed_samples = torch.from_numpy(processed_samples.unsqueeze(0))
        return processed_samples
    
class SpecAugment:
    """
    Class responsible for applying the spectrogram augmentation to the audio's mel spectrogram/spectrogram
    (applies frequency masking and/or time masking).
    """
    def __init__(
        self,
        p: float,
        transformations: list,
        mask_samples: int,
        feature: str
    ) -> None:
        """
        Args:
            p (float): the probability of the denoiser filter being applied to the audio. Default is 1.
            transformations (list): a list containing the transformations that will be applied ("time_mask" or "frequency_mask").
            mask_samples (int): maximum possible length of the mask. Indices uniformly sampled from [0, mask_samples).
        """
        _valid_transformations = ["time_mask", "frequency_mask"]
        _valid_features = ["mel_spectrogram", "mfcc"]
        
        assert 0 <= p <= 1.0
        assert all([t in _valid_transformations for t in transformations])
        assert mask_samples > 0
        assert feature in _valid_features
        
        self.p = p
        self.transformations = transformations
        self.mask_samples = mask_samples
        self.feature = feature
    
    def __call__(
        self,
        spec: torch.Tensor
    ) -> torch.Tensor:
        """
        Applies the spectrogram augmentation to the given audio's mel spectrogram/spectrogram.

        Args:
            spec (torch.Tensor): the audio's mel spectrogram/spectrogram.

        Returns:
            torch.Tensor: the augmented mel spectrogram/spectrogram.
        """
        rand = torch.rand(1).item()

        if rand < self.p:
            if self.feature == "melspectrogram":
                spec = self._augment_mel_spec(spec)
            elif self.feature == "mfcc":
                spec = self._augment_mfcc(spec)
        
        return spec
    
    def _augment_mel_spec(
        self,
        spec: torch.Tensor
    ) -> torch.Tensor:
        """
        Applies the data augmentation to the given mel spectrogram.

        Args:
            spec (torch.Tensor): the mel spectrogram.

        Returns:
            torch.Tensor: the augmented mel spectrogram.
        """
        for transformation in self.transformations:
            if transformation == "time_mask":
                masking = T.TimeMasking(time_mask_param=self.mask_samples)
            elif transformation == "frequency_mask":
                masking = T.FrequencyMasking(freq_mask_param=self.mask_samples)
            spec = masking(spec)

        return spec
    
    #this function is used to mask along multiple consecutive frames - see https://github.com/s3prl/s3prl/blob/master/pretrain/mockingjay/task.py
    def _starts_to_intervals(
        self,
        starts: torch.Tensor,
        consecutive: int
    ) -> torch.Tensor:
        tiled = starts.expand(consecutive, starts.size(0)).permute(1, 0)
        offset = torch.arange(consecutive).expand_as(tiled)
        intervals = tiled + offset
        return intervals.view(-1)

    def _augment_mfcc(
        self,
        spec: torch.Tensor
    ) -> torch.Tensor:
        """
        Applies data augmentation to the given MFCC.

        Args:
            spec (torch.Tensor): the MFCC.

        Returns:
            torch.Tensor: the augmented MFCC.
        """
        for transformation in self.transformations:
            time_len = spec.shape[2]
            n_coeffs = spec.shape[1]
            
            if transformation == "time_mask":
                perm_length = np.random.randint(self.mask_samples)
                valid_start_max = max(time_len - perm_length - 1, 0)
                chosen_starts = torch.randperm(valid_start_max + 1)[:1]
                chosen_intervals = self._starts_to_intervals(chosen_starts, perm_length)
                spec[:, :, chosen_intervals] = 0
            elif transformation == "frequency_mask":
                perm_length = np.random.randint(self.mask_samples)
                valid_start_max = max(n_coeffs - perm_length - 1, 0)
                chosen_starts = torch.randperm(valid_start_max + 1)[:1]
                chosen_intervals = self._starts_to_intervals(chosen_starts, perm_length)
                spec[:, chosen_intervals, :] = 0
        
        return spec

class Mixup:
    """
    Class responsible for applying the Mixup data augmentation technique.
    """
    def __init__(
        self,
        alpha: float = 1.0,
    ) -> None:
        """
        Args:
            alpha (float, optional): the alpha that will be used to extract sample
                                     from the beta distribution. Defaults to 1.0.
        """
        self.alpha = alpha
        self.lam = 0.0
    
    def __call__(
        self,
        x: torch.Tensor,
        y: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Applies the mixup technique to the given x and y.

        Args:
            x (torch.Tensor): the data features.
            y (torch.Tensor): the features label.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: the mixed x and y, respectively.
        """
        if self.alpha > 0:
            self.lam = np.random.beta(self.alpha, self.alpha, 1)[0]
        else:
            self.lam = 1
        
        batch_size = x.size()[0]
        index = torch.randperm(batch_size)

        mixed_x = self.lam * x + (1 - self.lam) * x[index, :]
        mixed_y = self.lam * y + (1 - self.lam) * y[index]
        return mixed_x, mixed_y