import torch
import torch.nn as nn
import scipy
import scipy.stats
import scipy.signal
import numpy as np
from typing import Tuple, List, Dict
from octave_filtering import FilterByOctaves
from process_utils import check_format,discard_last_n_percent,rir_onset,_discard_trailing_zeros

"""Georg Götz, Ricardo Falcón Pérez, Sebastian J. Schlecht, and Ville Pulkki, "Neural network for multi-exponential sound energy decay analysis", The Journal of the Acoustical Society of America, 152(2), pp. 942-953, 2022, https://doi.org/10.1121/10.0013416."""


class PreprocessRIR(nn.Module):
    """ Preprocess a RIR to extract the EDC and prepare it for the neural network model.
        The preprocessing includes:

        # RIR -> Filterbank -> octave-band filtered RIR
        # octave-band filtered RIR -> backwards integration -> EDC
        # EDC -> delete last 5% of samples -> EDC_crop
        # EDC_crop -> downsample to the smallest number above 2400, i.e. by factor floor(original_length / 2400)
            -> EDC_ds1
        # EDC_ds1 -> as it might still be a little more than 2400 samples, just cut away everything after 2400 samples
            -> EDC_ds2
        # EDC_ds2 -> dB scale-> EDC_db
        # EDC_db -> normalization -> EDC_final that is the input to the network
    """

# instead of Octave Band filtering we need to badpass the RIRs
#TBD

    def __init__(self, input_transform: Dict = None, sample_rate: int = 48000, output_size: int = None,
                 filter_frequencies: List = None):
        super(PreprocessRIR, self).__init__()

        self.input_transform = input_transform
        self.output_size = output_size
        self.sample_rate = sample_rate
        self.eps = 1e-10

        self.filterbank = FilterByOctaves(order=5, sample_rate=self.sample_rate, backend='scipy',
                                          center_frequencies=filter_frequencies)

    def set_filter_frequencies(self, filter_frequencies):
        self.filterbank.set_center_frequencies(filter_frequencies)

    def get_filter_frequencies(self):
        return self.filterbank.get_center_frequencies()

    def forward(self, input_rir: torch.Tensor, input_is_edc: bool = False, analyse_full_rir=True) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict]:
        input_rir = check_format(input_rir)

        if input_is_edc:
            norm_vals = torch.max(input_rir, dim=-1, keepdim=True).values  # per channel
            schroeder_decays = input_rir / norm_vals
            if len(input_rir.shape) == 2:
                schroeder_decays = schroeder_decays.unsqueeze(1)
        else:
            # Extract decays from RIR: Do backwards integration
            schroeder_decays, norm_vals = self.schroeder(input_rir, analyse_full_rir=analyse_full_rir)

        # Convert to dB
        schroeder_decays_db = 10 * torch.log10(schroeder_decays + self.eps)

        # N values have to be adjusted for downsampling
        n_adjust = schroeder_decays_db.shape[-1] / self.output_size

        # DecayFitNet: T value predictions have to be adjusted for the time-scale conversion
        if self.input_transform is not None:
            t_adjust = 10 / (schroeder_decays_db.shape[-1] / self.sample_rate)
        else:
            t_adjust = 1

        # DecayFitNet: Discard last 5%
        if self.input_transform is not None:
            schroeder_decays_db = discard_last_n_percent(schroeder_decays_db, 5)

        # Resample to self.output_size samples (if given, otherwise keep sampling rate)
        if self.output_size is not None:
            schroeder_decays_db = torch.nn.functional.interpolate(schroeder_decays_db, size=self.output_size,
                                                                  mode='linear', align_corners=True)

        # DecayFitNet: Normalize with input transform
        if self.input_transform is not None:
            schroeder_decays_db = 2 * schroeder_decays_db / self.input_transform["edcs_db_normfactor"]
            schroeder_decays_db = schroeder_decays_db + 1

        # Write adjust factors into one dict
        scale_adjust_factors = {"t_adjust": t_adjust, "n_adjust": n_adjust}

        # Calculate time axis: be careful, because schroeder_decays_db might be on a different time scale!
        time_axis = torch.linspace(0, (schroeder_decays.shape[2] - 1) / self.sample_rate, schroeder_decays_db.shape[2])

        # Reshape freq bands as batch size, shape = [batch * freqs, timesteps]
        schroeder_decays_db = schroeder_decays_db.view(-1, schroeder_decays_db.shape[-1]).type(torch.float32)

        return schroeder_decays_db, time_axis, norm_vals, scale_adjust_factors

    def schroeder(self, rir: torch.Tensor, analyse_full_rir=True) -> Tuple[torch.Tensor, torch.Tensor]:
        # Check that RIR is in correct format/shape and return it in correct format if it wasn't before
        rir = check_format(rir)

        if not analyse_full_rir:
            onset = rir_onset(rir)
            rir = rir[..., onset:]

        out = _discard_trailing_zeros(rir)

        # Filter
        out = self.filterbank(out)

        # Remove filtering artefacts (last 5 permille)
        out = discard_last_n_percent(out, 0.5)

        # Backwards integral
        out = torch.flip(out, [2])
        out = (1 / out.shape[2]) * torch.cumsum(out ** 2, 2)
        out = torch.flip(out, [2])

        # Normalize to 1
        norm_vals = torch.max(out, dim=-1, keepdim=True).values  # per channel
        out = out / norm_vals

        return out, norm_vals.squeeze(2)