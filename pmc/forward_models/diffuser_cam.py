import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from pmc.forward_models.base import BaseForwardModel
def nextPow2(n):
    return int(2**np.ceil(np.log2(n)))
import torch
import math

def next_pow2(n):
    return 2 ** math.ceil(math.log2(n))

def fft_conv(img, psf):
    """
    Performs FFT-based convolution using PyTorch.
    
    Args:
        img: Input image tensor (H, W)
        psf: Point spread function tensor (H, W)
    
    Returns:
        Convolved image tensor (H, W)
    """
    # Get dimensions

    psf = psf.squeeze(0)
    img = img.squeeze(0).squeeze(0)
    h_psf, w_psf = psf.shape
    h_img, w_img = img.shape

    # Calculate padded dimensions based on PSF size
    padded_shape = (
        next_pow2(2 * h_psf - 1),
        next_pow2(2 * w_psf - 1)
    )

    # Pad PSF
    pad_h_psf = (padded_shape[0] - h_psf)
    pad_w_psf = (padded_shape[1] - w_psf)
    psf_padded = torch.nn.functional.pad(
        psf,
        (pad_w_psf//2, pad_w_psf - pad_w_psf//2,
         pad_h_psf//2, pad_h_psf - pad_h_psf//2)
    )

    # Pad image
    pad_h_img = (padded_shape[0] - h_img)
    pad_w_img = (padded_shape[1] - w_img)
    img_padded = torch.nn.functional.pad(
        img,
        (pad_w_img//2, pad_w_img - pad_w_img//2,
         pad_h_img//2, pad_h_img - pad_h_img//2)
    )

    # Orthogonal normalization factor
    N = math.sqrt(padded_shape[0] * padded_shape[1])

    # FFT calculations
    H = torch.fft.fft2(psf_padded) / N
    V = torch.fft.fft2(img_padded) / N
    
    # Frequency domain multiplication
    result_freq = H * V

    # Inverse FFT and normalization
    result = torch.fft.ifft2(result_freq) * N
    result_real = result.real

    # Crop to original image dimensions
    start_h = (padded_shape[0] - h_img) // 2
    start_w = (padded_shape[1] - w_img) // 2
    
    return result_real[start_h:start_h+h_img, start_w:start_w+w_img].unsqueeze(0).unsqueeze(0)


class DiffuserCam(BaseForwardModel):
    def __init__(self, input_snr, var, psf_path, device, shape=128):
        super().__init__(input_snr, var)
        self.device = device
        self.psf = self._prepare_psf(psf_path, shape)  # Load and preprocess PSF
        self.psf = self.psf.to(device)
        self.conj = torch.conj(self.psf)

    def _prepare_psf(self, psf_path, size):
        """Loads, preprocesses, and converts the PSF to a tensor."""
        DIMS = 1
        psf_img = Image.open(psf_path).resize((size, size), Image.BICUBIC)
        psf_array = np.array(psf_img, dtype=np.float32)
        background = np.mean(psf_array[5:15, 5:15])
        psf_array -= background
        psf_array = np.clip(psf_array, 0, None)
        psf_array /= np.sum(psf_array)
        psf_array *= 3
        # repeat psf for each channel
        psf_tensor = torch.tensor(psf_array, dtype=torch.float32)
        psf_tensor = psf_tensor.unsqueeze(0).repeat(DIMS, 1, 1)
        return psf_tensor
    def forward(self, data):
        return self.A(data)
    def adjoint(self, x):
        return self.A_p(x)
    def grad(self, x, y):
        Av = self.A(x)
        diff = Av - y
        grad = torch.real(self.A_p(diff))
        print("grad shape", grad.shape)
        return grad
    def A(self, data, **kwargs):
        """Simulates blurring by convolving the input with the PSF."""
        x = fft_conv(data, self.psf)
        print("A shape", x.shape)
        return x

    def A_p(self, x):
        """Adjoint operation: Convolution with the conjugate of the flipped PSF."""
        x = fft_conv(x, self.conj)
        print("A_p shape", x.shape)
        return x

        
