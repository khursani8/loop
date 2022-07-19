# from data import augspec
import torch
from nnAudio import Spectrogram
import librosa

mult = 1
sample_rate = 16000
n_fft = 512
lowfreq = 0.
highfreq = 8000.
nfilt = 80

def normalize_batch_custom(x,seq_len):
    x_mean = torch.zeros((seq_len.shape[0], x.shape[1]), dtype=x.dtype, device=x.device)
    x_std = torch.zeros((seq_len.shape[0], x.shape[1]), dtype=x.dtype, device=x.device)
    for i in range(x.shape[0]):
        if x[i, :, : seq_len[i]].shape[1] == 1:
            raise ValueError(
                "normalize_batch with `per_feature` normalize_type received a tensor of length 1. This will result "
                "in torch.std() returning nan"
            )
        x_mean[i, :] = x[i, :, : seq_len[i]].mean(dim=1)
        x_std[i, :] = x[i, :, : seq_len[i]].std(dim=1)
    # make sure x_std is not zero
    x_std += 1e-5
    return (x - x_mean.unsqueeze(2)) / x_std.unsqueeze(2)

filterbank = torch.tensor(
    librosa.filters.mel(sample_rate, n_fft, n_mels=nfilt, fmin=lowfreq, fmax=highfreq), dtype=torch.float
).unsqueeze(0)

Conv_STFT = Spectrogram.STFT(sr=16000, n_fft=512,
                                    hop_length=160*mult, window='hann', center=True,win_length=320*mult,
                                    pad_mode='reflect',  fmin=lowfreq, fmax=highfreq,trainable=False) \

def get_seq_len(seq_len):
    # Assuming that center is True is stft_pad_amount = 0
    pad_amount = 512 // 2 * 2
    seq_len = torch.floor((seq_len + pad_amount - 512) / 160) + 1
    return seq_len.to(dtype=torch.long)

def get_melspec(y,seq_len=None,dither=0,pad_to=16,blank=27):
    # y = torch.as_tensor(y)
    if seq_len is None:
        if y.size(0) > 1 and y.ndim > 1:
            seq_len = torch.stack([torch.tensor(i.shape[-1]) for i in y])
        else:
            seq_len = torch.as_tensor([y.shape[-1]])
            y = y[None]
    seq_len = get_seq_len(seq_len.float())
    if dither:
        y += dither * torch.randn_like(y)
    y = torch.cat((y[:, 0].unsqueeze(1), y[:, 1:] - 0.97 * y[:, :-1]), dim=1)
    if str(y.device) != "cpu":
        Conv_STFT.to(y.device)
    y = Conv_STFT(y)
    if y.dtype in [torch.cfloat, torch.cdouble]:
        y = torch.view_as_real(y)
    y = torch.sqrt(y.pow(2).sum(-1) + 0)
    y = y.pow(2) # power spectrum
    y = torch.matmul(filterbank.to(y.dtype).to(y.device), y)
    y = torch.log(y + 2 ** -24)
    y = normalize_batch_custom(y, seq_len)
    max_len = y.size(-1)
    mask = torch.arange(max_len).to(y.device)
    mask = mask.expand(y.size(0), max_len) >= seq_len.unsqueeze(1)
    y = y.masked_fill(mask.unsqueeze(1).type(torch.bool).to(device=y.device), 0)
    del mask
    if pad_to:
        pad_amt = y.size(-1) % pad_to
        if pad_amt != 0:
            y = torch.nn.functional.pad(y, (0, pad_to - pad_amt), value=blank)

    return y,seq_len

class MelSpec(torch.nn.Module):
    def __init__(self,dither=1e-5,pad_to=8,blank=27):
        super().__init__()
        self.dither = dither
        self.pad_to = pad_to
        self.blank = blank
    def forward(self,x,xn):
        if self.training:
            x,xn = get_melspec(x,xn,self.dither,self.pad_to,self.blank)
            # x = augspec(x.unsqueeze(1)).squeeze(1)
        else:
            x,xn = get_melspec(x,xn,0,self.pad_to,self.blank)
        return x,xn