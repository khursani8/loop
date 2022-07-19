import math
from torch import nn
import torch
from .subsampling import StackingSubsampling, ConvSubsampling,TimeReductionModule
from .mha import *
from .squeezeformer_layer import *

class SqueezeformerEncoder(nn.Module):
    """Some Information about SqueezeformerEncoder"""
    def __init__(self,cfg):
        super(SqueezeformerEncoder, self).__init__()
        d_model = cfg.d_model
        ff_expansion_factor = cfg.ff_expansion_factor
        d_ff = d_model * ff_expansion_factor
        self.d_model = d_model
        att_context_size = cfg.att_context_size
        xscaling = cfg.xscaling
        adaptive_scale = cfg.adaptive_scale
        time_reduce_idx = cfg.time_reduce_idx
        time_recovery_idx = cfg.time_recovery_idx
        n_layers = cfg.n_layers
        subsampling_conv_channels = cfg.subsampling_conv_channels
        subsampling = cfg.subsampling
        subsampling_factor = cfg.subsampling_factor
        self_attention_model = cfg.self_attention_model
        n_heads = cfg.n_heads
        dropout = cfg.dropout
        pos_emb_max_len = cfg.pos_emb_max_len
        feat_out = cfg.feat_out

        self._feat_in = cfg.feat_in
        self.scale = math.sqrt(self.d_model)
        if att_context_size:
            self.att_context_size = att_context_size
        else:
            self.att_context_size = [-1, -1]

        if xscaling:
            self.xscale = math.sqrt(d_model)
        else:
            self.xscale = None
        self.adaptive_scale = adaptive_scale

        self.time_reduce_idx = time_reduce_idx
        if time_reduce_idx is not None:
            if time_recovery_idx is None:
                self.time_recovery_idx = n_layers - 1  # recover at last layer
            else:
                self.time_recovery_idx = time_recovery_idx  # recover at given layer

        if self.time_reduce_idx is not None:
            if self.time_reduce_idx < 0 or self.time_recovery_idx >= n_layers:
                raise ValueError(f"Time reduce index must lie between [0, {n_layers})")
            if self.time_recovery_idx < 0 or self.time_recovery_idx >= n_layers:
                raise ValueError(f"Time recovery index must lie between [0, {n_layers})")

        if subsampling_conv_channels == -1:
            subsampling_conv_channels = d_model
        if subsampling and subsampling_factor > 1:
            if subsampling == 'stacking':
                self.pre_encode = StackingSubsampling(
                    subsampling_factor=subsampling_factor, feat_in=self._feat_in, feat_out=d_model
                )
            else:
                self.pre_encode = ConvSubsampling(
                    subsampling=subsampling,
                    subsampling_factor=subsampling_factor,
                    feat_in=self._feat_in,
                    feat_out=d_model,
                    conv_channels=subsampling_conv_channels,
                    activation=nn.ReLU(),
                )
                # For Squeezeformer, initialize the parameters as required.
                self.pre_encode.reset_parameters()
        else:
            self.pre_encode = nn.Linear(self._feat_in, d_model)

        self._feat_out = d_model

        if not cfg.untie_biases and self_attention_model == "rel_pos":
            d_head = d_model // n_heads
            pos_bias_u = nn.Parameter(torch.Tensor(n_heads, d_head))
            pos_bias_v = nn.Parameter(torch.Tensor(n_heads, d_head))
            nn.init.zeros_(pos_bias_u)
            nn.init.zeros_(pos_bias_v)
        else:
            pos_bias_u = None
            pos_bias_v = None

        self.pos_emb_max_len = pos_emb_max_len
        if self_attention_model == "rel_pos":
            self.pos_enc = RelPositionalEncoding(
                d_model=d_model,
                dropout_rate=dropout,
                max_len=pos_emb_max_len,
                xscale=self.xscale,
                dropout_rate_emb=cfg.dropout_emb,
            )
        elif self_attention_model == "abs_pos":
            pos_bias_u = None
            pos_bias_v = None
            self.pos_enc = PositionalEncoding(
                d_model=d_model, dropout_rate=dropout, max_len=pos_emb_max_len, xscale=self.xscale
            )
        else:
            raise ValueError(f"Not valid self_attention_model: '{self_attention_model}'!")

        self.layers = nn.ModuleList()
        for i in range(n_layers):
            layer = SqueezeformerLayer(
                d_model=d_model,
                d_ff=d_ff,
                self_attention_model=self_attention_model,
                n_heads=n_heads,
                conv_kernel_size=cfg.conv_kernel_size,
                conv_norm_type=cfg.conv_norm_type,
                dropout=dropout,
                dropout_att=cfg.dropout_att,
                pos_bias_u=pos_bias_u,
                pos_bias_v=pos_bias_v,
                adaptive_scale=adaptive_scale,
            )
            self.layers.append(layer)

        # Time Reduction and Recovery layer setup
        self.time_reduce_layer = None
        self.time_recovery_layer = None
        self.time_reduce_pos_enc = None
        # Add time reduction layer
        if self.time_reduce_idx is not None:
            self.time_reduce_layer = TimeReductionModule(d_model, d_model, kernel_size=5, stride=2)
            self.time_recovery_layer = nn.Linear(d_model, d_model)

            # Chose same type of positional encoding as the originally determined above
            if self_attention_model == "rel_pos":
                self.time_reduce_pos_enc = RelPositionalEncoding(
                    d_model=d_model, dropout_rate=0.0, max_len=pos_emb_max_len, xscale=None, dropout_rate_emb=0.0,
                )
            else:
                self.time_reduce_pos_enc = PositionalEncoding(
                    d_model=d_model, dropout_rate=0.0, max_len=pos_emb_max_len, xscale=None, dropout_rate_emb=0.0
                )

        self.pre_ln = nn.LayerNorm(d_model)

        if feat_out > 0 and feat_out != self._feat_out:
            self.out_proj = nn.Linear(self._feat_out, feat_out)
            self._feat_out = feat_out
        else:
            self.out_proj = None
            self._feat_out = d_model
        self.set_max_audio_length(self.pos_emb_max_len)
        self.use_pad_mask = True
    def forward(self, audio_signal,length=None):
        self.update_max_seq_length(seq_length=audio_signal.size(2), device=audio_signal.device)
        return self.forward_for_export(audio_signal=audio_signal, length=length)
    def forward_for_export(self, audio_signal, length):
        max_audio_length: int = audio_signal.size(-1)

        if max_audio_length > self.max_audio_length:
            self.set_max_audio_length(max_audio_length)

        if length is None:
            length = audio_signal.new_full(
                audio_signal.size(0), max_audio_length, dtype=torch.int32, device=self.seq_range.device
            )

        audio_signal = torch.transpose(audio_signal, 1, 2)

        if isinstance(self.pre_encode, nn.Linear):
            audio_signal = self.pre_encode(audio_signal)
        else:
            audio_signal, length = self.pre_encode(audio_signal, length)

        audio_signal, pos_emb = self.pos_enc(audio_signal)
        # adjust size
        max_audio_length = audio_signal.size(1)
        # Create the self-attention and padding masks

        pad_mask = self.make_pad_mask(max_audio_length, length)
        att_mask = pad_mask.unsqueeze(1).repeat([1, max_audio_length, 1])
        att_mask = torch.logical_and(att_mask, att_mask.transpose(1, 2))
        if self.att_context_size[0] >= 0:
            att_mask = att_mask.triu(diagonal=-self.att_context_size[0])
        if self.att_context_size[1] >= 0:
            att_mask = att_mask.tril(diagonal=self.att_context_size[1])
        att_mask = ~att_mask

        if self.use_pad_mask:
            pad_mask = ~pad_mask
        else:
            pad_mask = None

        # Create cache of activations for the time reduction step
        # Note: NeMo codebase allows only a single time reduction step to occur
        recovery_activation_cache = []

        audio_signal = self.pre_ln(audio_signal)
        for lth, layer in enumerate(self.layers):
            # Perform time reduction
            if self.time_reduce_layer is not None and lth == self.time_reduce_idx:
                # Perform time reduction
                recovery_activation_cache.append((audio_signal, att_mask, pad_mask, pos_emb))
                audio_signal, att_mask, pad_mask = self.time_reduce_layer(
                    x=audio_signal, att_mask=att_mask, pad_mask=pad_mask
                )
                # Only update PE, not the original audio_signal
                _, pos_emb = self.time_reduce_pos_enc(audio_signal)

            # Perform time recovery
            if self.time_recovery_layer is not None and lth == self.time_recovery_idx:
                recovery_audio_signal, att_mask, pad_mask, pos_emb = recovery_activation_cache.pop(0)
                # repeat interleaved values for 2x seq length
                audio_signal = torch.repeat_interleave(audio_signal, repeats=2, dim=1)

                B, T, D = recovery_audio_signal.size()
                audio_signal = audio_signal[:, :T, :]  # Slice off the exact T timesteps as original cache value
                audio_signal = self.time_recovery_layer(audio_signal)  # learn non linear mapping
                audio_signal = recovery_audio_signal + audio_signal  # learn just the residual

            audio_signal = layer(x=audio_signal, att_mask=att_mask, pos_emb=pos_emb, pad_mask=pad_mask)

        if self.out_proj is not None:
            audio_signal = self.out_proj(audio_signal)

        audio_signal = torch.transpose(audio_signal, 1, 2)
        return audio_signal, length
    def update_max_seq_length(self, seq_length: int, device):
        # Find global max audio length across all nodes
        if torch.distributed.is_initialized():
            global_max_len = torch.tensor([seq_length], dtype=torch.float32, device=device)

            # Update across all ranks in the distributed system
            torch.distributed.all_reduce(global_max_len, op=torch.distributed.ReduceOp.MAX)

            seq_length = global_max_len.int().item()

        if seq_length > self.max_audio_length:
            self.set_max_audio_length(seq_length)
    def set_max_audio_length(self, max_audio_length):
        """ Sets maximum input length.
            Pre-calculates internal seq_range mask.
        """
        self.max_audio_length = max_audio_length
        device = next(self.parameters()).device
        seq_range = torch.arange(0, self.max_audio_length, device=device)
        if hasattr(self, 'seq_range'):
            self.seq_range = seq_range
        else:
            self.register_buffer('seq_range', seq_range, persistent=False)
        self.pos_enc.extend_pe(max_audio_length, device)

        if self.time_reduce_pos_enc is not None:
            self.time_reduce_pos_enc.extend_pe(max_audio_length, device)
    def make_pad_mask(self, max_audio_length, seq_lens):
        """Make masking for padding."""
        mask = self.seq_range[:max_audio_length].expand(seq_lens.size(0), -1) < seq_lens.unsqueeze(-1)
        return mask

    def enable_pad_mask(self, on=True):
        # On inference, user may chose to disable pad mask
        mask = self.use_pad_mask
        self.use_pad_mask = on
        return mask