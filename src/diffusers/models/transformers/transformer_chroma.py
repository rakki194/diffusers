# Copyright 2024 The Chroma Authors and The HuggingFace Team. All rights reserved.
# Copyright 2025 A Gay Wolf. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from ...configuration_utils import ConfigMixin, register_to_config
from ...loaders import (
    FromOriginalModelMixin,
    PeftAdapterMixin,
)  # Potentially add Chroma specific loaders
from ...utils import (
    USE_PEFT_BACKEND,
    deprecate,
    logging,
    scale_lora_layers,
    unscale_lora_layers,
)
from ...utils.torch_utils import maybe_allow_in_graph
from ..attention import (
    FeedForward,
)  # Chroma uses a specific MLP structure, might need custom
from ..attention_processor import (
    Attention,
    AttentionProcessor,
)  # Base for ChromaAttention
from ..embeddings import (
    get_timestep_embedding,
)  # Alternative for timestep_embedding if needed
from ..modeling_outputs import Transformer2DModelOutput
from ..modeling_utils import ModelMixin

# from ..normalization import AdaLayerNormContinuous, AdaLayerNormZero, AdaLayerNormZeroSingle # Chroma uses specific modulation

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


@dataclass
class ModulationOut:
    scale: torch.Tensor
    shift: torch.Tensor
    gate: Optional[torch.Tensor] = None


def timestep_embedding(
    timesteps: torch.Tensor,
    embed_dim: int,
    max_period: int = 10000,
    time_factor: float = 1000.0,
    repeat_only: bool = False,
):
    if not repeat_only:
        timesteps = timesteps * time_factor
    half = embed_dim // 2
    freqs = torch.exp(
        -math.log(max_period)
        * torch.arange(start=0, end=half, dtype=torch.float32)
        / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if embed_dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding.to(timesteps.dtype)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


class ChromaRoPE(nn.Module):
    def __init__(
        self, dim: int, theta: int = 10000, axes_dim: List[int] = [16, 56, 56]
    ):
        super().__init__()
        self.dim = (
            dim  # This is the total RoPE dimension per head, e.g., attention_head_dim
        )
        self.theta = theta
        self.axes_dim = axes_dim
        if sum(axes_dim) != dim:
            raise ValueError(
                f"Sum of axes_dim {axes_dim} ({sum(axes_dim)}) must be equal to RoPE dim {dim}"
            )

    @staticmethod
    def _math_rope(pos: torch.Tensor, dim_axis: int, theta: int) -> torch.Tensor:
        # pos: (B, L) - positional values for a single axis
        # dim_axis: RoPE dimension for THIS axis (e.g., axes_dim[i])
        # Output: (B, L, dim_axis/2, 2, 2) - freqs_cis components for this axis
        assert dim_axis % 2 == 0, "RoPE dimension for an axis must be even"
        # scale for omega calculation: (0, 2, ..., dim_axis-2) / dim_axis
        scale = (
            torch.arange(0, dim_axis, 2, dtype=torch.float64, device=pos.device)
            / dim_axis
        )
        # omega values: (dim_axis/2)
        omega = 1.0 / (theta**scale)

        # Calculate m_j = p * Omega_j
        # pos shape: (..., N_tokens) or (N_tokens)
        # omega shape: (D_axis/2)
        # out shape: (..., N_tokens, D_axis/2)
        out = torch.einsum("...n,d->...nd", pos, omega)

        # Create rotation matrix components: [cos(m_j), -sin(m_j), sin(m_j), cos(m_j)]
        # Stacked along a new dimension, then rearranged.
        # Output shape will be (..., N_tokens, D_axis/2, 2, 2)
        cos_val = torch.cos(out)
        sin_val = torch.sin(out)

        # Rotation matrix R = [[cos, -sin], [sin, cos]]
        # Stored as flat list [cos, -sin, sin, cos] then reshaped
        freq_components = torch.stack([cos_val, -sin_val, sin_val, cos_val], dim=-1)
        # Rearrange to (..., N_tokens, D_axis/2, 2, 2)
        freqs_cis_axis = rearrange(
            freq_components, "... d (i j) -> ... d i j", i=2, j=2
        )
        return freqs_cis_axis.float()

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        # ids: (B, L, N_axes) - e.g., (B, L, 3) for images, (B, L, 1) for text.
        # N_axes must match len(self.axes_dim).
        # Output: (B, 1, L, D_head_pairs, 2, 2) where D_head_pairs = self.dim / 2

        batch_size, seq_len, num_axes = ids.shape
        if num_axes != len(self.axes_dim):
            raise ValueError(
                f"Number of axes in ids ({num_axes}) does not match len(axes_dim) ({len(self.axes_dim)})"
            )

        all_freqs_cis_parts = []
        for i in range(num_axes):
            axis_ids = ids[..., i]  # (B, L)
            axis_rope_dim = self.axes_dim[i]
            if axis_rope_dim == 0:  # Skip if an axis dim is zero
                continue
            freqs_cis_part = ChromaRoPE._math_rope(
                pos=axis_ids, dim_axis=axis_rope_dim, theta=self.theta
            )  # (B, L, axis_rope_dim/2, 2, 2)
            all_freqs_cis_parts.append(freqs_cis_part)

        # Concatenate along the D_axis/2 dimension (the one representing pairs)
        # This corresponds to dim=-3 in flow's EmbedND if we consider (B, L, D_pairs, 2, 2)
        if (
            not all_freqs_cis_parts
        ):  # Handle case where all axes_dim might be zero (though unlikely)
            return torch.empty(
                batch_size, 1, seq_len, 0, 2, 2, device=ids.device, dtype=torch.float32
            )

        freqs_cis_combined = torch.cat(
            all_freqs_cis_parts, dim=-3
        )  # (B, L, D_total_pairs, 2, 2)
        # D_total_pairs = sum(axes_dim[i]/2) = self.dim / 2

        # Unsqueeze to add the head dimension for broadcasting with (B, H, L, D) Q/K tensors later.
        # Expected by flow.math.apply_rope: (B, 1, L, D_total_pairs, 2, 2)
        return freqs_cis_combined.unsqueeze(1)


def apply_rotary_pos_emb(
    xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary positional embeddings to query and key tensors.
    Args:
        xq (torch.Tensor): Query tensor with shape (B, H, L, D_head).
        xk (torch.Tensor): Key tensor with shape (B, H, L, D_head).
        freqs_cis (torch.Tensor): Precomputed cosine and sine frequencies
            with shape (B, 1, L, D_head_pairs, 2, 2) or (B, L, D_head_pairs, 2, 2)
            where D_head_pairs = D_head / 2.
            The (2,2) matrix is [[cos, -sin], [sin, cos]].
            freqs_cis[..., :, 0] gives the first column (cos, sin).
            freqs_cis[..., :, 1] gives the second column (-sin, cos).
    Returns:
        Tuple[torch.Tensor, torch.Tensor]: RoPE-applied query and key tensors.
    """
    # Reshape xq and xk to view the last dimension as pairs for complex number multiplication
    # xq_: (B, H, L, D_head/2, 2)
    xq_ = xq.float().reshape(*xq.shape[:-1], -1, 2)
    xk_ = xk.float().reshape(*xk.shape[:-1], -1, 2)

    # freqs_cis might be (B, 1, L, D_head/2, 2, 2) or (B, L, D_head/2, 2, 2)
    # We need to ensure broadcasting works. If freqs_cis has H dim, it should be 1 or match xq_.shape[1]
    # The `flow` implementation has freqs_cis as (B, 1, L, D_head/2, 2, 2), so it broadcasts over H.

    # Reshape for dot product: xq_ becomes (B, H, L, D_head/2, 1, 2)
    xq_reshaped = xq_.unsqueeze(-2)
    xk_reshaped = xk_.unsqueeze(-2)

    # Perform the rotation: R * x = [x_re, x_im] @ [[c, -s], [s, c]]^T (if x is col vec)
    # Rx = [c*x_re + s*x_im, -s*x_re + c*x_im]
    #
    # From flow.math.apply_rope:
    # xq_out = freqs_cis[..., 0] * xq_[..., 0] + freqs_cis[..., 1] * xq_[..., 1]
    # This implies xq_ is (..., D_pairs, 2) and freqs_cis is (..., D_pairs, 2, 2)
    # xq_[..., 0] is x_even, xq_[..., 1] is x_odd.
    # freqs_cis[..., 0] is the first column of R: (cos, sin)
    # freqs_cis[..., 1] is the second column of R: (-sin, cos)
    # So, (cos, sin) * x_even + (-sin, cos) * x_odd
    # = (cos*x_even - sin*x_odd, sin*x_even + cos*x_odd)
    # This is x_rot_re, x_rot_im. This is correct.

    # freqs_cis shape is (B, 1, L, D_head/2, 2, 2)
    # xq_ shape is (B, H, L, D_head/2, 2)
    # We need to align L and D_head/2. H dim on xq_ should broadcast with 1 on freqs_cis.

    # freqs_cis_col0 = freqs_cis[..., :, 0] # (B, 1, L, D_head/2, 2)
    # freqs_cis_col1 = freqs_cis[..., :, 1] # (B, 1, L, D_head/2, 2)

    # xq_even = xq_[..., 0].unsqueeze(-1) # (B, H, L, D_head/2, 1)
    # xq_odd = xq_[..., 1].unsqueeze(-1)  # (B, H, L, D_head/2, 1)

    # xq_out_pairs = freqs_cis_col0 * xq_even + freqs_cis_col1 * xq_odd # (B, H, L, D_head/2, 2)
    # xk_out_pairs = freqs_cis_col0 * xk_[..., 0].unsqueeze(-1) + freqs_cis_col1 * xk_[..., 1].unsqueeze(-1)

    # Simpler, directly using the original flow logic structure:
    # Ensure freqs_cis broadcasts correctly. flow freqs_cis is (B, 1, L, D_head_pairs, 2, 2)
    # xq_, xk_ are (B, H, L, D_head_pairs, 2)

    # xq_ [..., 0] -> (B, H, L, D_head_pairs) accessing x_even part
    # freqs_cis [..., 0] -> (B, 1, L, D_head_pairs, 2) first column (cos, sin)
    # We need (cos,sin) * x_even_scalar
    # Let's use the exact einsum or component-wise mult from a known good RoPE implementation
    # if direct translation of `flow.math.apply_rope` is tricky with current `xq_` shape.

    # Re-check flow.math.apply_rope:
    # xq_ = xq.float().reshape(*xq.shape[:-1], -1, 1, 2) -> (B, H, L, D_head/2, 1, 2)
    # freqs_cis is (B, 1, L, D_head/2, 2, 2)
    # xq_out = freqs_cis[..., 0] * xq_[..., 0] + freqs_cis[..., 1] * xq_[..., 1]
    # freqs_cis[..., 0] is (B, 1, L, D_head/2, 2) -> first column (cos, sin)
    # xq_[..., 0] is (B, H, L, D_head/2, 1) -> x_even
    # Result of product: (B, H, L, D_head/2, 2)
    # This means (cos, sin) * x_even_scalar -> (cos*x_even, sin*x_even)

    # And freqs_cis[..., 1] * xq_[..., 1] means (-sin, cos) * x_odd_scalar -> (-sin*x_odd, cos*x_odd)
    # Summing them: (cos*x_even - sin*x_odd, sin*x_even + cos*x_odd)
    # This matches the standard RoPE formula.

    # So, the key is the shapes for broadcasting.
    # xq_ shape: (B, H, L, D_head/2, 2)
    # freqs_cis shape: (B, 1, L, D_head/2, 2, 2)

    # Let x_even = xq_[..., 0] -> (B, H, L, D_head/2)
    # Let x_odd  = xq_[..., 1] -> (B, H, L, D_head/2)

    # freqs_cos = freqs_cis[..., 0, 0] -> (B, 1, L, D_head/2)
    # freqs_sin = freqs_cis[..., 1, 0] -> (B, 1, L, D_head/2) # sin from second row, first col of R
    # Actually, R = [[c, -s], [s, c]]
    # freqs_cis[..., 0, 0] is c. freqs_cis[..., 0, 1] is s. (Row vector from R^T)
    # freqs_cis[..., 1, 0] is -s. freqs_cis[..., 1, 1] is c.

    # From flow: freqs_cis[...,0] is first col (c,s). freqs_cis[...,1] is second col (-s,c)
    # xq_[...,0] is x_even scalar. xq_[...,1] is x_odd scalar.
    # out = col1 * x_even + col2 * x_odd

    q_r, q_i = xq_[..., 0], xq_[..., 1]
    k_r, k_i = xk_[..., 0], xk_[..., 1]

    # freqs_cis is (B, 1, L, D_head/2, 2, 2)
    # col0 = freqs_cis[..., :, 0] -> (B, 1, L, D_head/2, 2) i.e. (cos_vals, sin_vals)
    # col1 = freqs_cis[..., :, 1] -> (B, 1, L, D_head/2, 2) i.e. (-sin_vals, cos_vals)

    cos_comp = freqs_cis[..., 0, 0]  # (B, 1, L, D_head/2)
    sin_comp = freqs_cis[..., 1, 0]  # (B, 1, L, D_head/2)

    # x_rot_re = x_re * cos - x_im * sin
    # x_rot_im = x_re * sin + x_im * cos

    q_out_r = q_r * cos_comp - q_i * sin_comp
    q_out_i = q_r * sin_comp + q_i * cos_comp

    k_out_r = k_r * cos_comp - k_i * sin_comp
    k_out_i = k_r * sin_comp + k_i * cos_comp

    xq_out = torch.stack([q_out_r, q_out_i], dim=-1)
    xk_out = torch.stack([k_out_r, k_out_i], dim=-1)

    return xq_out.reshape(*xq.shape).type_as(xq), xk_out.reshape(*xk.shape).type_as(xk)


# Placeholder for ChromaSelfAttention
# Needs to handle QKNorm(RMSNorm) before RoPE, and shared QKV logic for DoubleStreamBlock
class ChromaSelfAttention(nn.Module):
    def __init__(
        self,
        query_dim: int,
        num_heads: int,
        head_dim: int,
        qk_norm: bool = True,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = head_dim**-0.5

        self.q_proj = nn.Linear(query_dim, num_heads * head_dim, bias=True)
        self.k_proj = nn.Linear(query_dim, num_heads * head_dim, bias=True)
        self.v_proj = nn.Linear(query_dim, num_heads * head_dim, bias=True)
        self.out_proj = nn.Linear(num_heads * head_dim, query_dim, bias=True)

        self.qk_norm = qk_norm
        if self.qk_norm:
            self.norm_q = RMSNorm(head_dim, eps=eps)
            self.norm_k = RMSNorm(head_dim, eps=eps)

    def forward(self, x_q, x_k, x_v, mask=None):  # Removed RoPE specific args for now
        q = self.q_proj(x_q)
        k = self.k_proj(x_k)
        v = self.v_proj(x_v)

        q = rearrange(q, "b l (h d) -> b h l d", h=self.num_heads)
        k = rearrange(k, "b l (h d) -> b h l d", h=self.num_heads)
        v = rearrange(v, "b l (h d) -> b h l d", h=self.num_heads)

        # QKNorm is applied by the block before RoPE
        # RoPE is applied by the block
        # This method now expects Q, K that are already normed and RoPE'd if needed by caller.
        # OR, the block calls q_proj, k_proj, then norms, then RoPEs, then uses these Q,K for einsum.
        # For now, assume Q,K passed in are ready for einsum (after potential RoPE by caller)

        attn_scores = torch.einsum("b h i d, b h j d -> b h i j", q, k) * self.scale
        if mask is not None:
            attn_scores = attn_scores + mask.unsqueeze(1)

        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_output = torch.einsum("b h i j, b h j d -> b h i d", attn_probs, v)
        attn_output = rearrange(attn_output, "b h l d -> b l (h d)")

        return self.out_proj(attn_output)


# Placeholders for MLPEmbedder, Approximator, Blocks, LastLayer, and Main Model
class ChromaMLPEmbedder(nn.Module):
    def __init__(
        self, in_dim, hidden_dim, out_dim, act=nn.SiLU()
    ):  # flow uses SiLU in MLPEmbedder
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), act, nn.Linear(hidden_dim, out_dim)
        )
        self.norm = RMSNorm(
            in_dim
        )  # flow.Approximator.MLPEmbedder uses RMSNorm before MLP

    def forward(self, x):  # flow.Approximator uses residual: x = x + layer(norm(x))
        # The nn.Sequential in flow.Approximator's MLPEmbedder IS the layer(norm(x)) part.
        # The residual is in the loop of Approximator.
        # So, this should be: norm -> mlp
        return self.mlp(self.norm(x))


class ChromaApproximator(nn.Module):
    def __init__(self, config):  # config will be ChromaTransformer2DConfig
        super().__init__()
        self.config = config
        self.in_dim_approximator = sum(config.approximator_in_dim_feature_splits)
        self.hidden_size_approximator = config.approximator_hidden_size
        self.out_dim_approximator = (
            config.num_attention_heads * config.attention_head_dim
        )  # model hidden_size

        self.in_proj = nn.Linear(
            self.in_dim_approximator, self.hidden_size_approximator, bias=True
        )

        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()  # Paired norms for each MLPEmbedder layer
        for _ in range(config.approximator_depth):
            # In flow.Approximator: layers.append(MLPEmbedder(hidden_dim, hidden_dim))
            # norms.append(RMSNorm(hidden_dim))
            # And forward is: x = x + layer(norm(x))
            # So, ChromaMLPEmbedder should take hidden_dim as input AND output.
            self.layers.append(
                ChromaMLPEmbedder(
                    self.hidden_size_approximator,
                    self.hidden_size_approximator,
                    self.hidden_size_approximator,
                )
            )  # flow.MLPEmbedder does in->hidden, hidden->hidden
            # It seems ChromaMLPEmbedder is just the MLP part after norm.
        # Revisit ChromaMLPEmbedder and flow.Approximator.forward:
        # flow.Approximator:
        # self.layers = nn.ModuleList([MLPEmbedder(hidden_dim, hidden_dim) for x in range(n_layers)])
        # self.norms = nn.ModuleList([RMSNorm(hidden_dim) for x in range(n_layers)])
        # forward: x = self.in_proj(x); for layer, norm in zip(self.layers, self.norms): x = x + layer(norm(x))
        # And flow.MLPEmbedder is: Linear(in,hid) -> SiLU -> Linear(hid,hid) (NO NORM INSIDE)
        # So our ChromaMLPEmbedder was wrong. It should not have a norm.
        # The norm is applied BEFORE calling the MLPEmbedder in Approximator's loop.

        # Corrected Approximator structure:
        self.layers_mlp = (
            nn.ModuleList()
        )  # Renamed to avoid confusion with self.layers in this scope
        for _ in range(config.approximator_depth):
            # flow.MLPEmbedder: in_layer(in, hid), silu, out_layer(hid, hid)
            # Here, input to MLPEmbedder is hidden_size_approximator
            self.layers_mlp.append(
                nn.Sequential(
                    nn.Linear(
                        self.hidden_size_approximator, self.hidden_size_approximator
                    ),
                    nn.SiLU(),
                    nn.Linear(
                        self.hidden_size_approximator, self.hidden_size_approximator
                    ),
                )
            )
        self.norms_approx = nn.ModuleList(
            [
                RMSNorm(self.hidden_size_approximator)
                for _ in range(config.approximator_depth)
            ]
        )
        self.out_proj = nn.Linear(
            self.hidden_size_approximator, self.out_dim_approximator
        )

    def forward(self, timestep, guidance, mod_indices_tensor_placeholder=None):
        # mod_indices_tensor_placeholder is not used, fixed_mod_indices are generated internally

        ts_embed_dim, guidance_embed_dim, mod_idx_embed_dim_cfg = (
            self.config.approximator_in_dim_feature_splits
        )

        distill_timestep = timestep_embedding(timestep, ts_embed_dim, time_factor=1.0)
        distill_guidance = timestep_embedding(
            guidance, guidance_embed_dim, time_factor=1.0
        )

        batch_size = timestep.shape[0]
        mod_len = self.config.mod_index_length

        fixed_mod_indices = torch.arange(
            self.config.mod_index_length, device=timestep.device
        )
        modulation_index_embedded = timestep_embedding(
            fixed_mod_indices, mod_idx_embed_dim_cfg, time_factor=1.0
        )
        modulation_index_batched = modulation_index_embedded.unsqueeze(0).repeat(
            batch_size, 1, 1
        )

        timestep_guidance_embed = torch.cat([distill_timestep, distill_guidance], dim=1)
        timestep_guidance_embed_repeated = timestep_guidance_embed.unsqueeze(1).repeat(
            1, mod_len, 1
        )

        x = torch.cat(
            [timestep_guidance_embed_repeated, modulation_index_batched], dim=-1
        )

        # Approximator MLP structure from flow.module.layers.Approximator
        x = self.in_proj(x)
        for layer_mlp_module, norm_module in zip(self.layers_mlp, self.norms_approx):
            x = x + layer_mlp_module(
                norm_module(x)
            )  # Residual connection around norm -> mlp_block
        x = self.out_proj(x)

        return x


class ChromaDoubleStreamBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        dim = config.num_attention_heads * config.attention_head_dim
        num_heads = config.num_attention_heads
        head_dim = config.attention_head_dim
        # mlp_ratio from config, defaulting if not present
        mlp_ratio = getattr(config, "mlp_ratio", 4.0)
        mlp_hidden_dim = int(dim * mlp_ratio)
        qkv_bias = getattr(config, "qkv_bias", True)

        # Image Stream Components (matching flow.DoubleStreamBlock)
        self.img_norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        # flow.SelfAttention: qkv=Linear, norm=QKNorm(RMS), proj=Linear
        # Our ChromaSelfAttention has q_proj, k_proj, v_proj, out_proj and norm_q, norm_k (RMS)
        # We need to make sure this maps correctly.
        # flow.DoubleStreamBlock creates img_qkv, then img_q,k,v, then norms q,k with SelfAttention.norm
        self.img_attn_qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.img_attn_qknorm = RMSNorm(
            head_dim, eps=1e-6
        )  # Assuming QKNorm applies to each head separately
        # flow.QKNorm is RMSNorm(dim) but applied to Q,K of shape (B,H,L,D_head)
        # So it should be RMSNorm(head_dim)
        self.img_attn_proj = nn.Linear(dim, dim)

        self.img_norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.img_mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim, bias=True),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_hidden_dim, dim, bias=True),
        )

        # Text Stream Components
        self.txt_norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.txt_attn_qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.txt_attn_qknorm = RMSNorm(head_dim, eps=1e-6)
        self.txt_attn_proj = nn.Linear(dim, dim)

        self.txt_norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.txt_mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim, bias=True),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_hidden_dim, dim, bias=True),
        )

        # Shared attention output projection (qkv derived from streams, normed, RoPE'd, then scaled_dot_product)
        # The shared attention in flow doesn't have its own QKV projections. It uses Q,K,V from streams.
        # It only needs an output projection.
        self.shared_out_proj = nn.Linear(
            dim, dim
        )  # Corresponds to self.img_attn.proj or self.txt_attn.proj in flow
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = head_dim**-0.5

    def _modulate(self, x, mod: ModulationOut):
        return (1 + mod.scale.squeeze(1)) * x + mod.shift.squeeze(
            1
        )  # Squeeze B,1,D to B,D for LayerNorm compat if needed

    def _apply_norm_to_q_or_k(self, q_or_k_tensor: torch.Tensor, norm_layer: RMSNorm):
        # q_or_k_tensor: (B, H, L, D_head)
        # norm_layer: RMSNorm(D_head)
        b, h, l, d = q_or_k_tensor.shape
        orig_dtype = q_or_k_tensor.dtype
        # Reshape for norm: (B*H*L, D_head)
        reshaped_tensor = q_or_k_tensor.reshape(-1, d)
        normed_tensor = norm_layer(reshaped_tensor)
        return normed_tensor.reshape(b, h, l, d).to(orig_dtype)

    def forward(self, img_embeds, txt_embeds, pe_freqs_cis, mod_dict_entry, mask):
        img_mod_attn, img_mod_mlp = mod_dict_entry["img_mod.lin"]
        txt_mod_attn, txt_mod_mlp = mod_dict_entry["txt_mod.lin"]

        # Image Path - QKV generation and Norm
        img_norm1_out = self.img_norm1(img_embeds)
        img_modulated_attn = self._modulate(img_norm1_out, img_mod_attn)
        img_qkv_proj = self.img_attn_qkv(img_modulated_attn)
        img_q, img_k, img_v = rearrange(
            img_qkv_proj,
            "b l (k h d) -> k b h l d",
            k=3,
            h=self.num_heads,
            d=self.head_dim,
        ).unbind(0)
        img_q = self._apply_norm_to_q_or_k(img_q, self.img_attn_qknorm)
        img_k = self._apply_norm_to_q_or_k(img_k, self.img_attn_qknorm)

        # Text Path - QKV generation and Norm
        txt_norm1_out = self.txt_norm1(txt_embeds)
        txt_modulated_attn = self._modulate(txt_norm1_out, txt_mod_attn)
        txt_qkv_proj = self.txt_attn_qkv(txt_modulated_attn)
        txt_q, txt_k, txt_v = rearrange(
            txt_qkv_proj,
            "b l (k h d) -> k b h l d",
            k=3,
            h=self.num_heads,
            d=self.head_dim,
        ).unbind(0)
        txt_q = self._apply_norm_to_q_or_k(txt_q, self.txt_attn_qknorm)
        txt_k = self._apply_norm_to_q_or_k(txt_k, self.txt_attn_qknorm)

        # Shared Attention
        q_shared = torch.cat((txt_q, img_q), dim=2)
        k_shared = torch.cat((txt_k, img_k), dim=2)
        v_shared = torch.cat((txt_v, img_v), dim=2)

        q_shared_rope, k_shared_rope = apply_rotary_pos_emb(
            q_shared, k_shared, pe_freqs_cis
        )

        # mask is (B, L_all, L_all), needs to be (B, H, L_all, L_all) or (B, 1, L_all, L_all) for SDPA
        # flow math.attention takes mask (B, H, L, D) - actually (B,H,L_q, L_k)
        # The mask from model.forward is (B, H, L_all, L_all)
        # My current txt_img_mask is (B, L_all, L_all) - need to unsqueeze for heads
        sdpa_mask = (
            mask.unsqueeze(1) if mask is not None else None
        )  # (B, 1, L_all, L_all)

        attn_shared_out_raw = F.scaled_dot_product_attention(
            q_shared_rope,
            k_shared_rope,
            v_shared,
            attn_mask=sdpa_mask,
            scale=self.scale,
        )  # Use self.scale if not built into SDPA
        # Default SDPA scale is (1/sqrt(d_k)). If custom scale is needed, it should be passed.
        # Flow's self.scale is head_dim**-0.5. Default SDPA uses this.

        attn_shared_out_flat = rearrange(attn_shared_out_raw, "b h l d -> b l (h d)")
        attn_shared_projected = self.shared_out_proj(attn_shared_out_flat)

        L_txt = txt_embeds.shape[1]
        txt_attn_res = attn_shared_projected[:, :L_txt, :]
        img_attn_res = attn_shared_projected[:, L_txt:, :]

        # Image Path (continued)
        img_embeds = img_embeds + img_mod_attn.gate.squeeze(1) * img_attn_res
        img_norm2_out = self.img_norm2(img_embeds)
        img_modulated_mlp = self._modulate(img_norm2_out, img_mod_mlp)
        img_mlp_processed = self.img_mlp(img_modulated_mlp)
        img_embeds = img_embeds + img_mod_mlp.gate.squeeze(1) * img_mlp_processed

        # Text Path (continued)
        txt_embeds = txt_embeds + txt_mod_attn.gate.squeeze(1) * txt_attn_res
        txt_norm2_out = self.txt_norm2(txt_embeds)
        txt_modulated_mlp = self._modulate(txt_norm2_out, txt_mod_mlp)
        txt_mlp_processed = self.txt_mlp(txt_modulated_mlp)
        txt_embeds = txt_embeds + txt_mod_mlp.gate.squeeze(1) * txt_mlp_processed

        return img_embeds, txt_embeds


class ChromaSingleStreamBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        dim = config.num_attention_heads * config.attention_head_dim
        num_heads = config.num_attention_heads
        head_dim = config.attention_head_dim
        mlp_ratio = getattr(config, "mlp_ratio", 4.0)
        mlp_hidden_dim = int(dim * mlp_ratio)
        # qkv_bias for linear1? flow.SingleStreamBlock.linear1 does not show bias param
        # but nn.Linear defaults to bias=True. Let's assume True for now.

        self.pre_norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.linear1 = nn.Linear(dim, dim * 3 + mlp_hidden_dim, bias=True)

        self.qk_norm = RMSNorm(
            head_dim, eps=1e-6
        )  # flow.SingleStreamBlock.norm is QKNorm(head_dim)
        # so it applies to Q and K separately.

        self.mlp_act = nn.GELU(approximate="tanh")
        self.linear2 = nn.Linear(dim + mlp_hidden_dim, dim, bias=True)

        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = head_dim**-0.5

    def _modulate_pre_norm(
        self, x, mod: ModulationOut
    ):  # Only scale and shift for pre_norm
        return (1 + mod.scale.squeeze(1)) * x + mod.shift.squeeze(1)

    def _apply_norm_to_q_or_k(self, q_or_k_tensor: torch.Tensor, norm_layer: RMSNorm):
        b, h, l, d = q_or_k_tensor.shape
        orig_dtype = q_or_k_tensor.dtype
        reshaped_tensor = q_or_k_tensor.reshape(-1, d)
        normed_tensor = norm_layer(reshaped_tensor)
        return normed_tensor.reshape(b, h, l, d).to(orig_dtype)

    def forward(self, x, pe_freqs_cis, mod_single: ModulationOut, mask):
        residual = x

        norm_out = self.pre_norm(x)
        # mod_single has scale, shift, gate. Squeeze the (B,1,D) to (B,D) if needed by LayerNorm
        x_mod = self._modulate_pre_norm(norm_out, mod_single)

        qkv_fused, mlp_in_val = torch.split(
            self.linear1(x_mod),
            [
                3 * self.config.num_attention_heads * self.config.attention_head_dim,
                int(
                    self.config.num_attention_heads
                    * self.config.attention_head_dim
                    * getattr(self.config, "mlp_ratio", 4.0)
                ),
            ],
            dim=-1,
        )

        q, k, v = rearrange(
            qkv_fused,
            "b l (three h d) -> three b h l d",
            three=3,
            h=self.num_heads,
            d=self.head_dim,
        ).unbind(dim=0)

        q_norm = self._apply_norm_to_q_or_k(q, self.qk_norm)
        k_norm = self._apply_norm_to_q_or_k(k, self.qk_norm)

        q_rope, k_rope = apply_rotary_pos_emb(q_norm, k_norm, pe_freqs_cis)

        sdpa_mask = mask.unsqueeze(1) if mask is not None else None
        attn_out_raw = F.scaled_dot_product_attention(
            q_rope, k_rope, v, attn_mask=sdpa_mask, scale=self.scale
        )
        attn_out = rearrange(attn_out_raw, "b h l d -> b l (h d)")

        mlp_act_out = self.mlp_act(mlp_in_val)

        combined_features = torch.cat((attn_out, mlp_act_out), dim=-1)
        block_output_val = self.linear2(combined_features)

        x = residual + mod_single.gate.squeeze(1) * block_output_val  # Squeeze gate mod
        return x


class ChromaLastLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        hidden_size = config.num_attention_heads * config.attention_head_dim
        out_channels = (
            config.out_channels
        )  # This is model's direct output channels (e.g., 64)

        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(
            hidden_size, out_channels, bias=True
        )  # flow.LastLayer has bias=True for linear

    def _modulate(self, x, shift, scale):  # No gate for LastLayer mod
        # shift, scale are (B, 1, D), need to be (B, D) or broadcastable with x (B,L,D)
        return (1 + scale.squeeze(1)) * x + shift.squeeze(1)

    def forward(self, x, mod_final_list: List[torch.Tensor]):
        shift, scale = mod_final_list[0], mod_final_list[1]

        norm_out = self.norm_final(x)
        x_mod = self._modulate(norm_out, shift, scale)
        return self.linear(x_mod)


# Main ChromaTransformer2DModel
class ChromaTransformer2DModel(
    ModelMixin, ConfigMixin, PeftAdapterMixin, FromOriginalModelMixin
):
    _supports_gradient_checkpointing = True
    _no_split_modules = ["ChromaDoubleStreamBlock", "ChromaSingleStreamBlock"]
    _skip_layerwise_casting_patterns = [
        "pe_embedder",
        "norm_final",
        "img_norm1",
        "img_norm2",
        "txt_norm1",
        "txt_norm2",
        "pre_norm",
    ]

    @register_to_config
    def __init__(
        self,
        patch_size: int = 2,  # From FLUX paper for VAE, Chroma uses this logic
        in_channels: int = 64,
        out_channels: Optional[int] = 64,
        num_layers: int = 19,
        num_single_layers: int = 38,
        attention_head_dim: int = 128,
        num_attention_heads: int = 24,
        joint_attention_dim: int = 4096,
        guidance_embeds: bool = True,
        axes_dims_rope: Tuple[int, ...] = (16, 56, 56),  # Made it Tuple[int, ...]
        mod_index_length: int = 344,
        approximator_in_dim_feature_splits: Tuple[int, int, int] = (16, 16, 32),
        approximator_depth: int = 5,
        approximator_hidden_size: int = 5120,
        rope_theta: int = 10000,
        qkv_bias: bool = True,
        mlp_ratio: float = 4.0,
        # pooled_projection_dim is not in flow.ChromaParams, so removed from __init__
        # It was part of FluxTransformer2DModel for CLIP pooled embeds.
    ):
        super().__init__()
        self.hidden_size = num_attention_heads * attention_head_dim
        self.patch_size_vae = patch_size  # VAE patch size, e.g. 2 for FLUX VAE
        # Model's "patch_size" in LastLayer is 1 as per Chroma analysis

        # self.in_channels_model: input channels to self.img_in (e.g., 64 for VAE B,16,H,W -> B, L, 16*2*2=64)
        # self.out_channels_model: output channels from self.final_layer (e.g., 64 for patched VAE latents)
        self.in_channels_model = in_channels
        self.out_channels_model = out_channels or self.in_channels_model

        self.img_in = nn.Linear(self.in_channels_model, self.hidden_size, bias=qkv_bias)
        self.txt_in = nn.Linear(joint_attention_dim, self.hidden_size, bias=qkv_bias)

        self.pe_embedder = ChromaRoPE(
            dim=attention_head_dim, theta=rope_theta, axes_dim=list(axes_dims_rope)
        )  # Ensure axes_dim is list

        self.distilled_guidance_layer = ChromaApproximator(self.config)

        self.double_blocks = nn.ModuleList(
            [ChromaDoubleStreamBlock(self.config) for _ in range(num_layers)]
        )

        self.single_blocks = nn.ModuleList(
            [ChromaSingleStreamBlock(self.config) for _ in range(num_single_layers)]
        )

        # Pass self.config to LastLayer, it will pick out_channels from there.
        # The LastLayer's "patch_size" argument in flow is 1, meaning it doesn't do unpatching.
        # Its output is (B, L_img, model_out_channels)
        self.final_layer = ChromaLastLayer(self.config)

        self.gradient_checkpointing = False

    def _get_mod_map(self) -> Tuple[Dict[str, int], int]:  # Return total count as well
        mod_map = {}
        idx_count = 0

        # Double Stream Blocks
        # Each mod (img_mod, txt_mod) has 2 sets (attn, mlp) of (s,s,g) = 3 vectors each
        # So, one call to mod_vectors_dict[f"double_blocks.{i}.img_mod.lin"] returns a list of 2 ModulationOut objects.
        # Each ModulationOut(s,s,g) consumes 3 vectors from the flat mod_vectors list.
        # Total for one stream (img or txt) in one DoubleStreamBlock is 2 * 3 = 6 vectors.
        for i in range(self.config.num_layers):
            mod_map[f"double_blocks.{i}.img_mod.lin"] = idx_count
            idx_count += 2 * 3
            mod_map[f"double_blocks.{i}.txt_mod.lin"] = idx_count
            idx_count += 2 * 3

        # Single Stream Blocks
        # Each block has 1 set of (s,s,g) = 3 vectors.
        for i in range(self.config.num_single_layers):
            mod_map[f"single_blocks.{i}.modulation.lin"] = idx_count
            idx_count += 1 * 3

        # Final Layer
        # Needs 1 set of (s,s) - no gate = 2 vectors.
        mod_map[f"final_layer.adaLN_modulation.1"] = idx_count
        idx_count += 1 * 2

        if idx_count != self.config.mod_index_length:
            # This check is also in flow.Chroma model, it seems the ComfyUI node might have a different mod_index_length (e.g. 344 vs 212 for lite)
            # For the standard model, this should match.
            logger.info(  # Changed to info as this might be intentional for lite models
                f"Calculated mod_index_length {idx_count} might not match config {self.config.mod_index_length}. "
                f"This might be expected for 'lite' model versions. Ensure correct mod_index_length in config."
            )
        return mod_map, idx_count

    def _distribute_modulations(
        self, mod_vectors: torch.Tensor, mod_map: Dict[str, int]
    ) -> Dict[str, Any]:
        distributed_mod_dict = {}
        # hidden_size = self.hidden_size # Not needed here, mod_vectors already have hidden_size dim

        for key, start_idx in mod_map.items():
            if "double_blocks" in key:
                s1, sh1, g1 = (
                    mod_vectors[:, start_idx, :],
                    mod_vectors[:, start_idx + 1, :],
                    mod_vectors[:, start_idx + 2, :],
                )
                s2, sh2, g2 = (
                    mod_vectors[:, start_idx + 3, :],
                    mod_vectors[:, start_idx + 4, :],
                    mod_vectors[:, start_idx + 5, :],
                )
                distributed_mod_dict[key] = [
                    ModulationOut(scale=s1, shift=sh1, gate=g1),
                    ModulationOut(scale=s2, shift=sh2, gate=g2),
                ]
            elif "single_blocks" in key:
                s, sh, g = (
                    mod_vectors[:, start_idx, :],
                    mod_vectors[:, start_idx + 1, :],
                    mod_vectors[:, start_idx + 2, :],
                )
                distributed_mod_dict[key] = ModulationOut(scale=s, shift=sh, gate=g)

            elif "final_layer" in key:
                sh, s = mod_vectors[:, start_idx, :], mod_vectors[:, start_idx + 1, :]
                distributed_mod_dict[key] = [sh, s]

        return distributed_mod_dict

    def _get_text_attention_mask(
        self,
        attention_mask: torch.Tensor,
        num_tokens_to_unmask: int,
        device: torch.device,
    ) -> torch.Tensor:
        # attention_mask: (B, L_txt) from tokenizer (1 for attend, 0 for pad)
        # Returns a mask for SDPA: (B, L_txt, L_txt) where 0 is attend, -inf is no_attend.
        # Implements roughly modify_mask_to_attend_padding logic.
        batch_size, max_seq_len_txt = attention_mask.shape

        # Create a base mask for SDPA: (0 for attend, -inf for no attend)
        # Start with self-attention fully allowed, then restrict based on padding.
        # For each item in batch, find its actual length.
        actual_lengths = attention_mask.sum(dim=1)  # (B,)

        output_mask = torch.zeros(
            batch_size, max_seq_len_txt, max_seq_len_txt, device=device
        )

        for i in range(batch_size):
            current_len = int(actual_lengths[i].item())

            # Unmask a few padding tokens if available and requested
            unmask_end = current_len
            if num_tokens_to_unmask > 0 and current_len < max_seq_len_txt:
                unmask_end = min(current_len + num_tokens_to_unmask, max_seq_len_txt)

            # Tokens from unmask_end to max_seq_len_txt are hard masked
            if unmask_end < max_seq_len_txt:
                output_mask[i, :, unmask_end:] = -float(
                    "inf"
                )  # Query cannot attend to these
                output_mask[i, unmask_end:, :] = -float(
                    "inf"
                )  # These tokens cannot attend query

            # Ensure that a token only attends up to unmask_end of other tokens
            # This creates a causal-like mask up to unmask_end for padded part
            # For true tokens (0 to current_len-1), they can attend each other fully
            # and up to unmask_end.
            # A simpler approach: for each query token `q`, if a key token `k` is beyond `unmask_end` for that batch item, mask it.
            # This might be too restrictive. Flow's modify_mask_to_attend_padding is simpler:
            # it just flips a few 0s to 1s in the original (B,L) mask.
            # Then it does: txt_img_mask = txt_mask_w_padding.float().T @ txt_mask_w_padding.float()
            # This creates a (L,L) mask based on which tokens *can* be attended.

            # Replicating flow's mask logic:
            # 1. Modify original (B,L) text mask
            modified_src_mask = attention_mask[i, :].clone()  # (L_txt)
            if num_tokens_to_unmask > 0 and current_len < max_seq_len_txt:
                can_unmask = min(num_tokens_to_unmask, max_seq_len_txt - current_len)
                modified_src_mask[current_len : current_len + can_unmask] = 1

            # 2. Create square mask from this modified (L_txt) mask
            # (L_q, L_k) where entry (q,k) is 1 if token q can attend to token k.
            # flow does: m.float().T @ m.float(). Then converts to bool.
            # This means M_qk = m_q * m_k. If m_k is 0 (padded), M_qk is 0.
            m = modified_src_mask.float()
            square_mask_logic = m.unsqueeze(1) * m.unsqueeze(0)  # (L_txt, L_txt)

            # Convert to SDPA format (-inf for no attend)
            output_mask[i] = torch.where(square_mask_logic.bool(), 0.0, -float("inf"))

        return output_mask

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        timestep: torch.LongTensor,
        img_ids: torch.Tensor,
        txt_ids: torch.Tensor,
        guidance: torch.Tensor,
        attention_mask: Optional[
            torch.Tensor
        ] = None,  # Text mask (B, L_txt) from tokenizer (1=attend, 0=pad)
        num_tokens_to_unmask_pad: int = 8,  # Corresponds to attn_padding in flow.Chroma.forward
        return_dict: bool = True,
    ) -> Union[torch.Tensor, Transformer2DModelOutput]:

        batch_size, seq_len_img, _ = hidden_states.shape
        seq_len_txt = encoder_hidden_states.shape[1]

        img_embeds = self.img_in(hidden_states)
        txt_embeds = self.txt_in(encoder_hidden_states)

        # Approximator - flow model uses torch.no_grad() context here.
        # For basic inference/usage in Diffusers, we typically don't manage grad contexts inside the model.
        # If this is critical for loading pretrained weights that expect no grad through approximator
        # during their PFP, it might need consideration for fine-tuning.
        # For now, standard forward pass.
        with torch.set_grad_enabled(self.training):  # Replicate no_grad if not training
            mod_vectors = self.distilled_guidance_layer(timestep, guidance)

        mod_map, _ = self._get_mod_map()
        mod_vectors_dict = self._distribute_modulations(mod_vectors, mod_map)

        if txt_ids.shape[-1] < img_ids.shape[-1]:
            padding = torch.zeros(
                batch_size,
                seq_len_txt,
                img_ids.shape[-1] - txt_ids.shape[-1],
                device=txt_ids.device,
                dtype=txt_ids.dtype,
            )
            txt_ids_padded = torch.cat((txt_ids, padding), dim=-1)
        elif (
            img_ids.shape[-1] < txt_ids.shape[-1]
        ):  # Handle case where img_ids might be simpler
            padding = torch.zeros(
                batch_size,
                seq_len_img,
                txt_ids.shape[-1] - img_ids.shape[-1],
                device=img_ids.device,
                dtype=img_ids.dtype,
            )
            img_ids_padded = torch.cat((img_ids, padding), dim=-1)
            ids_for_rope = torch.cat((txt_ids, img_ids_padded), dim=1)
        else:
            txt_ids_padded = txt_ids
            ids_for_rope = torch.cat((txt_ids_padded, img_ids), dim=1)

        pe_freqs_cis = self.pe_embedder(
            ids_for_rope
        )  # (B, 1, L_all, D_head_pairs, 2, 2)

        # Attention Mask for shared attention
        # txt_img_mask should be (B, L_all, L_all) for SDPA, with -inf for masked.
        # Or (B, H, L_all, L_all) if heads have different masks. Flow uses (B,H,L,L).
        # Let's assume mask is broadcast across heads for now: (B, 1, L_all, L_all).

        # Default: allow all attention if no text mask provided
        seq_len_all = seq_len_txt + seq_len_img
        txt_img_mask_for_sdpa = torch.zeros(
            batch_size, 1, seq_len_all, seq_len_all, device=img_embeds.device
        )

        if attention_mask is not None:  # tokenizer mask (B, L_txt), 1=attend, 0=pad
            # Get text-only square mask (B, L_txt, L_txt) with SDPA format
            text_square_sdpa_mask = self._get_text_attention_mask(
                attention_mask, num_tokens_to_unmask_pad, img_embeds.device
            )

            # Combine for full sequence for SDPA: (B, 1, L_all, L_all)
            # Top-left: text_square_sdpa_mask
            # Others: 0 (attend)
            txt_img_mask_for_sdpa[:, 0, :seq_len_txt, :seq_len_txt] = (
                text_square_sdpa_mask
            )
            # Image tokens can attend to all text tokens that are not hard-padded (respecting text_square_sdpa_mask's effect on text)
            # Text tokens can attend to all image tokens.
            # Image tokens can attend to all other image tokens.
            # This simpler setup means: if a text token K_txt is masked for Txt_Q to attend, it's also masked for Img_Q to attend.
            # More precise would be:
            # M_tt = text_square_sdpa_mask
            # M_ti = 0
            # M_it = mask reflecting which text tokens image can attend to (e.g. based on modified_src_mask)
            # M_ii = 0
            # For flow's M_qk = m_q * m_k logic, where m is the (L_all) combined mask vector:

            modified_txt_mask_flat = attention_mask.clone()  # (B, L_txt)
            for i in range(batch_size):
                current_len = int(attention_mask[i].sum().item())
                if num_tokens_to_unmask_pad > 0 and current_len < seq_len_txt:
                    can_unmask = min(
                        num_tokens_to_unmask_pad, seq_len_txt - current_len
                    )
                    modified_txt_mask_flat[
                        i, current_len : current_len + can_unmask
                    ] = 1

            img_mask_flat = torch.ones(
                batch_size,
                seq_len_img,
                device=img_embeds.device,
                dtype=modified_txt_mask_flat.dtype,
            )
            combined_m_vector = torch.cat(
                [modified_txt_mask_flat, img_mask_flat], dim=1
            )  # (B, L_all)

            # M_qk = m_q * m_k
            # Mask is (B, L_q, L_k). Entry (b,q,k) is 0 if attend, -inf if not.
            l_all = seq_len_all
            flow_style_mask = torch.zeros(
                batch_size, l_all, l_all, device=img_embeds.device
            )
            for b_idx in range(batch_size):
                m_b = combined_m_vector[b_idx].float()  # (L_all)
                square_logic = m_b.unsqueeze(1) * m_b.unsqueeze(0)  # (L_all, L_all)
                flow_style_mask[b_idx] = torch.where(
                    square_logic.bool(), 0.0, -float("inf")
                )
            txt_img_mask_for_sdpa = flow_style_mask.unsqueeze(1)  # (B, 1, L_all, L_all)

        for i, block in enumerate(self.double_blocks):
            current_mod_entry = {
                "img_mod.lin": mod_vectors_dict[f"double_blocks.{i}.img_mod.lin"],
                "txt_mod.lin": mod_vectors_dict[f"double_blocks.{i}.txt_mod.lin"],
            }
            if self.gradient_checkpointing and self.training:
                img_embeds, txt_embeds = torch.utils.checkpoint.checkpoint(
                    block,
                    img_embeds,
                    txt_embeds,
                    pe_freqs_cis,
                    current_mod_entry,
                    txt_img_mask_for_sdpa,
                    use_reentrant=False,
                )
            else:
                img_embeds, txt_embeds = block(
                    img_embeds,
                    txt_embeds,
                    pe_freqs_cis=pe_freqs_cis,
                    mod_dict_entry=current_mod_entry,
                    mask=txt_img_mask_for_sdpa,
                )

        merged_embeds = torch.cat((txt_embeds, img_embeds), dim=1)

        for i, block in enumerate(self.single_blocks):
            single_mod = mod_vectors_dict[f"single_blocks.{i}.modulation.lin"]
            if self.gradient_checkpointing and self.training:
                merged_embeds = torch.utils.checkpoint.checkpoint(
                    block,
                    merged_embeds,
                    pe_freqs_cis,
                    single_mod,
                    txt_img_mask_for_sdpa,
                    use_reentrant=False,
                )
            else:
                merged_embeds = block(
                    x=merged_embeds,
                    pe_freqs_cis=pe_freqs_cis,
                    mod_single=single_mod,
                    mask=txt_img_mask_for_sdpa,
                )

        img_output_embeds = merged_embeds[:, seq_len_txt:, :]

        final_mod_params = mod_vectors_dict["final_layer.adaLN_modulation.1"]
        output = self.final_layer(img_output_embeds, final_mod_params)

        if not return_dict:
            return (output,)

        return Transformer2DModelOutput(sample=output)
