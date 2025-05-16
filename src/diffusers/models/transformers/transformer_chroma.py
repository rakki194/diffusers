# Copyright 2024 The Chroma Authors and The HuggingFace Team. All rights reserved.
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


# Placeholder for RoPE, to be adapted from flow.models.chroma.math and module.layers.EmbedND
# Needs to handle concatenation of text and image ids, and axes_dim
class ChromaRoPE(nn.Module):
    def __init__(
        self, dim: int, theta: int = 10000, axes_dim: List[int] = [16, 56, 56]
    ):
        super().__init__()
        self.dim = dim
        self.theta = theta
        self.axes_dim = axes_dim
        # Ensure sum of axes_dim elements equals dim
        if sum(axes_dim) != dim:
            raise ValueError(
                f"Sum of axes_dim {axes_dim} must be equal to RoPE dim {dim}"
            )

    def forward(self, x: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
        # x shape: B, L, H, D_head or B, H, L, D_head
        # ids shape: B, L, N_axes (e.g., N_axes=3 for [id_type, y_pos, x_pos] or N_axes=1 for [seq_idx])
        # This is a simplified placeholder. The actual implementation needs to replicate
        # flow.models.chroma.math.rope and flow.models.chroma.math.apply_rope,
        # handling different numbers of axes in `ids` for text vs image.
        # It precomputes frequencies based on `ids` and `axes_dim`.
        # For now, returns x unchanged.
        logger.warning_once(
            "ChromaRoPE is a placeholder and does not apply rotary embeddings yet."
        )
        return x

    def _compute_freqs_cis(self, ids: torch.Tensor, n_elem: int) -> torch.Tensor:
        # Placeholder for actual frequency computation based on `ids` for each axis
        # and `self.theta`.
        # `ids` has shape (B, L, num_axes)
        # `n_elem` is `self.dim` (total RoPE dimensions for a head)
        # It should return `freqs_cis` with shape (B, L, 1, n_elem) or similar for broadcasting.
        # This is complex as it needs to iterate over axes_dim and apply rope logic for each part.
        batch_size, seq_len, _ = ids.shape
        freqs_cis = torch.zeros(
            batch_size, seq_len, 1, n_elem, device=ids.device, dtype=torch.float32
        )
        # Example: freqs_cis[..., 0:n_elem//2] = cos_terms
        #          freqs_cis[..., n_elem//2:n_elem] = sin_terms
        # This would be (cos(m*theta_i), sin(m*theta_i)) pairs essentially
        # For now, returns identity rotation (no change)
        cos_val = torch.ones_like(freqs_cis[..., 0 : n_elem // 2])
        sin_val = torch.zeros_like(freqs_cis[..., n_elem // 2 : n_elem])
        freqs_cis = torch.cat(
            (cos_val, sin_val), dim=-1
        )  # Placeholder for actual freqs
        return freqs_cis.to(
            x.dtype
        )  # This should be shaped correctly for apply_rotary_pos_emb


def apply_rotary_pos_emb(q, k, freqs_cis):
    # Simplified from flow.models.chroma.math.apply_rope
    # q, k: (B, H, L, D) or (B, L, H, D)
    # freqs_cis: (B, L, 1, D) - needs to be broadcastable

    # Transpose to (B, L, H, D) if necessary
    q_transformed = (
        q.transpose(1, 2) if q.ndim == 4 and q.shape[1] != freqs_cis.shape[1] else q
    )
    k_transformed = (
        k.transpose(1, 2) if k.ndim == 4 and k.shape[1] != freqs_cis.shape[1] else k
    )

    q_embed = (q_transformed * freqs_cis.cos()) + (
        rotate_half(q_transformed) * freqs_cis.sin()
    )
    k_embed = (k_transformed * freqs_cis.cos()) + (
        rotate_half(k_transformed) * freqs_cis.sin()
    )

    # Transpose back if needed
    q_embed = (
        q_embed.transpose(1, 2)
        if q.ndim == 4 and q.shape[1] != freqs_cis.shape[1]
        else q_embed
    )
    k_embed = (
        k_embed.transpose(1, 2)
        if k.ndim == 4 and k.shape[1] != freqs_cis.shape[1]
        else k_embed
    )
    return q_embed, k_embed


def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


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

        # RoPE will be applied externally if this attention is used within a block
        # that handles RoPE for its inputs

    def forward(
        self,
        x_q,
        x_k,
        x_v,
        rope_embedder: Optional[ChromaRoPE] = None,
        pe_ids: Optional[torch.Tensor] = None,
        mask=None,
    ):
        # x_q, x_k, x_v: (B, L, C)
        # pe_ids: (B, L, N_axes) for RoPE

        q = self.q_proj(x_q)
        k = self.k_proj(x_k)
        v = self.v_proj(x_v)

        q = rearrange(q, "b l (h d) -> b h l d", h=self.num_heads)
        k = rearrange(k, "b l (h d) -> b h l d", h=self.num_heads)
        v = rearrange(v, "b l (h d) -> b h l d", h=self.num_heads)

        if self.qk_norm:
            q_orig_dtype = q.dtype
            k_orig_dtype = k.dtype
            q = (
                self.norm_q(q.reshape(-1, self.head_dim))
                .reshape(q.shape)
                .to(q_orig_dtype)
            )
            k = (
                self.norm_k(k.reshape(-1, self.head_dim))
                .reshape(k.shape)
                .to(k_orig_dtype)
            )

        if rope_embedder is not None and pe_ids is not None:
            # This assumes rope_embedder.forward directly applies RoPE
            # The more accurate flow would be for the block to precompute freqs_cis
            # and pass them here for apply_rotary_pos_emb.
            # For now, let's assume a simplified path or external RoPE application.
            # freqs_cis = rope_embedder._compute_freqs_cis(pe_ids, self.head_dim)
            # q, k = apply_rotary_pos_emb(q, k, freqs_cis)
            pass  # RoPE application deferred or handled by the block

        attn_scores = torch.einsum("b h i d, b h j d -> b h i j", q, k) * self.scale
        if mask is not None:
            # Mask is (B, L_txt_img, L_txt_img) or similar
            # Needs broadcasting to (B, H, L, L)
            attn_scores = attn_scores + mask.unsqueeze(1)

        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_output = torch.einsum("b h i j, b h j d -> b h i d", attn_probs, v)
        attn_output = rearrange(attn_output, "b h l d -> b l (h d)")

        return self.out_proj(attn_output)


# Placeholders for MLPEmbedder, Approximator, Blocks, LastLayer, and Main Model
class ChromaMLPEmbedder(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, act=nn.SiLU()):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), act, nn.Linear(hidden_dim, out_dim)
        )
        self.norm = RMSNorm(in_dim)  # Chroma Approximator uses RMSNorm

    def forward(self, x):
        return x + self.mlp(self.norm(x))


class ChromaApproximator(nn.Module):
    def __init__(self, config):  # config will be ChromaTransformer2DConfig
        super().__init__()
        self.config = config
        self.in_dim = sum(config.approximator_in_dim_feature_splits)
        self.out_dim = (
            config.num_attention_heads * config.attention_head_dim
        )  # hidden_size

        layers = [nn.Linear(self.in_dim, config.approximator_hidden_size)]
        for _ in range(config.approximator_depth):
            layers.append(
                ChromaMLPEmbedder(
                    config.approximator_hidden_size,
                    config.approximator_hidden_size,
                    config.approximator_hidden_size,
                )
            )
        layers.append(nn.Linear(config.approximator_hidden_size, self.out_dim))
        self.mlp = nn.Sequential(*layers)

        # For mod_index embeddings
        self.mod_idx_embed_dim = config.approximator_in_dim_feature_splits[2]
        # self.mod_index_embedding = nn.Embedding(config.mod_index_length, self.mod_idx_embed_dim)
        # No, Chroma uses timestep_embedding for mod_index too.

    def forward(self, timestep, guidance, mod_indices_tensor):
        # timestep, guidance: (B,)
        # mod_indices_tensor: (B, mod_index_length) or (mod_index_length) - needs broadcasting for batch

        ts_embed_dim, guidance_embed_dim, mod_idx_embed_dim_cfg = (
            self.config.approximator_in_dim_feature_splits
        )

        distill_timestep = timestep_embedding(
            timestep, ts_embed_dim, time_factor=1.0
        )  # time_factor is 1.0 in ComfyUI_FluxMod
        distill_guidance = timestep_embedding(
            guidance, guidance_embed_dim, time_factor=1.0
        )

        # mod_indices_tensor should be (B, mod_index_length)
        # each row is torch.arange(mod_index_length)
        # We need to embed each index in mod_indices_tensor

        # modulation_index_embedded = self.mod_index_embedding(mod_indices_tensor) # If using nn.Embedding
        # For timestep_embedding approach:
        # mod_indices_tensor is (B, L_mod)
        # We need to embed each of the L_mod indices.
        # timestep_embedding expects (N,) or (N,1)

        batch_size = timestep.shape[0]
        mod_len = self.config.mod_index_length

        # Ensure mod_indices_tensor is (B, mod_len)
        if mod_indices_tensor.ndim == 1:
            mod_indices_tensor = mod_indices_tensor.unsqueeze(0).repeat(batch_size, 1)

        # Embed mod_indices for each item in batch
        # This is a bit tricky with timestep_embedding directly.
        # Let's assume mod_indices_tensor is pre-embedded or handled externally for now
        # Or we process it sequentially, which is inefficient.
        # The original code embeds torch.arange(344) once and then combines.

        # Simplified: Assume mod_indices_tensor is already the (B, mod_len, mod_idx_embed_dim)
        # This part needs careful implementation based on flow/ComfyUI

        # Correct approach from chroma_analysis.md & flow:
        # 1. Embed self.mod_index = torch.arange(mod_index_length) -> (mod_index_length, mod_idx_embed_dim)
        # 2. Batch it: (B, mod_index_length, mod_idx_embed_dim)
        # 3. Combine with batched timestep & guidance embeds

        # This is the fixed sequence of indices to embed
        fixed_mod_indices = torch.arange(
            self.config.mod_index_length, device=timestep.device
        )
        modulation_index_embedded = timestep_embedding(
            fixed_mod_indices, mod_idx_embed_dim_cfg, time_factor=1.0
        )  # (mod_len, D_mod_idx)
        modulation_index_batched = modulation_index_embedded.unsqueeze(0).repeat(
            batch_size, 1, 1
        )  # (B, mod_len, D_mod_idx)

        # distill_timestep (B, D_ts), distill_guidance (B, D_guidance)
        # Need to be (B, mod_len, D_ts/D_guidance)
        timestep_guidance_embed = torch.cat(
            [distill_timestep, distill_guidance], dim=1
        )  # (B, D_ts + D_guidance)
        timestep_guidance_embed_repeated = timestep_guidance_embed.unsqueeze(1).repeat(
            1, mod_len, 1
        )  # (B, mod_len, D_ts+D_guidance)

        input_vec = torch.cat(
            [timestep_guidance_embed_repeated, modulation_index_batched], dim=-1
        )  # (B, mod_len, D_in_approx)

        # input_vec.requires_grad_(True) in original, but this is handled by torch.enable_grad if needed
        # Original also uses torch.no_grad() for the Approximator pass in Chroma.forward, this is important.

        return self.mlp(input_vec)  # (B, mod_len, hidden_size)


class ChromaDoubleStreamBlock(nn.Module):
    def __init__(self, config):  # config will be ChromaTransformer2DConfig
        super().__init__()
        self.config = config
        dim = config.num_attention_heads * config.attention_head_dim  # hidden_size
        num_heads = config.num_attention_heads
        head_dim = config.attention_head_dim
        mlp_ratio = 4.0  # As per chroma_params, not in config directly but standard

        self.img_norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.img_attn = ChromaSelfAttention(dim, num_heads, head_dim, qk_norm=True)
        self.img_norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.img_mlp = FeedForward(
            dim, dim_out=dim, activation_fn="gelu-approximate", final_dropout=False
        )  # Or custom MLP

        self.txt_norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.txt_attn = ChromaSelfAttention(dim, num_heads, head_dim, qk_norm=True)
        self.txt_norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.txt_mlp = FeedForward(
            dim, dim_out=dim, activation_fn="gelu-approximate", final_dropout=False
        )

        # For shared attention, QK Norm is inside ChromaSelfAttention, RoPE is applied externally
        self.shared_attn = ChromaSelfAttention(
            dim, num_heads, head_dim, qk_norm=True
        )  # QK Norm applied to its inputs

        # RoPE needs to be instantiated at the model level and freqs_cis passed in
        # Or, apply RoPE inside the block before shared_attn if pe_ids is available here.
        # The analysis suggests RoPE frequencies `pe` are computed once and passed around.

    def _modulate(self, x, mod: ModulationOut):
        return (1 + mod.scale) * x + mod.shift

    def forward(self, img_embeds, txt_embeds, pe_freqs_cis, mod_dict_entry, mask):
        # mod_dict_entry: dict like {"img_mod": [ModulationOutAttn, ModulationOutMLP], "txt_mod": ...}
        # pe_freqs_cis: Precomputed RoPE frequencies for combined text+image sequence
        # mask: Attention mask for combined text+image sequence

        img_mod_attn, img_mod_mlp = mod_dict_entry["img_mod.lin"]  # As per Chroma keys
        txt_mod_attn, txt_mod_mlp = mod_dict_entry["txt_mod.lin"]

        # Image Path - Initial Self-Attention (stores QKV)
        img_norm_out = self._modulate(self.img_norm1(img_embeds), img_mod_attn)
        img_q = self.img_attn.q_proj(img_norm_out)
        img_k = self.img_attn.k_proj(img_norm_out)
        img_v = self.img_attn.v_proj(img_norm_out)
        # Reshape QKV for shared attention: B, H, L_img, D_head
        img_q = rearrange(img_q, "b l (h d) -> b h l d", h=self.img_attn.num_heads)
        img_k = rearrange(img_k, "b l (h d) -> b h l d", h=self.img_attn.num_heads)
        img_v = rearrange(img_v, "b l (h d) -> b h l d", h=self.img_attn.num_heads)

        # Text Path - Initial Self-Attention (stores QKV)
        txt_norm_out = self._modulate(self.txt_norm1(txt_embeds), txt_mod_attn)
        txt_q = self.txt_attn.q_proj(txt_norm_out)
        txt_k = self.txt_attn.k_proj(txt_norm_out)
        txt_v = self.txt_attn.v_proj(txt_norm_out)
        # Reshape QKV for shared attention: B, H, L_txt, D_head
        txt_q = rearrange(txt_q, "b l (h d) -> b h l d", h=self.txt_attn.num_heads)
        txt_k = rearrange(txt_k, "b l (h d) -> b h l d", h=self.txt_attn.num_heads)
        txt_v = rearrange(txt_v, "b l (h d) -> b h l d", h=self.txt_attn.num_heads)

        # Prepare for Shared Attention
        # The `pe_freqs_cis` will have parts for text and image
        # Let L_txt = txt_q.shape[2], L_img = img_q.shape[2]
        # pe_freqs_cis_txt = pe_freqs_cis[:, :L_txt, :, :] -> shape needs to match Q/K for apply_rotary_pos_emb
        # pe_freqs_cis_img = pe_freqs_cis[:, L_txt:, :, :]

        # Concatenate Qs, Ks, Vs for shared attention
        # Qs/Ks need RoPE. V does not.
        # QKNorm is applied *inside* ChromaSelfAttention based on current simplified impl.
        # RoPE should be applied *after* QKNorm.

        # Apply QKNorm (if not done inside shared_attn.q_proj etc.)
        # If ChromaSelfAttention's q_proj etc are just nn.Linear, then norm first.
        # Current ChromaSelfAttention applies QKNorm internally *after* projection from modulated norm_out.
        # This seems to match flow: norm_out = (1+scale)*norm(x)+shift, then this norm_out to QKV proj.
        # Then Q, K are QKNormed (RMSNorm), then RoPE.

        # This part is complex due to RoPE and QKNorm order.
        # Assuming `shared_attn` projects its own QKV from concatenated inputs.
        # This is NOT how Chroma works. Chroma shares the QKVs from individual stream attentions.

        # Corrected flow for shared attention:
        # 1. QKNorm the img_q, img_k, txt_q, txt_k (reshaped to B*L, H*D then normed, then reshaped back)
        #    OR QKNorm within ChromaSelfAttention is per-head (B,H,L,D) -> (B,H,L,D_normed)
        #    The analysis says: QKNorm (RMSNorm for Q, RMSNorm for K) is applied to q_shared, k_shared.
        #    This means after cat.

        q_shared = torch.cat((txt_q, img_q), dim=2)  # B, H, L_txt+L_img, D_head
        k_shared = torch.cat((txt_k, img_k), dim=2)  # B, H, L_txt+L_img, D_head
        v_shared = torch.cat((txt_v, img_v), dim=2)  # B, H, L_txt+L_img, D_head

        # Apply QKNorm to q_shared, k_shared (per-head)
        q_shared_norm = self.shared_attn.norm_q(
            q_shared.reshape(-1, self.shared_attn.head_dim)
        ).reshape(q_shared.shape)
        k_shared_norm = self.shared_attn.norm_k(
            k_shared.reshape(-1, self.shared_attn.head_dim)
        ).reshape(k_shared.shape)

        # Apply RoPE to q_shared_norm, k_shared_norm using pe_freqs_cis
        # pe_freqs_cis needs to be (B, L_txt+L_img, 1, D_head) or similar for broadcasting with (B,H,L,D)
        # Let's assume pe_freqs_cis is already shaped (B, H, L_txt+L_img, D_head) for element-wise ops
        # Or (B, 1, L_txt+L_img, D_head) if heads share RoPE freqs for the same token
        # Chroma analysis: `pe` based on `ids` (B, L, N_axes), `pe` applied within attention.
        # `apply_rope(xq, xk, freqs_cis)` takes (B,H,L,D) and freqs_cis.
        # So, `pe_freqs_cis` should be compatible.

        q_shared_rope, k_shared_rope = apply_rotary_pos_emb(
            q_shared_norm, k_shared_norm, pe_freqs_cis
        )  # pe_freqs_cis needs careful shaping

        # Shared Attention computation (using components of shared_attn for projection)
        attn_scores = (
            torch.einsum("b h i d, b h j d -> b h i j", q_shared_rope, k_shared_rope)
            * self.shared_attn.scale
        )
        if mask is not None:
            attn_scores = attn_scores + mask.unsqueeze(1)  # mask (B, L_all, L_all)

        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_shared_out_raw = torch.einsum(
            "b h i j, b h j d -> b h i d", attn_probs, v_shared
        )  # B, H, L_all, D_head
        attn_shared_out_flat = rearrange(
            attn_shared_out_raw, "b h l d -> b l (h d)"
        )  # B, L_all, C

        attn_shared_out = self.shared_attn.out_proj(attn_shared_out_flat)  # B, L_all, C

        # Split shared attention output
        L_txt = txt_embeds.shape[1]
        txt_attn_out = attn_shared_out[:, :L_txt, :]
        img_attn_out = attn_shared_out[:, L_txt:, :]

        # Image Path (continued)
        img_embeds = img_embeds + img_mod_attn.gate * img_attn_out  # Gated residual
        img_norm_mlp_out = self._modulate(self.img_norm2(img_embeds), img_mod_mlp)
        img_mlp_processed = self.img_mlp(img_norm_mlp_out)
        img_embeds = img_embeds + img_mod_mlp.gate * img_mlp_processed

        # Text Path (continued)
        txt_embeds = txt_embeds + txt_mod_attn.gate * txt_attn_out  # Gated residual
        txt_norm_mlp_out = self._modulate(self.txt_norm2(txt_embeds), txt_mod_mlp)
        txt_mlp_processed = self.txt_mlp(txt_norm_mlp_out)
        txt_embeds = txt_embeds + txt_mod_mlp.gate * txt_mlp_processed

        return img_embeds, txt_embeds


class ChromaSingleStreamBlock(nn.Module):
    def __init__(self, config):  # config will be ChromaTransformer2DConfig
        super().__init__()
        self.config = config
        dim = config.num_attention_heads * config.attention_head_dim
        num_heads = config.num_attention_heads
        head_dim = config.attention_head_dim
        mlp_ratio = 4.0
        mlp_hidden_dim = int(dim * mlp_ratio)

        self.pre_norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)

        # Projects to QKV and MLP input simultaneously
        self.linear1 = nn.Linear(dim, dim * 3 + mlp_hidden_dim)

        self.qk_norm_q = RMSNorm(head_dim, eps=1e-6)
        self.qk_norm_k = RMSNorm(head_dim, eps=1e-6)

        self.mlp_act = nn.GELU(approximate="tanh")  # Matches Chroma's GELU
        self.linear2 = nn.Linear(dim + mlp_hidden_dim, dim)

        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = head_dim**-0.5

    def _modulate(self, x, mod: ModulationOut):  # Only scale and shift for pre_norm
        return (1 + mod.scale) * x + mod.shift

    def forward(self, x, pe_freqs_cis, mod_single: ModulationOut, mask):
        # mod_single: ModulationOut object for this block (scale, shift, gate)
        # pe_freqs_cis: RoPE frequencies for the merged sequence
        # mask: Attention mask for the merged sequence

        residual = x

        norm_out = self.pre_norm(x)
        x_mod = self._modulate(
            norm_out, mod_single
        )  # Apply scale and shift part of modulation

        # Get QKV and MLP input
        qkv_mlp_fused = self.linear1(x_mod)

        qkv_val = qkv_mlp_fused[
            :,
            :,
            : -(
                self.config.num_attention_heads
                * self.config.attention_head_dim
                * 4
                // 4
            ),
        ]  # Mistake here, mlp_hidden_dim not head_dim
        dim = self.config.num_attention_heads * self.config.attention_head_dim
        mlp_hidden_dim = int(dim * 4.0)

        qkv_val = qkv_mlp_fused[:, :, :-mlp_hidden_dim]
        mlp_in_val = qkv_mlp_fused[:, :, -mlp_hidden_dim:]

        q, k, v = rearrange(
            qkv_val,
            "b l (three h d) -> three b h l d",
            three=3,
            h=self.num_heads,
            d=self.head_dim,
        ).unbind(dim=0)

        # QKNorm
        q_norm = self.qk_norm_q(q.reshape(-1, self.head_dim)).reshape(q.shape)
        k_norm = self.qk_norm_k(k.reshape(-1, self.head_dim)).reshape(k.shape)

        # RoPE
        q_rope, k_rope = apply_rotary_pos_emb(q_norm, k_norm, pe_freqs_cis)

        # Attention
        attn_scores = (
            torch.einsum("b h i d, b h j d -> b h i j", q_rope, k_rope) * self.scale
        )
        if mask is not None:
            attn_scores = attn_scores + mask.unsqueeze(1)
        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_out_raw = torch.einsum("b h i j, b h j d -> b h i d", attn_probs, v)
        attn_out = rearrange(attn_out_raw, "b h l d -> b l (h d)")  # B, L, C

        # MLP
        mlp_act_out = self.mlp_act(mlp_in_val)

        combined_features = torch.cat((attn_out, mlp_act_out), dim=-1)
        block_output_val = self.linear2(combined_features)

        # Gated Residual
        x = residual + mod_single.gate * block_output_val
        return x


class ChromaLastLayer(nn.Module):
    def __init__(self, config):  # ChromaTransformer2DConfig
        super().__init__()
        self.config = config
        hidden_size = config.num_attention_heads * config.attention_head_dim
        # patch_size = config.patch_size # Not used as per analysis if out_channels implies direct output
        out_channels = config.out_channels

        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        # Linear projects to patch_size * patch_size * out_channels, but Chroma analysis implies 1*1*64
        # The `patch_size` in `proj_out` of FluxTransformer2DModel seems to be for unpatching.
        # Chroma's LastLayer linear projects to 1*1*out_channels.
        self.linear = nn.Linear(hidden_size, out_channels)

    def _modulate(self, x, shift, scale):  # No gate for LastLayer mod
        return (1 + scale) * x + shift

    def forward(self, x, mod_final_list: List[torch.Tensor]):
        # mod_final_list: [shift_tensor, scale_tensor]
        shift, scale = mod_final_list[0], mod_final_list[1]

        norm_out = self.norm_final(x)
        x_mod = self._modulate(norm_out, shift, scale)
        return self.linear(x_mod)


# Main ChromaTransformer2DModel
class ChromaTransformer2DModel(
    ModelMixin, ConfigMixin, PeftAdapterMixin, FromOriginalModelMixin
):
    _supports_gradient_checkpointing = True
    # _no_split_modules = ["ChromaDoubleStreamBlock", "ChromaSingleStreamBlock"] # Add if checkpointing these
    _skip_layerwise_casting_patterns = ["pos_embed.rope", "norm"]  # Example

    @register_to_config
    def __init__(
        self,
        patch_size: int = 1,  # VAE patch size (e.g., 2 for FLUX VAE)
        in_channels: int = 64,  # VAE latent channels * patch_size * patch_size (e.g., 16*2*2 = 64 for FLUX)
        out_channels: Optional[
            int
        ] = None,  # Output VAE latent channels (e.g. 16 for FLUX VAE, so output of model is 16*2*2=64)
        # The config `out_channels` is for the model's direct output before unpatching.
        num_layers: int = 19,  # DoubleStreamBlocks
        num_single_layers: int = 38,  # SingleStreamBlocks
        attention_head_dim: int = 128,
        num_attention_heads: int = 24,
        joint_attention_dim: int = 4096,  # T5 text embedding dim (context_in_dim)
        # pooled_projection_dim: int = 768, # Not directly used if Chroma doesn't use CLIP pooled text like FLUX pipeline
        guidance_embeds: bool = True,  # Should always be true for Chroma
        axes_dims_rope: Tuple[int] = (16, 56, 56),
        mod_index_length: int = 344,
        approximator_in_dim_feature_splits: Tuple[int, int, int] = (
            16,
            16,
            32,
        ),  # (ts, guidance, mod_idx_embed)
        approximator_depth: int = 5,
        approximator_hidden_size: int = 5120,
        # Additional params from chroma_params if needed (e.g., RoPE theta, MLP ratio if not hardcoded)
        rope_theta: int = 10000,
        qkv_bias: bool = True,  # Chroma uses bias in its QKV projections
        mlp_ratio: float = 4.0,  # For MLPs in blocks
    ):
        super().__init__()
        self.hidden_size = num_attention_heads * attention_head_dim  # e.g. 3072
        self.patch_size = patch_size  # This is VAE patch size, e.g. 2
        self.in_channels_model = in_channels  # This is post-patching, e.g. 16*2*2=64
        self.out_channels_model = (
            out_channels or self.in_channels_model
        )  # Model's direct output dim before unpatching

        # Diffusers standard way for input projection for patched inputs
        # self.patch_embed = PatchEmbed(patch_size=patch_size, in_channels=...)
        # Chroma does this differently: VAE latents are already patched by rearrange.
        # Then img_in projects from C_vae*ph*pw (e.g. 64) to hidden_size

        self.img_in = nn.Linear(self.in_channels_model, self.hidden_size, bias=qkv_bias)
        self.txt_in = nn.Linear(joint_attention_dim, self.hidden_size, bias=qkv_bias)

        # Positional Embedder (RoPE)
        # The dim for RoPE is attention_head_dim per head.
        # axes_dims_rope sum should be attention_head_dim
        self.rope_dim_per_head = sum(axes_dims_rope)
        if self.rope_dim_per_head != attention_head_dim:
            logger.warning(
                f"Sum of axes_dims_rope {axes_dims_rope} ({self.rope_dim_per_head}) does not match attention_head_dim ({attention_head_dim}). Adjusting RoPE dim."
            )
            # This indicates a mismatch in understanding or config. Chroma analysis: axes_dim sum = hidden_size / num_heads
        self.pe_embedder = ChromaRoPE(
            dim=attention_head_dim, theta=rope_theta, axes_dim=axes_dims_rope
        )

        # Approximator (Distilled Guidance Layer)
        self.distilled_guidance_layer = ChromaApproximator(
            self.config
        )  # Pass the whole config object

        # Double Stream Blocks
        self.double_blocks = nn.ModuleList(
            [ChromaDoubleStreamBlock(self.config) for _ in range(num_layers)]
        )

        # Single Stream Blocks
        self.single_blocks = nn.ModuleList(
            [ChromaSingleStreamBlock(self.config) for _ in range(num_single_layers)]
        )

        # Final Layer
        # The output of the last single_block is (B, L_img, hidden_size)
        # LastLayer in Chroma takes hidden_size and outputs 1*1*VAE_latent_channels (e.g. 64 if in_channels_model was 64)
        # The config.out_channels should be the VAE latent channels (e.g. 16) if pipeline handles unpatching,
        # OR it's model's direct channel output (e.g. 64) if model outputs patched latents.
        # Based on chroma_analysis, LastLayer outputs 64 channels (direct latent channels).
        # So, config.out_channels should be this (e.g. 64). The pipeline then handles unpatching.
        self.final_layer = ChromaLastLayer(self.config)  # Pass config to it.

        self.gradient_checkpointing = False

    def _get_mod_map(self) -> Dict[str, int]:
        # This map defines how many modulation vectors (shift, scale, gate sets) each layer type needs.
        # And then it's used by distribute_modulations.
        # From ComfyUI_FluxMod model.py FluxMod.mod_index_offsets
        # DoubleStreamBlock: img_mod (attn+mlp) = 2 sets (s,s,g); txt_mod (attn+mlp) = 2 sets. Total 4.
        # SingleStreamBlock: 1 set (s,s,g).
        # LastLayer: 1 set (s,s) - no gate.
        # Each set is one unit in mod_index_length.
        # A set (s,s,g) means 3*hidden_size params. A set (s,s) means 2*hidden_size params.
        # The Approximator outputs (mod_index_length, hidden_size).
        # distribute_modulations then splits these (mod_index_length) vectors.
        # E.g., for (s,s,g), it takes 3 vectors from Approximator output.

        # This map should match the hardcoded keys in Chroma/FluxMod and the number of modulations
        # needed by each (e.g., how many vectors from Approximator output each ModulationOut needs).
        mod_map = {}
        idx_count = 0
        for i in range(self.config.num_layers):  # DoubleStreamBlocks
            # img_mod.lin (attn + mlp) needs 2 gates, 2 scales, 2 shifts = 6 vectors from approximator
            # txt_mod.lin (attn + mlp) needs 2 gates, 2 scales, 2 shifts = 6 vectors from approximator
            mod_map[f"double_blocks.{i}.img_mod.lin"] = idx_count
            idx_count += 2 * 3  # 2 (attn, mlp) * 3 (s,s,g) modulation vectors
            mod_map[f"double_blocks.{i}.txt_mod.lin"] = idx_count
            idx_count += 2 * 3

        for i in range(self.config.num_single_layers):  # SingleStreamBlocks
            # modulation.lin needs 1 gate, 1 scale, 1 shift = 3 vectors
            mod_map[f"single_blocks.{i}.modulation.lin"] = idx_count
            idx_count += 1 * 3

        # Final Layer (adaLN_modulation.1) needs 1 scale, 1 shift = 2 vectors
        mod_map[f"final_layer.adaLN_modulation.1"] = idx_count
        idx_count += 1 * 2

        if idx_count != self.config.mod_index_length:
            logger.warning(
                f"Calculated mod_index_length {idx_count} does not match config {self.config.mod_index_length}. "
                f"Modulation distribution might be incorrect."
            )
        return mod_map, idx_count

    def _distribute_modulations(
        self, mod_vectors: torch.Tensor, mod_map: Dict[str, int]
    ) -> Dict[str, Any]:
        # mod_vectors: (B, mod_index_length, hidden_size) - output from Approximator
        # mod_map: tells which index in mod_vectors corresponds to which layer's modulation
        # Output: Dict of ModulationOut objects or [shift, scale] for final_layer

        distributed_mod_dict = {}
        hidden_size = self.hidden_size

        for key, start_idx in mod_map.items():
            if "double_blocks" in key:  # Needs 2 * ModulationOut(s,s,g)
                # Example: double_blocks.0.img_mod.lin
                # Needs attn_mod (s,s,g) and mlp_mod (s,s,g)
                # Each (s,s,g) takes 3 vectors from mod_vectors
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
            elif "single_blocks" in key:  # Needs 1 * ModulationOut(s,s,g)
                s, sh, g = (
                    mod_vectors[:, start_idx, :],
                    mod_vectors[:, start_idx + 1, :],
                    mod_vectors[:, start_idx + 2, :],
                )
                distributed_mod_dict[key] = ModulationOut(scale=s, shift=sh, gate=g)

            elif "final_layer" in key:  # Needs [shift, scale]
                sh, s = mod_vectors[:, start_idx, :], mod_vectors[:, start_idx + 1, :]
                distributed_mod_dict[key] = [
                    sh,
                    s,
                ]  # shift first, then scale, as per analysis

        return distributed_mod_dict

    def forward(
        self,
        hidden_states: torch.Tensor,  # Image latents (already patched by rearrange), B, L_img, C_in_model (e.g. 64)
        encoder_hidden_states: torch.Tensor,  # Text embeddings, B, L_txt, C_txt (e.g. 4096)
        timestep: torch.LongTensor,  # B,
        img_ids: torch.Tensor,  # B, L_img, N_axes_img (e.g., 3 for [0,y,x])
        txt_ids: torch.Tensor,  # B, L_txt, N_axes_txt (e.g., 1 for [seq_idx])
        guidance: torch.Tensor,  # B, CFG scale
        # pooled_projections: Optional[torch.Tensor] = None, # Not used by Chroma directly
        attention_mask: Optional[torch.Tensor] = None,  # Text mask (B, L_txt)
        return_dict: bool = True,
    ) -> Union[torch.Tensor, Transformer2DModelOutput]:

        batch_size, seq_len_img, _ = hidden_states.shape
        seq_len_txt = encoder_hidden_states.shape[1]

        # 1. Input Projections
        img_embeds = self.img_in(hidden_states)  # B, L_img, H
        txt_embeds = self.txt_in(encoder_hidden_states)  # B, L_txt, H

        # 2. Modulation Vector Generation (Approximator)
        # The Approximator pass is under torch.no_grad() in Chroma.forward original.
        # This is important for training if gradients shouldn't flow back into it from main model.
        # For inference, it doesn't matter as much. Diffusers models typically don't do this internally.
        # We'll assume standard diffusers behavior for now.

        # Create mod_indices tensor for Approximator (B, mod_len)
        # This is just torch.arange repeated for batch.
        # mod_indices_for_approx = torch.arange(self.config.mod_index_length, device=timestep.device).unsqueeze(0).repeat(batch_size, 1)
        # No, the ChromaApproximator's forward now handles creation of fixed_mod_indices.

        mod_vectors = self.distilled_guidance_layer(
            timestep, guidance, None
        )  # Pass None for mod_indices_tensor, it handles it

        # Distribute modulations
        mod_map, _ = self._get_mod_map()
        mod_vectors_dict = self._distribute_modulations(mod_vectors, mod_map)

        # 3. RoPE Embeddings Precomputation
        # Concatenate IDs for RoPE: text_ids first, then image_ids
        # ids shape: (B, L_txt + L_img, N_axes_max)
        # This requires padding N_axes if txt_ids (e.g. 1D) and img_ids (e.g. 3D) differ in N_axes.
        # Chroma analysis: `ids = torch.cat((txt_ids, img_ids), dim=1)`. Assume they are made compatible before cat.
        # Example: txt_ids (B, L_txt, 1), img_ids (B, L_img, 3)
        # Pad txt_ids to (B, L_txt, 3) with zeros for other axes.
        if txt_ids.shape[-1] < img_ids.shape[-1]:
            padding = torch.zeros(
                batch_size,
                seq_len_txt,
                img_ids.shape[-1] - txt_ids.shape[-1],
                device=txt_ids.device,
                dtype=txt_ids.dtype,
            )
            txt_ids_padded = torch.cat((txt_ids, padding), dim=-1)
        else:
            txt_ids_padded = txt_ids
        # Similarly for img_ids if it's smaller (unlikely here)

        ids_for_rope = torch.cat(
            (txt_ids_padded, img_ids), dim=1
        )  # (B, L_all, N_axes_max)

        # The pe_embedder (ChromaRoPE) needs to output freqs_cis in a shape
        # compatible with apply_rotary_pos_emb: (B, L_all, 1, D_head_rope) or (B, 1, L_all, D_head_rope)
        # D_head_rope is self.pe_embedder.dim = attention_head_dim
        # This is a placeholder:
        pe_freqs_cis = self.pe_embedder._compute_freqs_cis(
            ids_for_rope, self.pe_embedder.dim
        )  # (B, L_all, 1, D_head_rope)
        pe_freqs_cis = pe_freqs_cis.transpose(
            1, 2
        )  # (B, 1, L_all, D_head_rope) for apply_rotary_pos_emb if it expects B H L D

        # 4. Attention Masking (for shared attention)
        # `attention_mask` is typically (B, L_txt) for text padding.
        # Image part is all ones.
        # `txt_img_mask` for shared attention in DoubleStreamBlock and SingleStreamBlock.
        # Diffusers convention for attention_mask is often 0 for attend, -inf for mask.
        # Chroma analysis: `modify_mask_to_attend_padding` then combined with image mask.
        # For simplicity, assume attention_mask (if provided) is already correctly formatted for text,
        # and we need to extend it for images.

        # This creates a mask where 0 means attend, large negative means don't.
        # Suitable for adding to attention scores.
        txt_img_mask = None
        if attention_mask is not None:  # (B, L_txt)
            # Invert and make float for adding to scores
            text_attn_mask_float = (
                1.0 - attention_mask.float()
            ) * -10000.0  # (B, L_txt)
            text_attn_mask_square = text_attn_mask_float.unsqueeze(1).expand(
                -1, seq_len_txt, -1
            )  # (B, L_txt, L_txt)

            img_attn_mask_part = torch.zeros(
                batch_size,
                seq_len_img,
                seq_len_txt + seq_len_img,
                device=img_embeds.device,
            )

            # Combine:
            # Top-left: text_attn_mask_square
            # Top-right: text -> image (all attend for now)
            # Bottom-left: image -> text (all attend for now)
            # Bottom-right: image -> image (all attend)
            # A simpler full mask for cat(txt,img) sequence:
            full_mask = torch.zeros(
                batch_size,
                seq_len_txt + seq_len_img,
                seq_len_txt + seq_len_img,
                device=img_embeds.device,
            )
            full_mask[:, :seq_len_txt, :seq_len_txt] = (
                text_attn_mask_square  # Text self-attention mask
            )
            # Potentially allow text to attend to some padding via modify_mask_to_attend_padding
            # And restrict image tokens not to attend text padding if needed.
            # For now, using a simplified version. The original `modify_mask_to_attend_padding` is specific.
            txt_img_mask = full_mask

        # 5. DoubleStreamBlocks
        pe_freqs_cis_txt = pe_freqs_cis[:, :, :seq_len_txt, :]
        pe_freqs_cis_img = pe_freqs_cis[:, :, seq_len_txt:, :]
        # This slicing is problematic if pe_freqs_cis is (B, 1, L_all, D).
        # It assumes L dimension is at index 2.
        # If pe_freqs_cis is (B, L_all, 1, D_head_rope) as from _compute_freqs_cis placeholder:
        pe_freqs_cis_txt_block = pe_freqs_cis[:, :seq_len_txt, :, :]
        pe_freqs_cis_img_block = pe_freqs_cis[:, seq_len_txt:, :, :]
        # These individual RoPEs are not directly used in DoubleStreamBlock's forward as written,
        # as it expects a single pe_freqs_cis for the combined q_shared/k_shared.

        for i, block in enumerate(self.double_blocks):
            block_mod_key = (
                f"double_blocks.{i}"  # The keys in mod_vectors_dict are more specific
            )
            # The keys are like "double_blocks.0.img_mod.lin"
            current_mod_entry = {
                "img_mod.lin": mod_vectors_dict[f"double_blocks.{i}.img_mod.lin"],
                "txt_mod.lin": mod_vectors_dict[f"double_blocks.{i}.txt_mod.lin"],
            }
            img_embeds, txt_embeds = block(
                img_embeds,
                txt_embeds,
                pe_freqs_cis=pe_freqs_cis,  # Pass the full RoPE for shared attention
                mod_dict_entry=current_mod_entry,
                mask=txt_img_mask,
            )
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                # TODO: Implement checkpointing for this block if needed
                pass

        # 6. Concatenation & SingleStreamBlocks
        merged_embeds = torch.cat((txt_embeds, img_embeds), dim=1)  # B, L_all, H
        # Mask and RoPE are for the merged sequence (already computed as txt_img_mask, pe_freqs_cis)

        for i, block in enumerate(self.single_blocks):
            single_mod = mod_vectors_dict[f"single_blocks.{i}.modulation.lin"]
            merged_embeds = block(
                x=merged_embeds,
                pe_freqs_cis=pe_freqs_cis,
                mod_single=single_mod,
                mask=txt_img_mask,
            )
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                # TODO: Implement checkpointing
                pass

        # 7. Final Processing & Output
        # Select only image part
        img_output_embeds = merged_embeds[:, seq_len_txt:, :]  # B, L_img, H

        final_mod_params = mod_vectors_dict[
            "final_layer.adaLN_modulation.1"
        ]  # [shift, scale]
        output = self.final_layer(
            img_output_embeds, final_mod_params
        )  # B, L_img, C_out_model (e.g. 64)

        if not return_dict:
            return (output,)

        return Transformer2DModelOutput(sample=output)

    @classmethod
    def from_pretrained_chroma(cls, pretrained_model_path, config_path=None, **kwargs):
        # Placeholder for a method to load Chroma safetensors correctly
        # This would involve:
        # 1. Loading the config (or using a provided one).
        # 2. Instantiating the ChromaTransformer2DModel with this config.
        # 3. Loading the state_dict from safetensors.
        # 4. Mapping keys from Chroma's original naming to Diffusers' naming if different
        #    (though we are trying to match them).
        # 5. Loading the state_dict into the model.

        # For now, standard from_pretrained might work if names are perfectly aligned.
        # Diffusers' FromOriginalModelMixin might be useful here too.
        if config_path is None:
            # Try to load config from the same directory or use default
            pass

        # model = cls(**config_dict_from_json)
        # state_dict = load_safetensors(pretrained_model_path)
        # model.load_state_dict(state_dict)
        # return model
        raise NotImplementedError("from_pretrained_chroma is not yet implemented.")
