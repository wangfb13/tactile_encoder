"""
触觉 MAE 模型 (基于 CLIP-ViT-B-16 视觉编码器)。

- Encoder: ViT-B-16 (768-d, 12 层, 12 头), 由 transformers.ViTLayer 堆叠而成。
- Decoder: 轻量 ViT (384-d, 4 层, 6 头)。
- patch_embed: Conv2d(in_channels, 768, 16, 16)。data_tac2_s 为单帧 3 通道。
- load_clip_pretrained: 加载 open_clip ViT-B-16 视觉编码器权重。

实现说明:
  本环境 transformers (5.x) 中 ViTLayer.forward 直接返回 Tensor (而非 tuple),
  且注意力子模块命名为 q_proj / k_proj / v_proj / o_proj, mlp 为 fc1 / fc2。
  因此 encoder/decoder 均以 nn.ModuleList[ViTLayer] 构建, 不使用 ViTEncoder。
"""

import numpy as np
import torch
from torch import nn
from transformers import ViTConfig
from transformers.models.vit.modeling_vit import ViTLayer


# ----------------------------------------------------------------------
# 2D sin-cos 位置编码 (来自 MAE)
# ----------------------------------------------------------------------
def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # w 在前
    grid = np.stack(grid, axis=0)
    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])
    emb = np.concatenate([emb_h, emb_w], axis=1)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float32)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000 ** omega
    pos = pos.reshape(-1)
    out = np.einsum("m,d->md", pos, omega)
    emb_sin = np.sin(out)
    emb_cos = np.cos(out)
    emb = np.concatenate([emb_sin, emb_cos], axis=1)
    return emb


def _make_vit_config(dim, layers, heads, img_size, patch_size, in_channels):
    cfg = ViTConfig(
        hidden_size=dim,
        num_hidden_layers=layers,
        num_attention_heads=heads,
        intermediate_size=dim * 4,
        image_size=img_size,
        patch_size=patch_size,
        num_channels=in_channels,
    )
    # 作为独立模块使用时显式指定 eager attention, 避免分发警告
    cfg._attn_implementation = "eager"
    return cfg


class TactileMAE(nn.Module):
    """触觉 MAE: 输入 (B, C, 224, 224), 输出 196 patch tokens。"""

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        encoder_dim: int = 768,
        encoder_layers: int = 12,
        encoder_heads: int = 12,
        decoder_dim: int = 384,
        decoder_layers: int = 4,
        decoder_heads: int = 6,
        mask_ratio: float = 0.75,
        norm_pix_loss: bool = True,
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.mask_ratio = mask_ratio
        self.norm_pix_loss = norm_pix_loss
        self.encoder_dim = encoder_dim
        self.encoder_layers = encoder_layers

        self.num_patches = (img_size // patch_size) ** 2  # 196

        # ---- Encoder ----
        enc_cfg = _make_vit_config(
            encoder_dim, encoder_layers, encoder_heads, img_size, patch_size, in_channels
        )
        self.patch_embed = nn.Conv2d(
            in_channels, encoder_dim, kernel_size=patch_size, stride=patch_size
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, encoder_dim))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches + 1, encoder_dim), requires_grad=False
        )
        self.encoder_blocks = nn.ModuleList(
            [ViTLayer(enc_cfg) for _ in range(encoder_layers)]
        )
        self.encoder_norm = nn.LayerNorm(encoder_dim)

        # ---- Decoder ----
        dec_cfg = _make_vit_config(
            decoder_dim, decoder_layers, decoder_heads, img_size, patch_size, in_channels
        )
        self.decoder_embed = nn.Linear(encoder_dim, decoder_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_dim))
        self.decoder_pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches + 1, decoder_dim), requires_grad=False
        )
        self.decoder_blocks = nn.ModuleList(
            [ViTLayer(dec_cfg) for _ in range(decoder_layers)]
        )
        self.decoder_norm = nn.LayerNorm(decoder_dim)
        self.decoder_pred = nn.Linear(decoder_dim, patch_size ** 2 * in_channels)

        self.initialize_weights()

    # ------------------------------------------------------------------
    def initialize_weights(self):
        grid = int(self.num_patches ** 0.5)  # 14
        pos = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], grid, cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos).float().unsqueeze(0))

        dec_pos = get_2d_sincos_pos_embed(
            self.decoder_pos_embed.shape[-1], grid, cls_token=True
        )
        self.decoder_pos_embed.data.copy_(
            torch.from_numpy(dec_pos).float().unsqueeze(0)
        )

        w = self.patch_embed.weight.data
        nn.init.xavier_uniform_(w.view(w.shape[0], -1))

        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.normal_(self.mask_token, std=0.02)

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    # ------------------------------------------------------------------
    # 随机 mask
    # ------------------------------------------------------------------
    def random_masking(self, x, mask_ratio):
        B, L, D = x.shape
        len_keep = int(L * (1 - mask_ratio))
        noise = torch.rand(B, L, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, 1, ids_keep.unsqueeze(-1).expand(-1, -1, D))
        mask = torch.ones(B, L, device=x.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, 1, ids_restore)
        return x_masked, mask, ids_restore

    # ------------------------------------------------------------------
    # patchify / unpatchify
    # ------------------------------------------------------------------
    def patchify(self, imgs):
        p = self.patch_size
        c = self.in_channels
        h = w = self.img_size // p
        x = imgs.reshape(imgs.shape[0], c, h, p, w, p)
        x = x.permute(0, 2, 4, 3, 5, 1)
        x = x.reshape(imgs.shape[0], h * w, p * p * c)
        return x

    def unpatchify(self, x):
        p = self.patch_size
        c = self.in_channels
        h = w = self.img_size // p
        x = x.reshape(x.shape[0], h, w, p, p, c)
        x = x.permute(0, 5, 1, 3, 2, 4)
        x = x.reshape(x.shape[0], c, h * p, w * p)
        return x

    # ------------------------------------------------------------------
    # Encoder / Decoder
    # ------------------------------------------------------------------
    def forward_encoder(self, x):
        B = x.shape[0]
        x = self.patch_embed(x).flatten(2).transpose(1, 2)  # (B, 196, D)
        x = x + self.pos_embed[:, 1:, :]
        x, mask, ids_restore = self.random_masking(x, self.mask_ratio)
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        x = torch.cat([cls_token.expand(B, -1, -1), x], dim=1)
        for blk in self.encoder_blocks:
            x = blk(x)  # ViTLayer 返回 Tensor
        latent = self.encoder_norm(x)
        return latent, mask, ids_restore

    def forward_decoder(self, x, ids_restore):
        x = self.decoder_embed(x)
        cls_dec = x[:, :1, :]
        visible = x[:, 1:, :]
        B, _, D = cls_dec.shape
        num_patches = self.num_patches
        len_keep = visible.shape[1]
        num_masked = num_patches - len_keep
        x_full = torch.cat([visible, self.mask_token.repeat(B, num_masked, 1)], dim=1)
        gather_idx = ids_restore.unsqueeze(-1).expand(-1, -1, D)
        x_full = torch.gather(x_full, 1, gather_idx)
        x_dec = torch.cat([cls_dec, x_full], dim=1)
        x_dec = x_dec + self.decoder_pos_embed
        for blk in self.decoder_blocks:
            x_dec = blk(x_dec)
        x_dec = self.decoder_norm(x_dec)
        pred = self.decoder_pred(x_dec)
        pred = pred[:, 1:, :]
        return pred

    def forward_loss(self, imgs, pred, mask):
        target = self.patchify(imgs)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1e-6).sqrt()
        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)
        loss = (loss * mask).sum() / mask.sum()
        return loss

    def forward(self, imgs, mask_ratio=None):
        if mask_ratio is not None:
            self.mask_ratio = mask_ratio
        latent, mask, ids_restore = self.forward_encoder(imgs)
        pred = self.forward_decoder(latent, ids_restore)
        loss = self.forward_loss(imgs, pred, mask)
        return loss, pred, mask

    # ------------------------------------------------------------------
    # 加载 CLIP (open_clip) ViT-B-16 视觉编码器权重
    # ------------------------------------------------------------------
    @torch.no_grad()
    def load_clip_pretrained(self, ckpt_path: str, verbose: bool = True):
        """加载 open_clip ViT-B-16 视觉编码器权重到 MAE encoder。

        映射 (CLIP visual.* -> 本模型):
          conv1.weight                -> patch_embed.weight (通道扩展)
          class_embedding             -> cls_token
          ln_post.{w,b}               -> encoder_norm.{w,b}
          transformer.resblocks.i:
            ln_1                      -> encoder_blocks.i.layernorm_before
            ln_2                      -> encoder_blocks.i.layernorm_after
            attn.in_proj (qkv 合并)    -> encoder_blocks.i.attention.{q,k,v}_proj (拆分)
            attn.out_proj             -> encoder_blocks.i.attention.o_proj
            mlp.c_fc                  -> encoder_blocks.i.mlp.fc1
            mlp.c_proj                -> encoder_blocks.i.mlp.fc2

        说明:
          - 位置编码: 保留 MAE 的冻结 sin-cos, 不加载 CLIP 的 learned positional_embedding
            (二者语义不同, 且 MAE 设计要求 pos_embed 冻结)。
          - ln_pre / visual.proj: MAE 结构中无对应项, 跳过。
        """
        sd = torch.load(ckpt_path, map_location="cpu")
        if "state_dict" in sd:
            sd = sd["state_dict"]

        loaded, skipped = [], []

        def has(k):
            return k in sd

        # patch_embed: conv1 (768, 3, 16, 16) -> (768, in_channels, 16, 16)
        if has("visual.conv1.weight"):
            w = sd["visual.conv1.weight"]  # (out, 3, p, p)
            src_c = w.shape[1]
            dst_c = self.in_channels
            if dst_c == src_c:
                new_w = w
            elif dst_c % src_c == 0:
                reps = dst_c // src_c
                # 复制拼接并按重复次数缩放, 保持前向激活幅度一致
                new_w = w.repeat(1, reps, 1, 1) / float(reps)
            else:
                # 通道数非整数倍: 用通道均值扩展
                mean_w = w.mean(dim=1, keepdim=True)  # (out,1,p,p)
                new_w = mean_w.repeat(1, dst_c, 1, 1)
            self.patch_embed.weight.data.copy_(new_w.to(self.patch_embed.weight.dtype))
            loaded.append("patch_embed.weight")

        # cls_token
        if has("visual.class_embedding"):
            self.cls_token.data.copy_(
                sd["visual.class_embedding"].reshape(1, 1, -1).to(self.cls_token.dtype)
            )
            loaded.append("cls_token")

        # encoder_norm <- ln_post
        if has("visual.ln_post.weight"):
            self.encoder_norm.weight.data.copy_(sd["visual.ln_post.weight"])
            self.encoder_norm.bias.data.copy_(sd["visual.ln_post.bias"])
            loaded.append("encoder_norm")

        # transformer blocks
        dim = self.encoder_dim
        for i in range(self.encoder_layers):
            pre = f"visual.transformer.resblocks.{i}."
            if not has(pre + "ln_1.weight"):
                break
            blk = self.encoder_blocks[i]

            blk.layernorm_before.weight.data.copy_(sd[pre + "ln_1.weight"])
            blk.layernorm_before.bias.data.copy_(sd[pre + "ln_1.bias"])
            blk.layernorm_after.weight.data.copy_(sd[pre + "ln_2.weight"])
            blk.layernorm_after.bias.data.copy_(sd[pre + "ln_2.bias"])

            # qkv 合并权重 (3*dim, dim) 拆分
            in_w = sd[pre + "attn.in_proj_weight"]
            in_b = sd[pre + "attn.in_proj_bias"]
            qw, kw, vw = in_w[:dim], in_w[dim : 2 * dim], in_w[2 * dim :]
            qb, kb, vb = in_b[:dim], in_b[dim : 2 * dim], in_b[2 * dim :]
            blk.attention.q_proj.weight.data.copy_(qw)
            blk.attention.q_proj.bias.data.copy_(qb)
            blk.attention.k_proj.weight.data.copy_(kw)
            blk.attention.k_proj.bias.data.copy_(kb)
            blk.attention.v_proj.weight.data.copy_(vw)
            blk.attention.v_proj.bias.data.copy_(vb)

            blk.attention.o_proj.weight.data.copy_(sd[pre + "attn.out_proj.weight"])
            blk.attention.o_proj.bias.data.copy_(sd[pre + "attn.out_proj.bias"])

            blk.mlp.fc1.weight.data.copy_(sd[pre + "mlp.c_fc.weight"])
            blk.mlp.fc1.bias.data.copy_(sd[pre + "mlp.c_fc.bias"])
            blk.mlp.fc2.weight.data.copy_(sd[pre + "mlp.c_proj.weight"])
            blk.mlp.fc2.bias.data.copy_(sd[pre + "mlp.c_proj.bias"])

            loaded.append(f"encoder_blocks.{i}")

        skipped = [
            "visual.positional_embedding (保留 MAE sin-cos)",
            "visual.ln_pre (无对应)",
            "visual.proj (无对应)",
            "decoder.* (随机初始化)",
        ]
        if verbose:
            print(f"[CLIP load] 已加载: patch_embed/cls_token/encoder_norm + "
                  f"{self.encoder_layers} 个 encoder block")
            print(f"[CLIP load] 跳过: {skipped}")
        return loaded, skipped
