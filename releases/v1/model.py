"""Single-finger tactile MAE with a shared, full-view wrench auxiliary branch.

All images are depth / deformation-x / deformation-y, not display RGB.
The encoder exports pre-projection mean patch features. No contrastive loss.
"""
import math
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F


def position_embedding(height, width, dim):
    if dim % 4:
        raise ValueError("Position embedding dimension must be divisible by four")
    y, x = torch.meshgrid(torch.arange(height), torch.arange(width), indexing="ij")
    frequency = 1. / (10000 ** (torch.arange(dim // 4).float() / (dim // 4)))
    x = x.reshape(-1, 1).float() * frequency
    y = y.reshape(-1, 1).float() * frequency
    return torch.cat((x.sin(), x.cos(), y.sin(), y.cos()), dim=1)[None]


class Attention(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        if dim % heads:
            raise ValueError("Attention width must divide the number of heads")
        self.heads = heads
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        b, n, d = x.shape
        q, k, v = self.qkv(x).reshape(b, n, 3, self.heads, d // self.heads).permute(2, 0, 3, 1, 4).unbind(0)
        x = F.scaled_dot_product_attention(q, k, v, dropout_p=0.)
        return self.proj(x.transpose(1, 2).reshape(b, n, d))


class Block(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, heads)
        self.norm2 = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, 4 * dim)
        self.fc2 = nn.Linear(4 * dim, dim)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        return x + self.fc2(F.gelu(self.fc1(self.norm2(x))))


class TactileMAE(nn.Module):
    def __init__(self, image_size=(192, 256), patch_size=16, in_channels=3,
                 encoder_dim=768, encoder_layers=12, encoder_heads=12,
                 decoder_dim=384, decoder_layers=4, decoder_heads=6,
                 mask_ratio=.75, norm_pix_loss=False, force_head_init_std=.01):
        super().__init__()
        self.image_size = tuple(image_size)
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.encoder_dim = encoder_dim
        self.mask_ratio = float(mask_ratio)
        self.norm_pix_loss = bool(norm_pix_loss)
        if norm_pix_loss:
            raise ValueError("Per-patch target normalization is disabled for tactile magnitude preservation")
        if any(s % patch_size for s in self.image_size):
            raise ValueError("Height and width must be divisible by patch size")
        self.grid_size = tuple(s // patch_size for s in self.image_size)
        self.num_patches = math.prod(self.grid_size)
        self.config = dict(image_size=list(self.image_size), patch_size=patch_size,
                           in_channels=in_channels, encoder_dim=encoder_dim,
                           encoder_layers=encoder_layers, encoder_heads=encoder_heads,
                           decoder_dim=decoder_dim, decoder_layers=decoder_layers,
                           decoder_heads=decoder_heads, mask_ratio=mask_ratio,
                           norm_pix_loss=False, force_head_init_std=float(force_head_init_std))
        self.patch_embed = nn.Conv2d(in_channels, encoder_dim, patch_size, patch_size, bias=False)
        self.cls_token = nn.Parameter(torch.empty(1, 1, encoder_dim))
        self.register_buffer("pos_embed", position_embedding(*self.grid_size, encoder_dim))
        self.encoder_pre_norm = nn.LayerNorm(encoder_dim)
        self.encoder_blocks = nn.ModuleList([Block(encoder_dim, encoder_heads) for _ in range(encoder_layers)])
        self.encoder_norm = nn.LayerNorm(encoder_dim)
        self.decoder_embed = nn.Linear(encoder_dim, decoder_dim)
        self.mask_token = nn.Parameter(torch.empty(1, 1, decoder_dim))
        self.register_buffer("decoder_pos_embed", position_embedding(*self.grid_size, decoder_dim))
        self.decoder_blocks = nn.ModuleList([Block(decoder_dim, decoder_heads) for _ in range(decoder_layers)])
        self.decoder_norm = nn.LayerNorm(decoder_dim)
        self.decoder_pred = nn.Linear(decoder_dim, patch_size * patch_size * in_channels)
        self.force_projector = nn.Sequential(nn.Linear(encoder_dim, 256), nn.GELU(),
                                             nn.Linear(256, 128), nn.GELU())
        self.force_head = nn.Linear(128, 6)
        self.apply(self._initialize)
        nn.init.xavier_uniform_(self.patch_embed.weight.view(encoder_dim, -1))
        nn.init.normal_(self.cls_token, std=.02)
        nn.init.normal_(self.mask_token, std=.02)
        # A new auxiliary head should not start by dominating a pretrained encoder.
        nn.init.normal_(self.force_head.weight, std=force_head_init_std)
        nn.init.zeros_(self.force_head.bias)

    @staticmethod
    def _initialize(module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def patchify(self, images):
        b, c, h, w = images.shape
        if (h, w) != self.image_size or c != self.in_channels:
            raise ValueError("Unexpected tactile input shape: " + str(tuple(images.shape)))
        p = self.patch_size
        gh, gw = self.grid_size
        return images.reshape(b, c, gh, p, gw, p).permute(0, 2, 4, 3, 5, 1).reshape(b, gh * gw, p * p * c)

    def unpatchify(self, patches):
        b = patches.shape[0]
        p = self.patch_size
        gh, gw = self.grid_size
        return patches.reshape(b, gh, gw, p, p, self.in_channels).permute(0, 5, 1, 3, 2, 4).reshape(b, self.in_channels, *self.image_size)

    def forward_encoder(self, images, mask_ratio=None):
        # A call-local ratio is crucial: full-view force encoding must not disable MAE.
        ratio = self.mask_ratio if mask_ratio is None else float(mask_ratio)
        if not 0 <= ratio < 1:
            raise ValueError("mask_ratio must be in [0,1)")
        b = images.shape[0]
        if tuple(images.shape[1:]) != (self.in_channels, *self.image_size):
            raise ValueError("Unexpected tactile image dimensions")
        patches = self.patch_embed(images).flatten(2).transpose(1, 2) + self.pos_embed
        n = patches.shape[1]
        if ratio == 0:
            # No permutation or RNG consumption in the full-view branch.
            restore = torch.arange(n, device=images.device).expand(b, -1)
            mask = torch.zeros(b, n, device=images.device)
        else:
            keep = max(1, int(n * (1 - ratio)))
            order = torch.rand(b, n, device=images.device).argsort(dim=1)
            restore = order.argsort(dim=1)
            patches = patches.gather(1, order[:, :keep, None].expand(-1, -1, patches.shape[-1]))
            mask = torch.ones(b, n, device=images.device)
            mask[:, :keep] = 0
            mask = mask.gather(1, restore)
        tokens = self.encoder_pre_norm(torch.cat((self.cls_token.expand(b, -1, -1), patches), dim=1))
        for block in self.encoder_blocks:
            tokens = block(tokens)
        return self.encoder_norm(tokens), mask, restore

    def encode(self, images):
        tokens, _, _ = self.forward_encoder(images, mask_ratio=0.)
        return tokens[:, 1:].mean(dim=1)

    def predict_force(self, images):
        return self.force_head(self.force_projector(self.encode(images)))

    def forward_decoder(self, tokens, restore):
        tokens = self.decoder_embed(tokens)
        b, visible, dim = tokens[:, 1:].shape
        patches = torch.cat((tokens[:, 1:], self.mask_token.expand(b, self.num_patches - visible, dim)), dim=1)
        patches = patches.gather(1, restore[..., None].expand(-1, -1, dim))
        patches = patches + self.decoder_pos_embed
        tokens = torch.cat((tokens[:, :1], patches), dim=1)
        for block in self.decoder_blocks:
            tokens = block(tokens)
        return self.decoder_pred(self.decoder_norm(tokens)[:, 1:])

    def forward(self, images, force=None, force_valid=None, lambda_force=0., mask_ratio=None):
        tokens, mask, restore = self.forward_encoder(images, mask_ratio)
        predicted = self.forward_decoder(tokens, restore)
        per_patch = (predicted.float() - self.patchify(images).float()).square().mean(dim=-1)
        if mask.sum() == 0:
            raise ValueError("MAE training requires at least one masked patch; use encode() for unmasked inference")
        mae_loss = (per_patch * mask.float()).sum() / mask.sum()
        force_loss = mae_loss.new_zeros(())
        force_prediction = None
        valid_count = 0
        if lambda_force > 0:
            if force is None or force_valid is None:
                raise ValueError("Force targets and validity are required for the force branch")
            force_prediction = self.predict_force(images).float()
            valid = force_valid.bool() & torch.isfinite(force)
            valid_count = int(valid.sum().detach().item())
            # Never let NaN labels enter the arithmetic, even when a validity mask is false.
            safe_target = torch.where(valid, force.float(), torch.zeros_like(force).float())
            loss = F.smooth_l1_loss(force_prediction, safe_target, reduction="none")
            force_loss = (loss * valid.float()).sum() / valid.sum().clamp_min(1)
        return {"loss": mae_loss + float(lambda_force) * force_loss,
                "mae_loss": mae_loss, "force_loss": force_loss,
                "force_valid_count": valid_count, "pred": predicted,
                "mask": mask, "force_prediction": force_prediction}

    @torch.no_grad()
    def load_clip_pretrained(self, path):
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        source = checkpoint.get("state_dict", checkpoint)
        if not isinstance(source, dict):
            raise ValueError("Unexpected CLIP checkpoint format")
        source = {k.removeprefix("module."): v for k, v in source.items()}
        target = self.state_dict()
        mapping = {"patch_embed.weight": "visual.conv1.weight",
                   "cls_token": "visual.class_embedding",
                   "encoder_pre_norm.weight": "visual.ln_pre.weight",
                   "encoder_pre_norm.bias": "visual.ln_pre.bias",
                   "encoder_norm.weight": "visual.ln_post.weight",
                   "encoder_norm.bias": "visual.ln_post.bias"}
        suffixes = {"norm1.weight": "ln_1.weight", "norm1.bias": "ln_1.bias",
                    "norm2.weight": "ln_2.weight", "norm2.bias": "ln_2.bias",
                    "attn.qkv.weight": "attn.in_proj_weight", "attn.qkv.bias": "attn.in_proj_bias",
                    "attn.proj.weight": "attn.out_proj.weight", "attn.proj.bias": "attn.out_proj.bias",
                    "fc1.weight": "mlp.c_fc.weight", "fc1.bias": "mlp.c_fc.bias",
                    "fc2.weight": "mlp.c_proj.weight", "fc2.bias": "mlp.c_proj.bias"}
        for i in range(len(self.encoder_blocks)):
            for dst, src in suffixes.items():
                mapping[f"encoder_blocks.{i}.{dst}"] = f"visual.transformer.resblocks.{i}.{src}"
        failures, adapted = [], {}
        for dst, src in mapping.items():
            if src not in source:
                failures.append(f"missing source {src}")
                continue
            value = source[src]
            if dst == "cls_token":
                value = value.reshape(1, 1, -1)
            if tuple(value.shape) != tuple(target[dst].shape):
                failures.append(f"shape mismatch {src}: {tuple(value.shape)} -> {tuple(target[dst].shape)}")
                continue
            adapted[dst] = value
        if failures:
            raise ValueError("Incomplete CLIP encoder load: " + "; ".join(failures[:12]))
        incompatible = self.load_state_dict(adapted, strict=False)
        expected_encoder = {k for k, _ in self.named_parameters()
                            if k.startswith(("patch_embed.", "encoder_", "cls_token"))}
        if expected_encoder != set(adapted):
            raise RuntimeError("Encoder parameter coverage mismatch")
        return {"source": str(path), "encoder_tensors_loaded": len(adapted),
                "encoder_tensors_expected": len(expected_encoder), "encoder_parameter_coverage": 1.,
                "auxiliary_force_output_initialization_std": self.config["force_head_init_std"],
                "intentionally_not_loaded": ["CLIP learned positional embedding (new 12x16 fixed sin-cos)",
                                            "CLIP visual.proj and text tower", "MAE decoder and wrench projector/head"],
                "missing_nonencoder_keys": list(incompatible.missing_keys),
                "unexpected_keys": list(incompatible.unexpected_keys)}


def build_model(size="base", **kwargs):
    if size == "tiny":
        return TactileMAE(encoder_dim=48, encoder_layers=2, encoder_heads=3,
                          decoder_dim=32, decoder_layers=1, decoder_heads=4, **kwargs)
    if size != "base":
        raise ValueError("Unknown model size")
    return TactileMAE(**kwargs)


def load_model(checkpoint, device="cpu"):
    record = torch.load(Path(checkpoint), map_location="cpu", weights_only=False)
    model = TactileMAE(**record["model_config"])
    model.load_state_dict(record["model"], strict=True)
    model.to(device).eval()
    model.checkpoint_metadata = {key: record.get(key) for key in
                                 ("epoch", "global_step", "args", "normalization", "data_fingerprint")}
    return model
