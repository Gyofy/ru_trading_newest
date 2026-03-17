"""멀티모달 크립토 분류기 v3 + Foundation Pretrain.

MOMENT (ICML 2024) 스타일 Masked Patch Reconstruction Pretraining 적용.
OLinear (NormLin) + RevIN + Media Source Attention + Cross-Modal Fusion.

Architecture:
- PatchEmbedding: 시계열을 패치 단위로 분할 (MOMENT)
- MaskedAutoEncoder: 패치 마스킹 → 복원 자기지도학습 (MOMENT)
- SelectiveMasking: 복원 난이도 기반 지능적 마스킹 전략 (MOMENT)
- RevIN: Reversible Instance Normalization (입력 정규화 → 출력 역정규화)
- Price Encoder: OLinear-style NormLin (orthogonal domain linear)
- Media Encoder: Source-aware Attention (소스별 중요도 학습)
- Fusion: Cross-Modal Attention (price ↔ media)
- Selective Learning: dual-mask for robust training

Training Strategy:
  Phase 1: Foundation Pretrain — 전 코인 통합 MAE (자기지도학습)
  Phase 2: Per-coin Fine-tune — 개별 코인 분류 (지도학습)
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torch.amp import autocast, GradScaler
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import accuracy_score, f1_score
from datetime import datetime
from pathlib import Path
import json

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 미디어 소스 ID (확장: 전문 미디어 추가)
MEDIA_SOURCES = [
    "google_news", "reddit", "x_twitter", "youtube",
    "tiktok_proxy", "instagram_proxy",
    "coindesk", "cointelegraph", "theblock", "decrypt",
    "coinness", "glassnode", "messari", "tiger_research", "onchain",
]
NUM_SOURCES = len(MEDIA_SOURCES)


# ==================== 시간축 미디어 Feature 생성 ====================

def create_temporal_media_features(
    ohlcv_df: pd.DataFrame,
    media_items: dict,
    ticker: str,
) -> pd.DataFrame:
    """미디어 데이터를 시간축에 매핑하여 시변 feature를 생성합니다."""
    result = ohlcv_df.copy()
    ticker_lower = ticker.lower()

    for source_name in MEDIA_SOURCES:
        src_data = media_items.get(source_name, {})
        base_sentiment = src_data.get("avg_sentiment", 0) if isinstance(src_data, dict) else 0
        item_count = src_data.get("count", 0) if isinstance(src_data, dict) else 0

        # 시간 변동성 (실제 데이터에서는 뉴스 발행 시간과 매칭)
        np.random.seed(hash(f"{ticker}_{source_name}") % 2**31)
        noise = np.random.normal(0, abs(base_sentiment) * 0.3 + 0.01, len(result))
        temporal_sentiment = base_sentiment + noise

        # 아이템 수 기반 볼륨 노이즈
        vol_base = np.log1p(item_count)
        vol_noise = np.random.poisson(max(1, item_count // 3), len(result)).astype(float)

        result[f"media_{source_name}_sentiment"] = temporal_sentiment
        result[f"media_{source_name}_rolling_6h"] = pd.Series(
            temporal_sentiment, index=result.index).rolling(6, min_periods=1).mean()
        result[f"media_{source_name}_rolling_24h"] = pd.Series(
            temporal_sentiment, index=result.index).rolling(24, min_periods=1).mean()
        result[f"media_{source_name}_momentum"] = pd.Series(
            temporal_sentiment, index=result.index).diff(3).fillna(0)
        result[f"media_{source_name}_vol"] = pd.Series(
            temporal_sentiment, index=result.index).rolling(12, min_periods=1).std().fillna(0)

    # 크로스 소스 features
    sent_cols = [f"media_{s}_sentiment" for s in MEDIA_SOURCES]
    existing_cols = [c for c in sent_cols if c in result.columns]
    if existing_cols:
        result["media_consensus"] = result[existing_cols].mean(axis=1)
        result["media_divergence"] = result[existing_cols].std(axis=1)
        result["media_extreme"] = result[existing_cols].abs().max(axis=1)

        # 전문 미디어 vs 소셜 미디어 분리 감성
        pro_cols = [f"media_{s}_sentiment" for s in
                    ["coindesk", "cointelegraph", "theblock", "decrypt", "glassnode", "messari"]
                    if f"media_{s}_sentiment" in result.columns]
        social_cols = [f"media_{s}_sentiment" for s in
                       ["reddit", "x_twitter", "youtube", "tiktok_proxy", "instagram_proxy"]
                       if f"media_{s}_sentiment" in result.columns]
        if pro_cols:
            result["media_pro_consensus"] = result[pro_cols].mean(axis=1)
        if social_cols:
            result["media_social_consensus"] = result[social_cols].mean(axis=1)
        if pro_cols and social_cols:
            result["media_pro_social_gap"] = result["media_pro_consensus"] - result["media_social_consensus"]

    return result


# ==================== RevIN ====================

class RevIN(nn.Module):
    """Reversible Instance Normalization (ICLR 2022).

    시계열의 non-stationary 분포 이동을 처리합니다.
    입력에서 인스턴스별 통계를 제거 → 모델 처리 → 출력에서 복원.
    """

    def __init__(self, num_features: int, eps: float = 1e-5, affine: bool = True):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        if self.affine:
            self.affine_weight = nn.Parameter(torch.ones(num_features))
            self.affine_bias = nn.Parameter(torch.zeros(num_features))
        self.mean = None
        self.stdev = None

    def forward(self, x: torch.Tensor, mode: str) -> torch.Tensor:
        if mode == "norm":
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == "denorm":
            x = self._denormalize(x)
        return x

    def _get_statistics(self, x: torch.Tensor):
        dim2reduce = tuple(range(1, x.ndim - 1))
        self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.stdev = torch.sqrt(
            torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps
        ).detach()

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        x = (x - self.mean) / self.stdev
        if self.affine:
            x = x * self.affine_weight + self.affine_bias
        return x

    def _denormalize(self, x: torch.Tensor) -> torch.Tensor:
        if self.affine:
            x = (x - self.affine_bias) / (self.affine_weight + self.eps * self.eps)
        x = x * self.stdev + self.mean
        return x


# ==================== NormLin (OLinear Core) ====================

class NormLin(nn.Module):
    """Normalized Linear Layer (OLinear의 핵심).

    Self-attention을 대체하는 경량 cross-series mixing:
    - softplus로 비음수 가중치 보장
    - L1 정규화로 확률적 mixing (softmax 대체)
    - 초기값: 단위행렬 근처 → 학습 안정성
    """

    def __init__(self, token_num: int, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        # Cross-series mixing weight (attention 대체)
        init_weight = torch.eye(token_num) + torch.randn(token_num, token_num) * 0.02
        self.weight_mat = nn.Parameter(init_weight.unsqueeze(0))  # (1, N, N)

        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.norm1 = nn.LayerNorm(d_model)

        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, N, d_model) → (B, N, d_model)"""
        # NormLin mixing
        values = self.v_proj(x)
        A = F.softplus(self.weight_mat)
        A = F.normalize(A, p=1, dim=-1)  # L1 normalize rows
        A = self.dropout(A)
        new_x = torch.matmul(A, values)
        x = self.norm1(x + self.dropout(self.out_proj(new_x)))

        # FFN
        x = self.norm2(x + self.ffn(x))
        return x


# ==================== MOMENT: Patch Embedding ====================

class PatchEmbedding(nn.Module):
    """시계열을 패치 단위로 분할하여 임베딩합니다 (MOMENT, ICML 2024).

    시계열 T를 patch_len 크기의 겹치는 패치로 나누고,
    각 패치를 d_model 차원으로 투영합니다.
    """

    def __init__(self, patch_len: int, stride: int, d_model: int, input_dim: int):
        super().__init__()
        self.patch_len = patch_len
        self.stride = stride
        self.input_dim = input_dim
        self.proj = nn.Linear(patch_len * input_dim, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, D) → (B, num_patches, d_model)"""
        B, T, D = x.shape
        # Unfold: (B, T, D) → (B, num_patches, D, patch_len)
        patches = x.unfold(1, self.patch_len, self.stride)
        # → (B, num_patches, patch_len, D) → (B, num_patches, patch_len*D)
        patches = patches.permute(0, 1, 3, 2).contiguous()
        num_patches = patches.shape[1]
        patches = patches.reshape(B, num_patches, -1)
        return self.norm(self.proj(patches))

    def num_patches(self, seq_len: int) -> int:
        return max(1, (seq_len - self.patch_len) // self.stride + 1)


# ==================== MOMENT: Selective Masking Strategy ====================

class SelectiveMasking:
    """복원 난이도 기반 선택적 마스킹 전략 (MOMENT).

    - Phase 1 (초기): 랜덤 마스킹으로 전반적 패턴 학습
    - Phase 2 (중기): 복원 Loss가 높은 패치를 더 자주 마스킹 → 어려운 패턴 집중
    - Phase 3 (후기): 복원 Loss가 낮은(쉬운) 패치도 가끔 마스킹 → curriculum learning

    이를 통해 노이즈에 강건한 표현을 학습합니다.
    """

    def __init__(self, num_patches: int, mask_ratio: float = 0.4):
        self.num_patches = num_patches
        self.mask_ratio = mask_ratio
        self.patch_loss_history: dict[int, list[float]] = {}
        self.total_updates = 0

    def generate_mask(self, batch_size: int, device: torch.device,
                      epoch: int, total_epochs: int) -> torch.Tensor:
        """마스크 생성: True = 마스킹된 패치."""
        num_to_mask = max(1, int(self.num_patches * self.mask_ratio))
        mask = torch.zeros(batch_size, self.num_patches, dtype=torch.bool, device=device)

        progress = epoch / max(total_epochs, 1)

        if progress < 0.3 or not self.patch_loss_history:
            # Phase 1: 랜덤 마스킹
            for b in range(batch_size):
                indices = torch.randperm(self.num_patches, device=device)[:num_to_mask]
                mask[b, indices] = True
        else:
            # Phase 2/3: 복원 난이도 기반 선택적 마스킹
            avg_losses = torch.zeros(self.num_patches)
            for p_idx in range(self.num_patches):
                hist = self.patch_loss_history.get(p_idx, [])
                avg_losses[p_idx] = np.mean(hist[-10:]) if hist else 1.0

            # 난이도 기반 확률: 복원 어려운 패치를 더 자주 마스킹
            if progress < 0.7:
                # 중기: 어려운 패치 집중 (hard mining)
                probs = F.softmax(avg_losses * 2.0, dim=0).numpy().astype(np.float64)
            else:
                # 후기: curriculum — 쉬운 것도 섞어서 안정화
                uniform = np.ones(self.num_patches, dtype=np.float64) / self.num_patches
                hard_probs = F.softmax(avg_losses * 2.0, dim=0).numpy().astype(np.float64)
                probs = 0.5 * hard_probs + 0.5 * uniform

            # float 정밀도 보정: 합이 정확히 1이 되도록
            probs = probs / probs.sum()

            for b in range(batch_size):
                indices = np.random.choice(
                    self.num_patches, size=num_to_mask, replace=False, p=probs)
                mask[b, torch.tensor(indices, device=device)] = True

        return mask

    def update_loss_history(self, patch_losses: torch.Tensor):
        """패치별 복원 Loss 기록 업데이트."""
        # patch_losses: (num_patches,) — 패치별 평균 Loss
        self.total_updates += 1
        for p_idx in range(min(len(patch_losses), self.num_patches)):
            self.patch_loss_history.setdefault(p_idx, []).append(
                float(patch_losses[p_idx].item()))
            # 최근 20개만 유지
            if len(self.patch_loss_history[p_idx]) > 20:
                self.patch_loss_history[p_idx] = self.patch_loss_history[p_idx][-20:]


# ==================== MOMENT: Masked AutoEncoder ====================

class MaskedAutoEncoder(nn.Module):
    """MOMENT 스타일 Masked Patch Reconstruction.

    시계열 패치를 선택적으로 마스킹한 후 복원하는 자기지도학습.
    이를 통해 시계열의 구조적 패턴(트렌드, 변동성, 주기)을 사전학습합니다.
    """

    def __init__(self, num_patches: int, d_model: int, patch_len: int,
                 input_dim: int, num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.num_patches = num_patches
        self.d_model = d_model

        # Learnable mask token (마스킹된 패치를 대체)
        self.mask_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        # Positional embedding for patches
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches, d_model) * 0.02)

        # Encoder: NormLin layers (패치 간 관계 학습)
        self.encoder_layers = nn.ModuleList([
            NormLin(num_patches, d_model, d_model * 4, dropout)
            for _ in range(num_layers)
        ])

        # Reconstruction head: d_model → patch_len * input_dim
        self.reconstruction_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, patch_len * input_dim),
        )

    def forward(self, patch_embeddings: torch.Tensor,
                mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            patch_embeddings: (B, num_patches, d_model) — PatchEmbedding 출력
            mask: (B, num_patches) — True = 마스킹된 패치

        Returns:
            encoded: (B, num_patches, d_model) — 인코딩된 패치
            reconstructed: (B, num_patches, patch_len*input_dim) — 복원된 패치
        """
        B, N, D = patch_embeddings.shape

        # Add positional embedding
        x = patch_embeddings + self.pos_embed[:, :N, :]

        # Mask token으로 마스킹된 패치 대체
        mask_expanded = mask.unsqueeze(-1).expand(-1, -1, D)  # (B, N, D)
        x = torch.where(mask_expanded, self.mask_token.expand(B, N, D), x)

        # Encode (마스킹된 상태에서 패치 간 관계 학습)
        for layer in self.encoder_layers:
            x = layer(x)

        encoded = x

        # Reconstruct only masked patches (효율적)
        reconstructed = self.reconstruction_head(x)

        return encoded, reconstructed

    def compute_reconstruction_loss(
        self, reconstructed: torch.Tensor, original_patches: torch.Tensor,
        mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """마스킹된 패치에 대해서만 복원 Loss 계산.

        Returns:
            total_loss: 전체 복원 Loss
            per_patch_loss: (num_patches,) 패치별 Loss (SelectiveMasking 업데이트용)
        """
        B, N, D = reconstructed.shape

        # MSE loss only on masked patches
        diff = (reconstructed - original_patches) ** 2  # (B, N, D)
        per_sample_loss = diff.mean(dim=-1)  # (B, N)

        # Masked positions only
        masked_loss = per_sample_loss * mask.float()
        num_masked = mask.float().sum(dim=1, keepdim=True).clamp(min=1)
        total_loss = (masked_loss.sum(dim=1) / num_masked.squeeze(-1)).mean()

        # Per-patch average loss (for selective masking update)
        per_patch_loss = masked_loss.sum(dim=0) / mask.float().sum(dim=0).clamp(min=1)

        return total_loss, per_patch_loss


# ==================== Dataset (증강 포함) ====================

class MultiModalDataset(Dataset):
    """멀티모달 데이터셋 (슬라이딩 윈도우 + MixUp 지원)."""

    def __init__(self, price: np.ndarray, media: np.ndarray, labels: np.ndarray,
                 seq_len: int = 48, stride: int = 1):
        self.price = torch.FloatTensor(price)
        self.media = torch.FloatTensor(media)
        self.labels = torch.LongTensor(labels)
        self.seq_len = seq_len
        self.stride = stride

    def __len__(self):
        return max(0, (len(self.price) - self.seq_len) // self.stride)

    def __getitem__(self, idx):
        real_idx = idx * self.stride
        price_seq = self.price[real_idx:real_idx + self.seq_len]
        media_seq = self.media[real_idx:real_idx + self.seq_len]
        label = self.labels[real_idx + self.seq_len - 1]
        return price_seq, media_seq, label, real_idx


# ==================== 모델 아키텍처 ====================

class MediaSourceAttention(nn.Module):
    """미디어 소스별 중요도를 학습하는 Attention 모듈."""

    def __init__(self, d_media: int, num_sources: int = NUM_SOURCES):
        super().__init__()
        self.num_sources = num_sources
        features_per_source = 5

        self.source_embed = nn.Embedding(num_sources, d_media)
        self.query = nn.Linear(d_media, d_media)
        self.key = nn.Linear(features_per_source, d_media)
        self.value = nn.Linear(features_per_source, d_media)
        self.scale = d_media ** 0.5

        self.source_importance = nn.Parameter(torch.ones(num_sources) / num_sources)

    def forward(self, media_features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        B, T, D = media_features.shape
        features_per_source = D // self.num_sources
        usable_D = features_per_source * self.num_sources
        source_feats = media_features[:, :, :usable_D].view(B, T, self.num_sources, features_per_source)

        src_ids = torch.arange(self.num_sources, device=media_features.device)
        src_embed = self.source_embed(src_ids)

        Q = self.query(src_embed).unsqueeze(0).unsqueeze(0)
        K = self.key(source_feats)
        V = self.value(source_feats)

        attn = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        attn_weights = torch.softmax(attn, dim=-1)

        source_weights = torch.softmax(self.source_importance, dim=0)

        attended = torch.matmul(attn_weights, V)
        weighted = attended * source_weights.view(1, 1, -1, 1)
        encoded = weighted.sum(dim=2)

        return encoded, source_weights


class CrossModalFusion(nn.Module):
    """Price ↔ Media Cross-Attention Fusion."""

    def __init__(self, d_price: int, d_media: int, d_fusion: int, nhead: int = 4):
        super().__init__()
        self.price_proj = nn.Linear(d_price, d_fusion)
        self.media_proj = nn.Linear(d_media, d_fusion)
        self.cross_attn_p2m = nn.MultiheadAttention(d_fusion, nhead, batch_first=True, dropout=0.1)
        self.cross_attn_m2p = nn.MultiheadAttention(d_fusion, nhead, batch_first=True, dropout=0.1)
        self.norm1 = nn.LayerNorm(d_fusion)
        self.norm2 = nn.LayerNorm(d_fusion)
        self.ffn = nn.Sequential(
            nn.Linear(d_fusion * 2, d_fusion),
            nn.GELU(),
            nn.Dropout(0.1),
        )

    def forward(self, price_enc: torch.Tensor, media_enc: torch.Tensor) -> torch.Tensor:
        p = self.price_proj(price_enc)
        m = self.media_proj(media_enc)
        p2m, _ = self.cross_attn_p2m(p, m, m)
        m2p, _ = self.cross_attn_m2p(m, p, p)
        p_fused = self.norm1(p + p2m)
        m_fused = self.norm2(m + m2p)
        fused = torch.cat([p_fused, m_fused], dim=-1)
        return self.ffn(fused)


class MultiModalCryptoClassifier(nn.Module):
    """멀티모달 크립토 분류 모델 v3.

    MOMENT (ICML 2024) 스타일 Masked Patch Reconstruction Pretraining 적용.

    Price Path:
      RevIN → PatchEmbedding → [MaskedAutoEncoder pretrain] → NormLin Encoder
    Media Path:
      RevIN → MediaSourceAttention
    Fusion:
      CrossModalFusion → Classification

    2-Phase Training:
      Phase 1: Masked Reconstruction Pretraining (자기지도학습)
      Phase 2: Fine-tune for Classification (지도학습)
    """

    # Patch hyperparameters
    PATCH_LEN = 8    # 5분봉 8개 = 40분 단위 패치
    PATCH_STRIDE = 4  # 50% overlap → 정보 보존

    def __init__(self, num_price_features: int, num_media_features: int,
                 seq_len: int = 48, d_model: int = 64, nhead: int = 4,
                 num_layers: int = 2, num_classes: int = 3, dropout: float = 0.3):
        super().__init__()
        self.seq_len = seq_len
        self.num_price_features = num_price_features
        self.num_media_features = num_media_features
        self.d_model = d_model

        # RevIN for normalization
        self.revin_price = RevIN(num_price_features)
        self.revin_media = RevIN(num_media_features)

        # === MOMENT: Patch-based Price Encoding ===
        self.price_patch_embed = PatchEmbedding(
            patch_len=self.PATCH_LEN,
            stride=self.PATCH_STRIDE,
            d_model=d_model,
            input_dim=num_price_features,
        )
        n_patches = self.price_patch_embed.num_patches(seq_len)
        self.n_patches = n_patches

        # MOMENT: Masked AutoEncoder for pretraining
        self.masked_ae = MaskedAutoEncoder(
            num_patches=n_patches,
            d_model=d_model,
            patch_len=self.PATCH_LEN,
            input_dim=num_price_features,
            num_layers=num_layers,
            dropout=dropout,
        )

        # MOMENT: Selective Masking
        self.selective_masking = SelectiveMasking(
            num_patches=n_patches, mask_ratio=0.4)

        # Post-pretrain NormLin layers (fine-tune encoder)
        self.normlin_layers = nn.ModuleList([
            NormLin(n_patches, d_model, d_model * 4, dropout)
            for _ in range(num_layers)
        ])

        # Media encoder with source attention
        self.media_attention = MediaSourceAttention(d_media=d_model)

        # Cross-modal fusion
        self.fusion = CrossModalFusion(d_model, d_model, d_model, nhead)

        # Classifier
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes),
        )

        # Pretrain state
        self._pretrained = False

    def get_original_patches(self, price_normed: torch.Tensor) -> torch.Tensor:
        """RevIN 정규화된 가격 입력에서 원본 패치를 추출합니다 (복원 Loss 계산용)."""
        B, T, D = price_normed.shape
        patches = price_normed.unfold(1, self.PATCH_LEN, self.PATCH_STRIDE)
        patches = patches.permute(0, 1, 3, 2).contiguous()
        num_patches = patches.shape[1]
        return patches.reshape(B, num_patches, -1)  # (B, num_patches, patch_len*D)

    def forward_pretrain(self, price_input: torch.Tensor,
                         epoch: int, total_epochs: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Phase 1: Masked Reconstruction Pretraining.

        Returns:
            reconstruction_loss: 마스킹된 패치 복원 Loss
            per_patch_loss: 패치별 Loss (SelectiveMasking 업데이트용)
        """
        B = price_input.shape[0]

        # RevIN normalize
        price_normed = self.revin_price(price_input, mode="norm")

        # Get original patches (복원 타겟)
        original_patches = self.get_original_patches(price_normed)

        # Patch embedding
        patch_emb = self.price_patch_embed(price_normed)

        # Generate selective mask
        mask = self.selective_masking.generate_mask(
            B, price_input.device, epoch, total_epochs)

        # Masked encoding + reconstruction
        encoded, reconstructed = self.masked_ae(patch_emb, mask)

        # Compute reconstruction loss (마스킹된 패치만)
        recon_loss, per_patch_loss = self.masked_ae.compute_reconstruction_loss(
            reconstructed, original_patches, mask)

        # Update selective masking history
        self.selective_masking.update_loss_history(per_patch_loss.detach())

        return recon_loss, per_patch_loss

    def forward(self, price_input: torch.Tensor, media_input: torch.Tensor):
        """Phase 2: Classification (pretrained encoder 사용)."""
        B, T, _ = price_input.shape

        # RevIN normalize
        price_normed = self.revin_price(price_input, mode="norm")
        media_normed = self.revin_media(media_input, mode="norm")

        # MOMENT: Patch embedding (pretrained)
        patch_emb = self.price_patch_embed(price_normed)

        # Use pretrained MAE encoder (without masking)
        no_mask = torch.zeros(B, self.n_patches, dtype=torch.bool,
                              device=price_input.device)
        encoded, _ = self.masked_ae(patch_emb, no_mask)

        # NormLin layers (fine-tune cross-patch mixing)
        for normlin in self.normlin_layers:
            encoded = normlin(encoded)

        # Pool over patches → expand to temporal
        price_pooled = encoded.mean(dim=1)
        price_temporal = price_pooled.unsqueeze(1).expand(-1, T, -1)

        # Media encoding (skip when dummy media — e.g. num_media_features <= 1)
        if self.num_media_features > 1 and media_input.shape[-1] > 1:
            media_enc, source_weights = self.media_attention(media_normed)
            # Cross-modal fusion
            fused = self.fusion(price_temporal, media_enc)
        else:
            fused = price_temporal
            source_weights = torch.zeros(len(MEDIA_SOURCES), device=price_input.device)

        # Classification from last timestep
        last = fused[:, -1, :]
        logits = self.classifier(last)

        return logits, source_weights

    def get_source_weights(self) -> dict[str, float]:
        weights = torch.softmax(self.media_attention.source_importance.detach(), dim=0)
        return {name: round(w.item(), 4) for name, w in zip(MEDIA_SOURCES, weights)}


# ==================== Selective Learning ====================

class LightSelectiveLearning:
    """경량 Selective Learning (dual-mask)."""

    def __init__(self, r_u: float = 0.1, r_a: float = 0.1):
        self.r_u = r_u
        self.r_a = r_a
        self.loss_history = {}

    def selective_loss(self, logits: torch.Tensor, targets: torch.Tensor,
                       indices: torch.Tensor, epoch: int) -> torch.Tensor:
        ce = nn.CrossEntropyLoss(reduction="none")(logits, targets)

        if epoch < 10:
            return ce.mean()

        for i, idx in enumerate(indices.cpu().numpy()):
            idx = int(idx)
            self.loss_history.setdefault(idx, []).append(ce[i].item())
            if len(self.loss_history[idx]) > 5:
                self.loss_history[idx] = self.loss_history[idx][-5:]

        variances = []
        for i, idx in enumerate(indices.cpu().numpy()):
            hist = self.loss_history.get(int(idx), [])
            variances.append(np.var(hist) if len(hist) >= 3 else 0.0)

        var_tensor = torch.FloatTensor(variances).to(logits.device)

        if var_tensor.sum() > 0:
            u_threshold = torch.quantile(var_tensor, 1 - self.r_u)
            u_mask = var_tensor <= u_threshold
        else:
            u_mask = torch.ones(len(ce), dtype=torch.bool, device=logits.device)

        a_threshold = torch.quantile(ce.detach(), 1 - self.r_a)
        a_mask = ce.detach() <= a_threshold

        mask = u_mask & a_mask
        if mask.sum() > 0:
            return (ce * mask.float()).sum() / mask.sum()
        return ce.mean()


# ==================== 통합 학습기 ====================

class MultiModalTrainer:
    """멀티모달 모델 학습 및 평가.

    2-Phase Training (MOMENT, ICML 2024):
      Phase 1: Masked Patch Reconstruction Pretraining (자기지도학습)
      Phase 2: Fine-tune for Classification (지도학습 + Selective Learning)
    """

    def __init__(self, num_price_features: int, num_media_features: int,
                 seq_len: int = 48, num_classes: int = 3):
        self.seq_len = seq_len
        self.price_scaler = RobustScaler()
        self.media_scaler = RobustScaler()

        self.model = MultiModalCryptoClassifier(
            num_price_features=num_price_features,
            num_media_features=num_media_features,
            seq_len=seq_len, d_model=64, nhead=4, num_layers=2,
            num_classes=num_classes, dropout=0.3,
        ).to(device)

        # torch.compile (PyTorch 2.x JIT 가속) — Triton 필요, 없으면 eager
        try:
            self.model = torch.compile(self.model, mode="default", backend="eager")
            print("    [torch.compile] Enabled (eager backend)")
        except Exception:
            print("    [torch.compile] Skipped, using eager mode")

        # Mixed Precision (FP16) — RTX 3090 Tensor Core 활용
        self.amp_scaler = GradScaler("cuda")
        self.use_amp = device.type == "cuda"

        self.selective = LightSelectiveLearning(r_u=0.1, r_a=0.1)

    # ---- Phase 1: MOMENT Masked Reconstruction Pretraining ----

    def pretrain_moment(self, price_train: np.ndarray,
                        pretrain_epochs: int = 30, batch_size: int = 256,
                        lr: float = 1e-3, num_workers: int = 6) -> dict:
        """MOMENT 스타일 마스킹 자기지도 사전학습.

        가격 시계열만 사용하여 패치 복원 능력을 학습합니다.
        이를 통해 트렌드, 변동성, 주기 등 시계열 구조를 사전에 이해합니다.
        """
        print("    [MOMENT] Phase 1: Masked Patch Reconstruction Pretraining...")

        price_scaled = self.price_scaler.fit_transform(
            np.nan_to_num(price_train, nan=0, posinf=0, neginf=0))

        # Price-only dataset (라벨 불필요 — 자기지도학습)
        price_tensor = torch.FloatTensor(price_scaled)
        n_samples = max(0, len(price_tensor) - self.seq_len)
        if n_samples < 20:
            print("    [MOMENT] Insufficient data for pretraining, skipping.")
            return {"status": "skipped", "pretrain_epochs": 0}

        # 사전학습용 optimizer (MAE + PatchEmbed만 학습)
        pretrain_params = (
            list(self.model.price_patch_embed.parameters()) +
            list(self.model.masked_ae.parameters()) +
            list(self.model.revin_price.parameters())
        )
        optimizer = optim.AdamW(pretrain_params, lr=lr, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=pretrain_epochs)

        best_loss = float("inf")
        loss_history = []

        for epoch in range(pretrain_epochs):
            self.model.train()
            epoch_loss = 0
            batches = 0

            # Random sampling of sequences
            indices = np.random.permutation(n_samples)
            for start in range(0, len(indices) - batch_size, batch_size):
                batch_idx = indices[start:start + batch_size]
                price_batch = torch.stack([
                    price_tensor[i:i + self.seq_len] for i in batch_idx
                ]).to(device)

                optimizer.zero_grad()

                # AMP: Mixed Precision (FP16)
                if self.use_amp:
                    with autocast("cuda"):
                        recon_loss, per_patch_loss = self.model.forward_pretrain(
                            price_batch, epoch, pretrain_epochs)
                    self.amp_scaler.scale(recon_loss).backward()
                    self.amp_scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(pretrain_params, 1.0)
                    self.amp_scaler.step(optimizer)
                    self.amp_scaler.update()
                else:
                    recon_loss, per_patch_loss = self.model.forward_pretrain(
                        price_batch, epoch, pretrain_epochs)
                    recon_loss.backward()
                    torch.nn.utils.clip_grad_norm_(pretrain_params, 1.0)
                    optimizer.step()

                epoch_loss += recon_loss.item()
                batches += 1

            scheduler.step()
            avg_loss = epoch_loss / max(batches, 1)
            loss_history.append(avg_loss)

            if avg_loss < best_loss:
                best_loss = avg_loss

            if (epoch + 1) % 10 == 0:
                masking_info = self.model.selective_masking
                phase = "random" if epoch < pretrain_epochs * 0.3 else (
                    "hard-mining" if epoch < pretrain_epochs * 0.7 else "curriculum")
                print(f"      [MOMENT] Epoch {epoch+1}/{pretrain_epochs}: "
                      f"recon_loss={avg_loss:.4f}, best={best_loss:.4f}, "
                      f"strategy={phase}")

        self.model._pretrained = True
        improvement = (loss_history[0] - loss_history[-1]) / (loss_history[0] + 1e-10) * 100

        print(f"    [MOMENT] Pretraining done: loss {loss_history[0]:.4f} → {loss_history[-1]:.4f} "
              f"({improvement:.1f}% improvement)")

        return {
            "status": "completed",
            "pretrain_epochs": pretrain_epochs,
            "initial_loss": round(loss_history[0], 4),
            "final_loss": round(loss_history[-1], 4),
            "best_loss": round(best_loss, 4),
            "improvement_pct": round(improvement, 1),
        }

    # ---- Phase 2: Fine-tune Classification ----

    def train(self, price_train: np.ndarray, media_train: np.ndarray,
              y_train: np.ndarray, epochs: int = 80, batch_size: int = 256,
              lr: float = 3e-4, num_workers: int = 6,
              extra_datasets: list = None,
              pretrain_epochs: int = 30) -> dict:
        """2-Phase 학습: MOMENT pretrain → Classification fine-tune."""

        # === Phase 1: MOMENT Pretraining (자기지도학습) ===
        # Foundation pretrain이 로드된 경우 또는 pretrain_epochs=0이면 스킵
        if not self.model._pretrained and pretrain_epochs > 0:
            pretrain_result = self.pretrain_moment(
                price_train, pretrain_epochs=pretrain_epochs,
                batch_size=batch_size, lr=lr * 3, num_workers=num_workers)
        elif self.model._pretrained:
            pretrain_result = {"status": "foundation_loaded"}
            print("    [Foundation] Using shared pretrained weights — skip per-coin pretrain")
        else:
            pretrain_result = {"status": "skipped"}

        # === Phase 2: Classification Fine-tune ===
        print("    [MOMENT] Phase 2: Classification Fine-tuning...")

        # price_scaler가 pretrain에서 이미 fit된 경우 transform만
        if self.model._pretrained:
            price_scaled = self.price_scaler.transform(
                np.nan_to_num(price_train, nan=0, posinf=0, neginf=0))
        else:
            price_scaled = self.price_scaler.fit_transform(
                np.nan_to_num(price_train, nan=0, posinf=0, neginf=0))
        media_scaled = self.media_scaler.fit_transform(
            np.nan_to_num(media_train, nan=0, posinf=0, neginf=0))

        # 슬라이딩 윈도우 + 오프셋 증강
        datasets = [
            MultiModalDataset(price_scaled, media_scaled, y_train, self.seq_len, stride=1),
        ]
        if len(price_scaled) > self.seq_len + 12:
            datasets.append(
                MultiModalDataset(price_scaled[6:], media_scaled[6:], y_train[6:],
                                  self.seq_len, stride=1))

        if extra_datasets:
            datasets.extend(extra_datasets)

        combined = ConcatDataset(datasets)
        if len(combined) < 20:
            return {"status": "insufficient_data", "pretrain": pretrain_result}

        # Class weights (supports binary and multi-class)
        unique, counts = np.unique(y_train, return_counts=True)
        num_cls = self.model.classifier[-1].out_features  # from model definition
        cw = np.ones(num_cls)
        for u, c in zip(unique, counts):
            if int(u) < num_cls:
                cw[int(u)] = len(y_train) / (num_cls * c + 1)
        class_weights = torch.FloatTensor(cw).to(device)

        # Validation split
        val_size = max(int(len(combined) * 0.15), batch_size)
        train_size = len(combined) - val_size
        train_ds, val_ds = torch.utils.data.random_split(combined, [train_size, val_size])

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                                  drop_last=True, num_workers=num_workers,
                                  pin_memory=num_workers > 0,
                                  persistent_workers=num_workers > 0)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                                drop_last=False, num_workers=0)

        # Fine-tune: 더 낮은 lr로 pretrained 가중치 보존
        pretrained_lr = lr * 0.1  # 사전학습 파라미터는 느리게 업데이트
        finetune_lr = lr

        param_groups = [
            {"params": list(self.model.price_patch_embed.parameters()) +
                       list(self.model.masked_ae.parameters()),
             "lr": pretrained_lr},  # Pretrained layers: 낮은 lr
            {"params": list(self.model.normlin_layers.parameters()) +
                       list(self.model.media_attention.parameters()) +
                       list(self.model.fusion.parameters()) +
                       list(self.model.classifier.parameters()) +
                       list(self.model.revin_media.parameters()),
             "lr": finetune_lr},   # New layers: 표준 lr
        ]
        optimizer = optim.AdamW(param_groups, weight_decay=5e-3)
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=15, T_mult=2)

        best_val_loss = float("inf")
        best_state = None
        patience_counter = 0
        patience = 20
        source_weights_history = []

        for epoch in range(epochs):
            # Train
            self.model.train()
            total_loss = 0
            batches = 0

            for price_batch, media_batch, labels, indices in train_loader:
                price_batch = price_batch.to(device)
                media_batch = media_batch.to(device)
                labels = labels.to(device)
                indices = indices.to(device)

                optimizer.zero_grad()

                # AMP: Mixed Precision (FP16) — RTX 3090 Tensor Core
                if self.use_amp:
                    with autocast("cuda"):
                        logits, src_weights = self.model(price_batch, media_batch)
                        loss = self.selective.selective_loss(logits, labels, indices, epoch)
                        diversity_loss = -torch.std(src_weights) * 0.1
                        total = loss + diversity_loss
                    self.amp_scaler.scale(total).backward()
                    self.amp_scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.amp_scaler.step(optimizer)
                    self.amp_scaler.update()
                else:
                    logits, src_weights = self.model(price_batch, media_batch)
                    loss = self.selective.selective_loss(logits, labels, indices, epoch)
                    diversity_loss = -torch.std(src_weights) * 0.1
                    total = loss + diversity_loss
                    total.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    optimizer.step()

                total_loss += loss.item()
                batches += 1

            scheduler.step()
            avg_loss = total_loss / max(batches, 1)

            # Validation
            self.model.eval()
            val_loss = 0
            val_batches = 0
            with torch.no_grad():
                for p_b, m_b, lb, _ in val_loader:
                    p_b, m_b, lb = p_b.to(device), m_b.to(device), lb.to(device)
                    logits_v, _ = self.model(p_b, m_b)
                    vl = nn.CrossEntropyLoss(weight=class_weights)(logits_v, lb)
                    val_loss += vl.item()
                    val_batches += 1
            avg_val_loss = val_loss / max(val_batches, 1)

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_state = {k: v.clone() for k, v in self.model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1

            if (epoch + 1) % 10 == 0:
                sw = self.model.get_source_weights()
                source_weights_history.append(sw)
                top_source = max(sw, key=sw.get)
                print(f"      Epoch {epoch+1}: loss={avg_loss:.4f}, val={avg_val_loss:.4f}, top={top_source}({sw[top_source]:.1%})")

            if patience_counter >= patience and epoch >= 40:
                print(f"      Early stop at epoch {epoch+1}")
                break

        if best_state is not None:
            self.model.load_state_dict(best_state)

        final_weights = self.model.get_source_weights()

        return {
            "final_loss": round(best_val_loss, 4),
            "epochs": epochs,
            "source_weights": final_weights,
            "source_weights_history": source_weights_history,
            "data_size": len(combined),
            "pretrain": pretrain_result,
            "moment_pretrained": self.model._pretrained,
        }

    def predict(self, price_input: np.ndarray, media_input: np.ndarray) -> tuple:
        price_scaled = self.price_scaler.transform(
            np.nan_to_num(price_input, nan=0, posinf=0, neginf=0))
        media_scaled = self.media_scaler.transform(
            np.nan_to_num(media_input, nan=0, posinf=0, neginf=0))

        if len(price_scaled) < self.seq_len:
            pad_p = np.zeros((self.seq_len - len(price_scaled), price_scaled.shape[1]))
            pad_m = np.zeros((self.seq_len - len(media_scaled), media_scaled.shape[1]))
            price_scaled = np.vstack([pad_p, price_scaled])
            media_scaled = np.vstack([pad_m, media_scaled])

        p_seq = torch.FloatTensor(price_scaled[-self.seq_len:]).unsqueeze(0).to(device)
        m_seq = torch.FloatTensor(media_scaled[-self.seq_len:]).unsqueeze(0).to(device)

        self.model.eval()
        with torch.no_grad():
            logits, src_weights = self.model(p_seq, m_seq)
            proba = torch.softmax(logits, dim=1).cpu().numpy()
            pred_class = np.argmax(proba, axis=1)

        return pred_class, proba, self.model.get_source_weights()

    def evaluate(self, price_test: np.ndarray, media_test: np.ndarray,
                 y_test: np.ndarray) -> dict:
        price_scaled = self.price_scaler.transform(
            np.nan_to_num(price_test, nan=0, posinf=0, neginf=0))
        media_scaled = self.media_scaler.transform(
            np.nan_to_num(media_test, nan=0, posinf=0, neginf=0))

        dataset = MultiModalDataset(price_scaled, media_scaled, y_test, self.seq_len)
        if len(dataset) == 0:
            return {"accuracy": 0, "f1": 0}

        loader = DataLoader(dataset, batch_size=256, shuffle=False)
        all_preds, all_labels = [], []

        self.model.eval()
        with torch.no_grad():
            for p_batch, m_batch, labels, _ in loader:
                p_batch = p_batch.to(device)
                m_batch = m_batch.to(device)

                logits, _ = self.model(p_batch, m_batch)
                preds = torch.argmax(logits, dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(labels.numpy())

        return {
            "accuracy": round(accuracy_score(all_labels, all_preds), 4),
            "f1": round(f1_score(all_labels, all_preds, average="macro", zero_division=0), 4),
            "source_weights": self.model.get_source_weights(),
        }

    def load_foundation(self, checkpoint_path: str) -> bool:
        """Foundation pretrain 체크포인트에서 공유 가중치를 로드.

        로드 대상: PatchEmbedding + MaskedAutoEncoder + RevIN(price)
        → per-coin finetune에서 이 가중치 위에 분류 레이어를 학습.
        """
        ckpt_path = Path(checkpoint_path)
        if not ckpt_path.exists():
            print(f"    [Foundation] Checkpoint not found: {checkpoint_path}")
            return False

        checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
        foundation_state = checkpoint["model_state"]

        # Foundation 가중치만 선택적 로드 (PatchEmbed + MAE + RevIN_price)
        model_state = self.model.state_dict()
        loaded_keys = []
        for key, value in foundation_state.items():
            # torch.compile _orig_mod. 접두사 양쪽 모두 시도
            candidates = [key]
            if key.startswith("_orig_mod."):
                candidates.append(key[len("_orig_mod."):])
            else:
                candidates.append(f"_orig_mod.{key}")

            for candidate in candidates:
                if candidate in model_state and model_state[candidate].shape == value.shape:
                    model_state[candidate] = value
                    loaded_keys.append(candidate)
                    break

        self.model.load_state_dict(model_state)
        self.model._pretrained = True

        # Foundation scaler도 로드
        if "price_scaler" in checkpoint:
            self.price_scaler = checkpoint["price_scaler"]

        print(f"    [Foundation] Loaded {len(loaded_keys)} params from {ckpt_path.name}")
        return True


# ==================== Foundation Pretrainer ====================

class FoundationPretrainer:
    """전 코인 통합 Foundation MAE Pretraining.

    전략:
    1. 모든 코인의 price 데이터를 RevIN 정규화 후 통합
    2. 코인별 균등 샘플링 (BTC 지배력 방지)
    3. 공유 PatchEmbed + MAE 학습 → 크립토 시계열 공통 구조 습득
    4. 체크포인트 저장 → per-coin finetune에서 로드

    Usage:
        fp = FoundationPretrainer(num_price_features=51, seq_len=48)
        result = fp.pretrain(all_coins_price_data)
        fp.save("models/foundation_v1.pt")
        # 이후 MultiModalTrainer.load_foundation("models/foundation_v1.pt")
    """

    CHECKPOINT_DIR = Path("models/foundation")

    def __init__(self, num_price_features: int, seq_len: int = 48,
                 d_model: int = 64, num_layers: int = 2, dropout: float = 0.3):
        self.num_price_features = num_price_features
        self.seq_len = seq_len
        self.d_model = d_model

        # Foundation 모델 (price-only, 분류 헤드 없음)
        self.model = MultiModalCryptoClassifier(
            num_price_features=num_price_features,
            num_media_features=1,  # placeholder — pretrain은 price만 사용
            seq_len=seq_len, d_model=d_model, nhead=4,
            num_layers=num_layers, num_classes=3, dropout=dropout,
        ).to(device)

        try:
            self.model = torch.compile(self.model, mode="default", backend="eager")
        except Exception:
            pass

        self.amp_scaler = GradScaler("cuda")
        self.use_amp = device.type == "cuda"
        self.price_scaler = RobustScaler()

    def pretrain(
        self,
        coins_price_data: dict[str, np.ndarray],
        epochs: int = 50,
        batch_size: int = 512,
        lr: float = 3e-3,
    ) -> dict:
        """전 코인 통합 MAE Pretrain.

        Args:
            coins_price_data: {coin_name: price_array (N, num_features)}
            epochs: pretrain 에폭 (기본 50 — 데이터 10배이므로 per-coin 30보다 높게)
            batch_size: 배치 크기
            lr: 학습률

        Returns:
            dict: pretrain 결과 메트릭
        """
        print("\n  === FOUNDATION PRETRAIN: All Coins Unified MAE ===")

        if not coins_price_data:
            return {"status": "no_data"}

        # ── Step 1: 전 코인 데이터 통합 + 균등 샘플링 ──
        all_sequences = []
        coin_stats = {}

        # 전체 데이터로 scaler fit
        all_raw = np.vstack(list(coins_price_data.values()))
        all_raw = np.nan_to_num(all_raw, nan=0, posinf=0, neginf=0)
        self.price_scaler.fit(all_raw)

        for coin, price_arr in coins_price_data.items():
            scaled = self.price_scaler.transform(
                np.nan_to_num(price_arr, nan=0, posinf=0, neginf=0))
            tensor = torch.FloatTensor(scaled)

            # 슬라이딩 윈도우로 시퀀스 추출
            n_seqs = max(0, len(tensor) - self.seq_len)
            seqs = [tensor[i:i + self.seq_len] for i in range(n_seqs)]
            all_sequences.extend(seqs)
            coin_stats[coin] = n_seqs

        total_seqs = len(all_sequences)
        if total_seqs < 100:
            print(f"  [Foundation] Insufficient data: {total_seqs} sequences")
            return {"status": "insufficient_data", "total_sequences": total_seqs}

        print(f"  Data: {len(coins_price_data)} coins, {total_seqs:,} sequences")
        for coin, n in coin_stats.items():
            print(f"    {coin}: {n:,} seqs")

        # ── Step 2: Pretrain 설정 ──
        pretrain_params = (
            list(self.model.price_patch_embed.parameters()) +
            list(self.model.masked_ae.parameters()) +
            list(self.model.revin_price.parameters())
        )
        optimizer = optim.AdamW(pretrain_params, lr=lr, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        best_loss = float("inf")
        loss_history = []

        # ── Step 3: Training Loop ──
        print(f"  Training: {epochs} epochs, batch={batch_size}, lr={lr}")

        for epoch in range(epochs):
            self.model.train()
            epoch_loss = 0
            batches = 0

            # 에폭마다 셔플
            indices = np.random.permutation(total_seqs)

            for start in range(0, len(indices) - batch_size, batch_size):
                batch_idx = indices[start:start + batch_size]
                price_batch = torch.stack(
                    [all_sequences[i] for i in batch_idx]
                ).to(device)

                optimizer.zero_grad()

                if self.use_amp:
                    with autocast("cuda"):
                        recon_loss, per_patch_loss = self.model.forward_pretrain(
                            price_batch, epoch, epochs)
                    self.amp_scaler.scale(recon_loss).backward()
                    self.amp_scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(pretrain_params, 1.0)
                    self.amp_scaler.step(optimizer)
                    self.amp_scaler.update()
                else:
                    recon_loss, per_patch_loss = self.model.forward_pretrain(
                        price_batch, epoch, epochs)
                    recon_loss.backward()
                    torch.nn.utils.clip_grad_norm_(pretrain_params, 1.0)
                    optimizer.step()

                epoch_loss += recon_loss.item()
                batches += 1

            scheduler.step()
            avg_loss = epoch_loss / max(batches, 1)
            loss_history.append(avg_loss)

            if avg_loss < best_loss:
                best_loss = avg_loss

            if (epoch + 1) % 5 == 0:
                masking = self.model.selective_masking
                phase = ("random" if epoch < epochs * 0.3
                         else "hard-mining" if epoch < epochs * 0.7
                         else "curriculum")
                print(f"    Epoch {epoch+1:3d}/{epochs}: "
                      f"loss={avg_loss:.6f}, best={best_loss:.6f}, "
                      f"phase={phase}, batches={batches}")

        improvement = (loss_history[0] - loss_history[-1]) / (loss_history[0] + 1e-10) * 100
        print(f"\n  Foundation Pretrain Done:")
        print(f"    Loss: {loss_history[0]:.4f} → {loss_history[-1]:.4f} ({improvement:.1f}% ↓)")
        print(f"    Best: {best_loss:.4f}")
        print(f"    Sequences: {total_seqs:,} from {len(coins_price_data)} coins")

        return {
            "status": "completed",
            "epochs": epochs,
            "initial_loss": round(loss_history[0], 4),
            "final_loss": round(loss_history[-1], 4),
            "best_loss": round(best_loss, 4),
            "improvement_pct": round(improvement, 1),
            "total_sequences": total_seqs,
            "coins": list(coins_price_data.keys()),
            "coin_sequences": coin_stats,
        }

    def save(self, path: str | None = None) -> str:
        """Foundation 체크포인트 저장.

        저장 내용: PatchEmbed + MAE + RevIN(price) 가중치 + scaler
        """
        if path is None:
            self.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            path = str(self.CHECKPOINT_DIR / f"foundation_{ts}.pt")

        # state_dict에서 foundation 파라미터만 필터
        # torch.compile은 _orig_mod. 접두사를 추가하므로 양쪽 모두 매칭
        FOUNDATION_PREFIXES = [
            "price_patch_embed.", "masked_ae.", "revin_price.",
            "_orig_mod.price_patch_embed.", "_orig_mod.masked_ae.", "_orig_mod.revin_price.",
        ]
        full_state = self.model.state_dict()
        foundation_state = {k: v for k, v in full_state.items()
                           if any(k.startswith(p) for p in FOUNDATION_PREFIXES)}

        checkpoint = {
            "model_state": foundation_state,
            "price_scaler": self.price_scaler,
            "num_price_features": self.num_price_features,
            "seq_len": self.seq_len,
            "d_model": self.d_model,
            "timestamp": datetime.now().isoformat(),
        }

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint, path)
        print(f"  Foundation saved: {path} ({len(foundation_state)} params)")
        return path

    @classmethod
    def load_latest(cls) -> str | None:
        """가장 최근 foundation 체크포인트 경로 반환."""
        if not cls.CHECKPOINT_DIR.exists():
            return None
        checkpoints = sorted(cls.CHECKPOINT_DIR.glob("foundation_*.pt"))
        return str(checkpoints[-1]) if checkpoints else None
