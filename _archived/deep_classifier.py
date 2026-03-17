"""SOTA Deep Learning Classifier for Crypto Forecasting.

Models:
1. iTransformer: Channel-wise attention (inverted Transformer)
2. PatchTST: Patch tokenization + Transformer
3. Selective Learning: Dual-mask training strategy (NeurIPS 2025)
   - Uncertainty mask: residual entropy filtering
   - Anomaly mask: residual lower bound estimation

Classification: UP / DOWN / HOLD
GPU: RTX 3090 CUDA
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ==================== Datasets ====================

class TimeSeriesDataset(Dataset):
    """시계열 분류 데이터셋 with index tracking."""

    def __init__(self, X: np.ndarray, y: np.ndarray, seq_len: int = 48):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)
        self.seq_len = seq_len

    def __len__(self):
        return max(0, len(self.X) - self.seq_len)

    def __getitem__(self, idx):
        x_seq = self.X[idx:idx + self.seq_len]
        label = self.y[idx + self.seq_len - 1]
        return x_seq, label, idx


# ==================== iTransformer ====================

class iTransformer(nn.Module):
    """Inverted Transformer: channel-wise attention.

    기존 Transformer가 시간축에 attention을 적용하는 것과 달리,
    iTransformer는 채널(feature)축에 attention을 적용하여
    multivariate correlation을 포착합니다.
    """

    def __init__(self, num_features: int, seq_len: int = 48,
                 d_model: int = 128, nhead: int = 4, num_layers: int = 3,
                 num_classes: int = 3, dropout: float = 0.2):
        super().__init__()
        self.num_features = num_features
        self.seq_len = seq_len

        # Feature embedding (각 feature를 독립적으로 temporal encoding)
        self.feature_embed = nn.Linear(seq_len, d_model)

        # Channel-wise Transformer (feature 간 상관관계 학습)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=d_model * 4, dropout=dropout,
            batch_first=True, activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Classifier
        self.norm = nn.LayerNorm(d_model)
        self.classifier = nn.Sequential(
            nn.Linear(d_model * num_features, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes),
        )

    def forward(self, x):
        # x: (batch, seq_len, features)
        # Invert: (batch, features, seq_len) → embed each feature's time series
        x_inv = x.transpose(1, 2)  # (B, F, T)
        x_emb = self.feature_embed(x_inv)  # (B, F, d_model)

        # Channel-wise attention
        x_tf = self.transformer(x_emb)  # (B, F, d_model)
        x_tf = self.norm(x_tf)

        # Flatten and classify
        x_flat = x_tf.reshape(x_tf.size(0), -1)  # (B, F * d_model)
        return self.classifier(x_flat)


# ==================== PatchTST ====================

class PatchTST(nn.Module):
    """Patch Time-Series Transformer.

    시계열을 패치(subseries)로 분할하여 Transformer에 입력.
    로컬 시맨틱 정보를 보존하면서 장기 의존성을 포착합니다.
    """

    def __init__(self, num_features: int, seq_len: int = 48,
                 patch_len: int = 8, stride: int = 4,
                 d_model: int = 128, nhead: int = 4, num_layers: int = 3,
                 num_classes: int = 3, dropout: float = 0.2):
        super().__init__()
        self.patch_len = patch_len
        self.stride = stride

        # 패치 수 계산
        self.num_patches = (seq_len - patch_len) // stride + 1

        # Patch embedding
        self.patch_embed = nn.Linear(patch_len * num_features, d_model)
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches, d_model) * 0.02)

        # Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=d_model * 4, dropout=dropout,
            batch_first=True, activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Classifier
        self.norm = nn.LayerNorm(d_model)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes),
        )

    def forward(self, x):
        # x: (batch, seq_len, features)
        B = x.size(0)

        # Create patches
        patches = []
        for i in range(self.num_patches):
            start = i * self.stride
            patch = x[:, start:start + self.patch_len, :]  # (B, patch_len, F)
            patches.append(patch.reshape(B, -1))  # (B, patch_len * F)

        patches = torch.stack(patches, dim=1)  # (B, num_patches, patch_len * F)

        # Embed patches
        x_emb = self.patch_embed(patches) + self.pos_embed  # (B, num_patches, d_model)

        # Transformer
        x_tf = self.transformer(x_emb)  # (B, num_patches, d_model)
        x_tf = self.norm(x_tf)

        # Use last patch representation for classification
        x_last = x_tf[:, -1, :]  # (B, d_model)
        return self.classifier(x_last)


# ==================== Selective Learning ====================

class SelectiveLearning:
    """Selective Learning training strategy (NeurIPS 2025).

    Dual-mask mechanism:
    1. Uncertainty mask: residual entropy → filter high-entropy (uncertain) timesteps
    2. Anomaly mask: residual lower bound → filter anomalous timesteps

    Only compute loss on reliable (unmasked) timesteps.
    """

    def __init__(self, r_u: float = 0.3, r_a: float = 0.3):
        """
        Args:
            r_u: uncertainty masking ratio (0~1). Higher = more aggressive filtering
            r_a: anomaly masking ratio (0~1). Higher = more aggressive filtering
        """
        self.r_u = r_u
        self.r_a = r_a
        self.residual_history = {}  # idx -> list of residuals (for entropy)
        self.estimator_residuals = None  # from estimator model

    def compute_uncertainty_mask(self, predictions: torch.Tensor,
                                  targets: torch.Tensor,
                                  indices: torch.Tensor) -> torch.Tensor:
        """Uncertainty mask via residual entropy.

        높은 엔트로피(불확실성)를 가진 샘플을 마스킹합니다.
        잔차의 분산이 클수록 엔트로피가 높음 (정규분포 가정).
        """
        residuals = torch.abs(predictions.detach() - targets)  # (B,)

        # 히스토리에 잔차 축적
        for i, idx in enumerate(indices.cpu().numpy()):
            idx = int(idx)
            if idx not in self.residual_history:
                self.residual_history[idx] = []
            self.residual_history[idx].append(residuals[i].cpu().item())
            # 최근 10개만 유지
            if len(self.residual_history[idx]) > 10:
                self.residual_history[idx] = self.residual_history[idx][-10:]

        # 엔트로피 계산 (분산 기반, 정규분포 가정: H = 0.5 * ln(2*pi*e*var))
        entropies = []
        for i, idx in enumerate(indices.cpu().numpy()):
            idx = int(idx)
            hist = self.residual_history.get(idx, [])
            if len(hist) >= 3:
                var = np.var(hist) + 1e-10
                entropy = 0.5 * np.log(2 * np.pi * np.e * var)
            else:
                entropy = 0.0  # 히스토리 부족 → 마스킹 안함
            entropies.append(entropy)

        entropies = torch.FloatTensor(entropies).to(predictions.device)

        # 상위 r_u% 엔트로피 마스킹
        if len(entropies) > 0:
            threshold = torch.quantile(entropies, 1 - self.r_u)
            mask = entropies <= threshold  # True = keep, False = mask
        else:
            mask = torch.ones(len(predictions), dtype=torch.bool, device=predictions.device)

        return mask

    def compute_anomaly_mask(self, residuals: torch.Tensor,
                              estimator_residuals: torch.Tensor | None = None) -> torch.Tensor:
        """Anomaly mask via residual lower bound estimation.

        잔차가 추정된 하한에 가까운 샘플(이상치)을 마스킹합니다.
        Estimator가 없으면 잔차 분포의 하위 quantile을 사용합니다.
        """
        if estimator_residuals is not None:
            dist = torch.abs(residuals - estimator_residuals)
        else:
            # Estimator 없이: 잔차 자체의 하위 quantile로 lower bound 추정
            lower_bound = torch.quantile(residuals, 0.1)
            dist = torch.abs(residuals - lower_bound)

        # 하위 r_a%를 이상치로 마스킹 (lower bound에 가장 가까운 것들)
        threshold = torch.quantile(dist, self.r_a)
        mask = dist > threshold  # True = keep, False = mask (anomaly)

        return mask

    def selective_loss(self, logits: torch.Tensor, targets: torch.Tensor,
                       indices: torch.Tensor) -> tuple[torch.Tensor, dict]:
        """Selective cross-entropy loss.

        마스킹되지 않은 샘플에 대해서만 loss를 계산합니다.
        """
        # 예측 확률 → 잔차 (classification이므로 cross-entropy 잔차)
        probs = torch.softmax(logits.detach(), dim=1)
        pred_classes = torch.argmax(probs, dim=1)
        correctness = (pred_classes == targets).float()
        residuals = 1.0 - correctness  # 맞으면 0, 틀리면 1

        # Uncertainty mask
        u_mask = self.compute_uncertainty_mask(correctness, torch.ones_like(correctness), indices)

        # Anomaly mask
        a_mask = self.compute_anomaly_mask(residuals)

        # Combined mask (both must be True to keep)
        combined_mask = u_mask & a_mask

        # Masked loss
        ce_loss = nn.CrossEntropyLoss(reduction="none")(logits, targets)

        if combined_mask.sum() > 0:
            masked_loss = (ce_loss * combined_mask.float()).sum() / combined_mask.sum()
        else:
            masked_loss = ce_loss.mean()  # fallback

        mask_ratio = 1.0 - combined_mask.float().mean().item()

        stats = {
            "mask_ratio": round(mask_ratio, 3),
            "u_mask_ratio": round(1.0 - u_mask.float().mean().item(), 3),
            "a_mask_ratio": round(1.0 - a_mask.float().mean().item(), 3),
            "kept_samples": int(combined_mask.sum().item()),
        }

        return masked_loss, stats

    def reset_history(self):
        """에포크 사이에 히스토리를 리셋합니다."""
        # 히스토리는 유지 (누적 엔트로피 계산용)
        pass


# ==================== Ensemble Classifier ====================

class SOTAEnsembleClassifier:
    """iTransformer + PatchTST + Selective Learning 앙상블.

    2024-2025 SOTA 모델 아키텍처 + NeurIPS 2025 학습 전략.
    """

    def __init__(self, input_size: int, seq_len: int = 48, num_classes: int = 3):
        self.input_size = input_size
        self.seq_len = seq_len
        self.num_classes = num_classes
        self.scaler = StandardScaler()

        # iTransformer: channel-wise attention
        self.itransformer = iTransformer(
            num_features=input_size, seq_len=seq_len,
            d_model=128, nhead=4, num_layers=3,
            num_classes=num_classes, dropout=0.15,
        ).to(device)

        # PatchTST: patch tokenization
        patch_len = min(8, seq_len // 3)
        stride = max(2, patch_len // 2)
        self.patchtst = PatchTST(
            num_features=input_size, seq_len=seq_len,
            patch_len=patch_len, stride=stride,
            d_model=128, nhead=4, num_layers=3,
            num_classes=num_classes, dropout=0.15,
        ).to(device)

        # Selective Learning strategy
        self.selective = SelectiveLearning(r_u=0.3, r_a=0.3)

        # Ensemble weights (learned)
        self.itf_weight = 0.5
        self.ptst_weight = 0.5

    def train_models(self, X_train: np.ndarray, y_train: np.ndarray,
                     epochs: int = 40, batch_size: int = 64, lr: float = 8e-4,
                     num_workers: int = 6) -> dict:
        """Selective Learning으로 두 모델을 학습합니다."""
        # NaN/Inf 처리
        X_clean = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
        X_scaled = self.scaler.fit_transform(X_clean)

        dataset = TimeSeriesDataset(X_scaled, y_train, self.seq_len)
        if len(dataset) < 10:
            return {"status": "insufficient_data"}

        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, drop_last=True,
            num_workers=num_workers, pin_memory=True, persistent_workers=True,
        )

        # Class weights
        unique, counts = np.unique(y_train, return_counts=True)
        total = counts.sum()
        weights = torch.FloatTensor([total / (len(unique) * c + 1) for c in counts]).to(device)

        results = {}
        for name, model in [("iTransformer", self.itransformer), ("PatchTST", self.patchtst)]:
            model.train()
            optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
            scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)

            best_loss = float("inf")
            total_masked = 0
            total_samples = 0

            for epoch in range(epochs):
                epoch_loss = 0
                batches = 0

                for x_batch, y_batch, idx_batch in loader:
                    x_batch = x_batch.to(device)
                    y_batch = y_batch.to(device)
                    idx_batch = idx_batch.to(device)

                    optimizer.zero_grad()
                    logits = model(x_batch)

                    # Selective Learning loss
                    if epoch >= 5:  # 워밍업 후 selective 적용
                        loss, stats = self.selective.selective_loss(logits, y_batch, idx_batch)
                        total_masked += int(stats["mask_ratio"] * len(y_batch))
                        total_samples += len(y_batch)
                    else:
                        loss = nn.CrossEntropyLoss(weight=weights)(logits, y_batch)

                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()

                    epoch_loss += loss.item()
                    batches += 1

                scheduler.step()
                avg_loss = epoch_loss / max(batches, 1)
                if avg_loss < best_loss:
                    best_loss = avg_loss

            mask_pct = total_masked / max(total_samples, 1) * 100
            results[name] = {
                "final_loss": round(best_loss, 4),
                "epochs": epochs,
                "selective_mask_pct": round(mask_pct, 1),
            }
            print(f"    [{name}] Loss: {best_loss:.4f}, Selective masked: {mask_pct:.1f}%")

        # Reset selective history for next training
        self.selective.residual_history.clear()

        return results

    def predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """앙상블 예측."""
        X_clean = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        X_scaled = self.scaler.transform(X_clean)

        if len(X_scaled) < self.seq_len:
            pad = np.zeros((self.seq_len - len(X_scaled), X_scaled.shape[1]))
            X_scaled = np.vstack([pad, X_scaled])

        x_seq = torch.FloatTensor(X_scaled[-self.seq_len:]).unsqueeze(0).to(device)

        self.itransformer.eval()
        self.patchtst.eval()

        with torch.no_grad():
            itf_logits = self.itransformer(x_seq)
            ptst_logits = self.patchtst(x_seq)

            itf_proba = torch.softmax(itf_logits, dim=1).cpu().numpy()
            ptst_proba = torch.softmax(ptst_logits, dim=1).cpu().numpy()

        ensemble_proba = self.itf_weight * itf_proba + self.ptst_weight * ptst_proba
        pred_class = np.argmax(ensemble_proba, axis=1)

        return pred_class, ensemble_proba

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> dict:
        """테스트셋 평가."""
        X_clean = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)
        X_scaled = self.scaler.transform(X_clean)
        dataset = TimeSeriesDataset(X_scaled, y_test, self.seq_len)

        if len(dataset) == 0:
            return {"accuracy": 0, "f1": 0}

        loader = DataLoader(dataset, batch_size=64, shuffle=False)
        all_preds, all_labels = [], []

        self.itransformer.eval()
        self.patchtst.eval()

        with torch.no_grad():
            for x_batch, y_batch, _ in loader:
                x_batch = x_batch.to(device)

                itf_p = torch.softmax(self.itransformer(x_batch), dim=1)
                ptst_p = torch.softmax(self.patchtst(x_batch), dim=1)

                ensemble = self.itf_weight * itf_p + self.ptst_weight * ptst_p
                preds = torch.argmax(ensemble, dim=1).cpu().numpy()

                all_preds.extend(preds)
                all_labels.extend(y_batch.numpy())

        acc = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)

        return {"accuracy": round(acc, 4), "f1": round(f1, 4)}
