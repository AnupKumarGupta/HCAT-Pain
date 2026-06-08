import torch
import torch.nn as nn
import torch.nn.functional as F


def generate_positional_encoding(max_len: int, d_model: int) -> torch.Tensor:
    """
    Standard sinusoidal positional encoding. Returns (max_len, d_model).
    """
    pos = torch.arange(0, max_len).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, d_model, 2) * (-torch.log(torch.tensor(10000.0)) / d_model)
    )
    pe = torch.zeros(max_len, d_model)
    pe[:, 0::2] = torch.sin(pos * div_term)
    pe[:, 1::2] = torch.cos(pos * div_term)
    return pe


class ModalitySpecificTransformer(nn.Module):
    """
    Projects a single modality into its own internal embedding space, adds positional encoding, encodes temporal
    dynamics with a Transformer encoder, then applies a final MLP that projects the representation to the
    SHARED output dimension.

    Input shape  : (B, F, C)
    Output shape : (B, F, output_dim)
    """

    def __init__(self, input_dim: int, model_dim: int, output_dim: int, num_heads: int, num_layers: int,
                 dim_feedforward: int, dropout: float, max_seq_length: int, num_classes: int = 3):
        super().__init__()

        self.proj = nn.Linear(input_dim, model_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="relu",
            batch_first=True,
            norm_first=False,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        pe = generate_positional_encoding(max_seq_length, model_dim)
        self.register_buffer("pos_encoding", pe)

        self.output_mlp = nn.Sequential(
            nn.Linear(model_dim, output_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(output_dim, output_dim),
        )

        self.aux_head = nn.Linear(output_dim, num_classes)

    def forward(self, x: torch.Tensor):
        z = self.proj(x)
        pe = self.pos_encoding[: z.size(1)].unsqueeze(0)
        z = z + pe
        z = self.transformer(z)
        H = self.output_mlp(z)

        pooled = H.mean(dim=1)
        logits = self.aux_head(pooled)
        return H, logits


class SymmetricCrossAttentionBlock(nn.Module):
    """
    One cross-attention layer: each modality attends to the other.
        H_{phys -> vis} = T_{cross-phys}(Q_phys^ca,  K_vis^ca,  V_vis^ca)
        H_{vis -> phys} = T_{cross-vis} (Q_vis^ca,   K_phys^ca, V_phys^ca)
    """

    def __init__(self, model_dim: int, num_heads: int, dropout: float):
        super().__init__()

        self.ca_phys_to_vis = nn.MultiheadAttention(
            model_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.ca_vis_to_phys = nn.MultiheadAttention(
            model_dim, num_heads, dropout=dropout, batch_first=True
        )

        # Post-attention layer norms
        self.norm_phys = nn.LayerNorm(model_dim)
        self.norm_vis = nn.LayerNorm(model_dim)

        # Feed-forward sub-layers
        self.ffn_hidden_expansion = 4
        self.ffn_phys = nn.Sequential(
            nn.Linear(model_dim, model_dim * self.ffn_hidden_expansion),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(model_dim * self.ffn_hidden_expansion, model_dim),
            nn.Dropout(dropout),
        )
        self.ffn_vis = nn.Sequential(
            nn.Linear(model_dim, model_dim * self.ffn_hidden_expansion),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(model_dim * self.ffn_hidden_expansion, model_dim),
            nn.Dropout(dropout),
        )

        self.norm_phys2 = nn.LayerNorm(model_dim)
        self.norm_vis2 = nn.LayerNorm(model_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, h_phys: torch.Tensor, h_vis: torch.Tensor):
        phys_attn, _ = self.ca_phys_to_vis(
            query=h_phys, key=h_vis, value=h_vis
        )
        vis_attn, _ = self.ca_vis_to_phys(
            query=h_vis, key=h_phys, value=h_phys
        )

        h_phys = self.norm_phys(h_phys + self.dropout(phys_attn))
        h_phys = self.norm_phys2(h_phys + self.ffn_phys(h_phys))

        h_vis = self.norm_vis(h_vis + self.dropout(vis_attn))
        h_vis = self.norm_vis2(h_vis + self.ffn_vis(h_vis))

        return h_phys, h_vis


class SymmetricCrossAttentionTransformer(nn.Module):
    """
    Stacks `num_layers` SymmetricCrossAttentionBlocks.
    Input/Output shapes: same as SymmetricCrossAttentionBlock.
    """

    def __init__(self, model_dim: int, num_heads: int, num_layers: int, dropout: float):
        super().__init__()
        self.layers = nn.ModuleList(
            [SymmetricCrossAttentionBlock(model_dim, num_heads, dropout)
             for _ in range(num_layers)]
        )

    def forward(self, h_phys: torch.Tensor, h_vis: torch.Tensor):
        for layer in self.layers:
            h_phys, h_vis = layer(h_phys, h_vis)
        return h_phys, h_vis


class FusionTransformer(nn.Module):
    """
    Concatenates the two cross-attended embeddings along the time axis, projects into the fusion embedding space,
    applies a Transformer encoder, mean-pools, then classifies via an MLP with softmax.

        X_fusion = [H_{phys->vis} ; H_{vis->phys}]   (B, 2F, d_cross)
        H_fusion = T_fusion(X_fusion)                (B, 2F, d_fusion)
        h_final  = mean_pool(H_fusion)               (B, d_fusion)
        y        = softmax(MLP(h_final))             (B, num_classes)
    """

    def __init__(self, cross_dim: int, fusion_dim: int, num_heads: int, num_layers: int, dim_feedforward: int,
                 dropout: float, num_classes: int, mlp_hidden_dim: int = 256):
        super().__init__()

        self.input_proj = nn.Linear(cross_dim, fusion_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=fusion_dim,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="relu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.mlp = nn.Sequential(
            nn.Linear(fusion_dim, mlp_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, num_classes),
        )

    def forward(self, h_phys_to_vis: torch.Tensor, h_vis_to_phys: torch.Tensor):
        x_fusion = torch.cat([h_phys_to_vis, h_vis_to_phys], dim=1)
        x_fusion = self.input_proj(x_fusion)
        h_fusion = self.transformer(x_fusion)

        h_final = h_fusion.mean(dim=1)
        logits = self.mlp(h_final)
        return logits


class HCAT(nn.Module):
    """
    Hierarchical Cross-Attention Transformer (HCAT) for non-contact
    multimodal pain classification.

    Args:
        phys_input_dim   : Channels for physiological modality
        vis_input_dim    : Channels for visual modality
        phys_model_dim   : embedding dim for T_phys
        vis_model_dim    : embedding dim for T_vis
        cross_model_dim  : embedding dim for cross-attn stage
        fusion_model_dim : embedding dim for fusion Transformer
        num_heads        : attention heads
        num_layers_phys  : layers in T_phys
        num_layers_vis   : layers in T_vis
        num_layers_cross : layers per cross-attention block
        num_layers_fusion: layers in T_fusion
        dim_feedforward  : FFN hidden dim inside Transformers
        dropout          : dropout rate
        num_classes      : number of classes
        max_seq_length   : maximum frame count
    """

    def __init__(self, phys_input_dim: int = 2, vis_input_dim: int = 17, phys_model_dim: int = 128,
                 vis_model_dim: int = 256, cross_model_dim: int = 256, fusion_model_dim: int = 512,
                 num_heads: int = 4, num_layers_phys: int = 2, num_layers_vis: int = 2,
                 num_layers_cross: int = 4, num_layers_fusion: int = 2, dim_feedforward: int = 1024,
                 dropout: float = 0.1, num_classes: int = 3, max_seq_length: int = 270, mlp_hidden_dim: int = 256):
        super().__init__()

        # Modality-Specific Transformers for physiological and visual features
        self.t_phys = ModalitySpecificTransformer(
            input_dim=phys_input_dim,
            model_dim=phys_model_dim,
            output_dim=cross_model_dim,
            num_heads=num_heads,
            num_layers=num_layers_phys,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            max_seq_length=max_seq_length,
            num_classes=num_classes
        )

        self.t_vis = ModalitySpecificTransformer(
            input_dim=vis_input_dim,
            model_dim=vis_model_dim,
            output_dim=cross_model_dim,
            num_heads=num_heads,
            num_layers=num_layers_vis,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            max_seq_length=max_seq_length,
            num_classes=num_classes
        )

        # Symmetric Cross-Attention Transformer
        self.cross_transformer = SymmetricCrossAttentionTransformer(
            model_dim=cross_model_dim,
            num_heads=num_heads,
            num_layers=num_layers_cross,
            dropout=dropout
        )

        # Fusion Transformer
        self.fusion_transformer = FusionTransformer(
            cross_dim=cross_model_dim,
            fusion_dim=fusion_model_dim,
            num_heads=num_heads,
            num_layers=num_layers_fusion,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            num_classes=num_classes,
            mlp_hidden_dim=mlp_hidden_dim,
        )

    def forward(self, x_phys: torch.Tensor, x_vis: torch.Tensor):
        h_phys, logits_phys = self.t_phys(x_phys)
        h_vis, logits_vis = self.t_vis(x_vis)
        h_phys_to_vis, h_vis_to_phys = self.cross_transformer(h_phys, h_vis)
        logits_fusion = self.fusion_transformer(h_phys_to_vis, h_vis_to_phys)
        return logits_fusion, logits_phys, logits_vis, h_phys, h_vis


class OrdinalWeightedCrossEntropy(nn.Module):
    """
    Ordinal Weighted Cross-Entropy (OWCE) loss.

    Penalises misclassification proportionally to the ordinal distance between the predicted and true class,
    and compensates for class imbalance via per-class weights.

    Args:
        class_counts : list/tensor of per-class sample counts
        phi          : ordinal penalty scale
        alpha        : ordinal penalty exponent
        num_classes  : C
    """

    def __init__(self, class_counts: list, phi: float = 1.0, alpha: float = 1.0, num_classes: int = 3):
        super().__init__()
        assert len(class_counts) == num_classes, f"class_counts has {len(class_counts)} entries, expected {num_classes}"
        counts = torch.tensor(class_counts, dtype=torch.float)
        weights = counts.sum() / counts
        self.register_buffer("weights", weights)
        self.phi = phi
        self.alpha = alpha
        self.num_classes = num_classes

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        preds = logits.argmax(dim=-1)

        wce = F.cross_entropy(logits, targets, weight=self.weights, reduction="none")
        # reduction = 'none' kept on purpose so as the output is a vector.

        ordinal_dist = (targets - preds).abs().float()
        p = self.phi * (ordinal_dist / (self.num_classes - 1)) ** self.alpha

        loss = ((1 + p) * wce).mean()
        return loss


class HCATLoss(nn.Module):
    """
    L_total = alpha·L_DF^phys + beta·L_DF^vis + gamma·L_DF^fusion + lambda·L_CM

    Args:
        owce_kwargs : passed to OrdinalWeightedCrossEntropy
        alpha, beta, gamma, lambda : loss weighting coefficients
    """

    def __init__(self, class_counts: list, alpha: float = 0.7, beta: float = 0.7, gamma: float = 1.0, lam: float = 0.5,
                 num_classes: int = 3):
        super().__init__()
        self.owce = OrdinalWeightedCrossEntropy(class_counts=class_counts, num_classes=num_classes)
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.lam = lam

    def forward(self, logits_fusion: torch.Tensor, logits_phys: torch.Tensor, logits_vis: torch.Tensor,
                h_phys: torch.Tensor, h_vis: torch.Tensor, targets: torch.Tensor) -> dict:
        l_phys = self.owce(logits_phys, targets)
        l_vis = self.owce(logits_vis, targets)
        l_fusion = self.owce(logits_fusion, targets)

        l_cm = F.mse_loss(h_phys, h_vis)

        total = self.alpha * l_phys + self.beta * l_vis + self.gamma * l_fusion + self.lam * l_cm

        return {
            "loss": total,
            "l_phys": l_phys.detach(),
            "l_vis": l_vis.detach(),
            "l_fusion": l_fusion.detach(),
            "l_cm": l_cm.detach(),
        }


if __name__ == "__main__":
    model = HCAT(phys_input_dim=2, vis_input_dim=17, phys_model_dim=128, vis_model_dim=256, cross_model_dim=256,
                 fusion_model_dim=512, num_heads=4, num_layers_phys=2, num_layers_vis=2, num_layers_cross=4,
                 num_layers_fusion=2, dim_feedforward=1024, dropout=0.1, num_classes=3, max_seq_length=270)

    B, T = 4, 270
    x_phys = torch.randn(B, T, 2)
    x_vis = torch.randn(B, T, 17)
    logits_fusion, logits_phys, logits_vis, h_phys, h_vis = model(x_phys, x_vis)
    targets = torch.randint(0, 3, (B,))
    criterion = HCATLoss(class_counts=[100, 120, 120], num_classes=3)
    losses = criterion(logits_fusion, logits_phys, logits_vis, h_phys, h_vis, targets)

    print(f"logits_fusion shape: {logits_fusion.shape}")
    print(f"total loss: {losses['loss'].item():.4f}")
