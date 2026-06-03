import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm import Mamba

class EventTokenizerSessions(nn.Module):
    def __init__(self, n_event_types, n_products, d_model,
                 event_emb_dim=64, prod_emb_dim=128, num_proj_dim=64, input_num_dim=4):
        super().__init__()

        self.input_num_dim = input_num_dim

        # 1. Embeddings Categóricos
        self.event_emb = nn.Embedding(n_event_types + 1, event_emb_dim, padding_idx=0)
        self.prod_emb  = nn.Embedding(n_products + 1, prod_emb_dim, padding_idx=0)

        # 2. Proyección Numérica Dinámica para 4 variables
        self.num_projection = nn.Sequential(
            nn.Linear(self.input_num_dim, num_proj_dim),
            nn.SiLU(),
            nn.Linear(num_proj_dim, num_proj_dim)
        )

        # 3. Fusión Final
        total_input_dim = event_emb_dim + prod_emb_dim + num_proj_dim
        self.fusion = nn.Sequential(
            nn.Linear(total_input_dim, d_model),
            nn.LayerNorm(d_model),
            nn.Dropout(0.1)
        )

    def forward(self, event_type, product_id, num_feats):
        # Tomamos exactamente las variables que necesitamos
        num_feats = num_feats[:, :, :self.input_num_dim]

        e = self.event_emb(event_type)
        p = self.prod_emb(product_id)
        n = self.num_projection(num_feats)

        combined = torch.cat([e, p, n], dim=-1)
        return self.fusion(combined)

class MambaModelSessions(nn.Module):
    def __init__(self, n_event_types, n_products, d_model=128, d_state=32, d_conv=4,
                 event_emb_dim=64, prod_emb_dim=128, num_proj_dim=64, input_num_dim=4):
        super().__init__()

        # Usar el Tokenizer de Sesiones
        self.tokenizer = EventTokenizerSessions(
            n_event_types=n_event_types,
            n_products=n_products,
            d_model=d_model,
            event_emb_dim=event_emb_dim,
            prod_emb_dim=prod_emb_dim,
            num_proj_dim=num_proj_dim,
            input_num_dim=input_num_dim
        )

        self.backbone = Mamba(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=2,
        )

        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_model, n_event_types + 1),
        )

        self.contrastive_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )

    def forward(self, event_type, product_id, num_feats, return_embeddings=False):
        x = self.tokenizer(event_type, product_id, num_feats)
        h = self.backbone(x)
        logits = self.head(h)

        if return_embeddings:
            user_emb = F.normalize(self.contrastive_proj(h[:, -1, :]), dim=-1)
            return logits, user_emb

        return logits
