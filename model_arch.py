"""
model_arch.py
─────────────
Definiciones de arquitectura del modelo Mamba.
Separadas de Mamba.py para poder importarlas en scripts de inferencia
sin ejecutar el código de entrenamiento/configuración.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm import Mamba

class EventTokenizer(nn.Module):
    def __init__(self, n_event_types, n_products, d_model,
                 use_balance=True,
                 event_emb_dim=64, prod_emb_dim=128, num_proj_dim=64):
        super().__init__()

        self.use_balance = use_balance

        # 1. Embeddings Categóricos
        self.event_emb = nn.Embedding(n_event_types + 1, event_emb_dim, padding_idx=0)
        self.prod_emb  = nn.Embedding(n_products + 1, prod_emb_dim, padding_idx=0)

        # 2. Proyección Numérica Dinámica
        input_num_dim = 3 if use_balance else 2
        self.num_projection = nn.Sequential(
            nn.Linear(input_num_dim, num_proj_dim),
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
        if not self.use_balance:
            num_feats = num_feats[:, :, :2]

        e = self.event_emb(event_type)
        p = self.prod_emb(product_id)
        n = self.num_projection(num_feats)

        combined = torch.cat([e, p, n], dim=-1)
        return self.fusion(combined)

class MambaModelMultiTask(nn.Module):
    def __init__(self, n_event_types, n_products, d_model=128, d_state=32, d_conv=4,
                 use_balance=True, event_emb_dim=64, prod_emb_dim=128, num_proj_dim=64):
        super().__init__()

        self.use_balance = use_balance
        
        # 1. Tokenizador original (Fusión Temprana para ver el contexto financiero)
        self.tokenizer = EventTokenizer(
            n_event_types=n_event_types,
            n_products=n_products,
            d_model=d_model,
            use_balance=use_balance,
            event_emb_dim=event_emb_dim,
            prod_emb_dim=prod_emb_dim,
            num_proj_dim=num_proj_dim
        )

        # 2. Backbone Mamba
        self.backbone = Mamba(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=2,
        )

        # 3. CABEZA DE CLASIFICACIÓN (Siguiente Acción)
        self.action_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_model, n_event_types + 1),
        )

        # 4. CABEZA DE REGRESIÓN (Predicción Financiera)
        num_dim = 3 if use_balance else 2
        self.finance_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.LayerNorm(d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, num_dim)
        )

        # 5. Proyección Contrastiva para Embeddings de Usuario
        self.contrastive_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )

    def forward(self, event_type, product_id, num_feats, return_embeddings=False):
        # Paso por el tokenizer y el bloque Mamba
        x = self.tokenizer(event_type, product_id, num_feats)
        h = self.backbone(x)
        
        # Tarea 1: Logits de eventos
        logits_action = self.action_head(h)
        
        # Tarea 2: Predicción de valores numéricos
        preds_finance = self.finance_head(h)

        if return_embeddings:
            # Mantener la lógica de normalización original sobre el último token
            user_emb = F.normalize(self.contrastive_proj(h[:, -1, :]), dim=-1)
            return logits_action, preds_finance, user_emb

        return logits_action, preds_finance

    def get_user_embedding(self, event_type, product_id, num_feats):
        """
        Devuelve el embedding L2-normalizado del usuario.
        Mantiene exactamente la lógica del código original.
        """
        self.eval() # Asegurar modo evaluación
        with torch.no_grad():
            x = self.tokenizer(event_type, product_id, num_feats)
            h = self.backbone(x)
            # Tomamos el último estado de la secuencia y proyectamos
            user_emb = self.contrastive_proj(h[:, -1, :])
            return F.normalize(user_emb, dim=-1)
            
class EventTokenizerLate(nn.Module):
    def __init__(self, n_event_types, n_products, d_model,
                 event_emb_dim=64, prod_emb_dim=128):
        super().__init__()
        
        self.event_emb = nn.Embedding(n_event_types + 1, event_emb_dim, padding_idx=0)
        self.prod_emb  = nn.Embedding(n_products + 1, prod_emb_dim, padding_idx=0)

        total_cat_dim = event_emb_dim + prod_emb_dim
        self.fusion = nn.Sequential(
            nn.Linear(total_cat_dim, d_model),
            nn.LayerNorm(d_model),
            nn.Dropout(0.1)
        )

    def forward(self, event_type, product_id):
        e = self.event_emb(event_type)
        p = self.prod_emb(product_id)
        combined = torch.cat([e, p], dim=-1)
        return self.fusion(combined)

class MambaModel(nn.Module):
    def __init__(self, n_event_types, n_products, d_model=128, d_state=32, d_conv=4,
                 use_balance=True, event_emb_dim=64, prod_emb_dim=128, num_proj_dim=64):
        super().__init__()

        self.tokenizer = EventTokenizer(
            n_event_types=n_event_types,
            n_products=n_products,
            d_model=d_model,
            use_balance=use_balance,
            event_emb_dim=event_emb_dim,
            prod_emb_dim=prod_emb_dim,
            num_proj_dim=num_proj_dim
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

    def get_user_embedding(self, event_type, product_id, num_feats):
        """
        Devuelve el embedding L2-normalizado del usuario.
        Toma el último token de la secuencia, lo proyecta con contrastive_proj
        y lo normaliza. Resultado: vector unitario (norma = 1.0).

        Args:
            event_type : LongTensor  (batch, seq_len)
            product_id : LongTensor  (batch, seq_len) — IDs remapeados con product_id_map
            num_feats  : FloatTensor (batch, seq_len, 3) — VALUE_NORM, TIME_DELTA_NORM, BALANCE_NORM

        Returns:
            FloatTensor (batch, d_model) — embedding normalizado
        """
        with torch.no_grad():
            x = self.tokenizer(event_type, product_id, num_feats)
            h = self.backbone(x)
            return F.normalize(self.contrastive_proj(h[:, -1, :]), dim=-1)


class MambaModelFusion(nn.Module):
    def __init__(self, n_event_types, n_products, d_model=128, d_state=32, d_conv=4,
                 use_balance=True, event_emb_dim=64, prod_emb_dim=128):
        super().__init__()

        self.use_balance = use_balance
        
        # 1. Tokenizador puro de comportamiento
        self.tokenizer = EventTokenizerLate(
            n_event_types=n_event_types,
            n_products=n_products,
            d_model=d_model,
            event_emb_dim=event_emb_dim,
            prod_emb_dim=prod_emb_dim
        )

        # 2. Backbone Mamba
        self.backbone = Mamba(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=2,
        )

        # 3. Late Fusion Head
        # Mamba saca `d_model`. Le concatenamos las features numéricas brutas (2 o 3)
        num_dim = 3 if use_balance else 2
        
        self.head = nn.Sequential(
            nn.Linear(d_model + num_dim, d_model),
            nn.LayerNorm(d_model), # Estabiliza la varianza tras concatenar raw features
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_model, n_event_types + 1),
        )

        # 4. Proyección Contrastiva (Opcional: puedes decidir si el embedding
        #    de usuario debe incluir el estado financiero o solo el conductual. 
        #    Aquí lo mantenemos solo conductual basado en Mamba).
        self.contrastive_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )

    def forward(self, event_type, product_id, num_feats, return_embeddings=False):
        # Mamba solo procesa la secuencia categórica
        x = self.tokenizer(event_type, product_id)
        h_mamba = self.backbone(x)
        
        # Ajustar features numéricas si no se usa balance
        if not self.use_balance:
            num_feats = num_feats[:, :, :2]
            
        # FUSIÓN TARDÍA: Concatenar salida del Mamba con las variables numéricas
        # h_mamba: [Batch, Seq_Len, d_model]
        # num_feats: [Batch, Seq_Len, num_dim]
        # fused_h: [Batch, Seq_Len, d_model + num_dim]
        fused_h = torch.cat([h_mamba, num_feats], dim=-1)
        
        # Clasificación final
        logits = self.head(fused_h)

        if return_embeddings:
            user_emb = F.normalize(self.contrastive_proj(h_mamba[:, -1, :]), dim=-1)
            return logits, user_emb

        return logits

    def get_user_embedding(self, event_type, product_id, num_feats):
        """
        Devuelve el embedding L2-normalizado del usuario.
        Toma el último token de la secuencia, lo proyecta con contrastive_proj
        y lo normaliza. Resultado: vector unitario (norma = 1.0).

        Args:
            event_type : LongTensor  (batch, seq_len)
            product_id : LongTensor  (batch, seq_len) — IDs remapeados con product_id_map
            num_feats  : FloatTensor (batch, seq_len, 3) — VALUE_NORM, TIME_DELTA_NORM, BALANCE_NORM

        Returns:
            FloatTensor (batch, d_model) — embedding normalizado
        """
        with torch.no_grad():
            x = self.tokenizer(event_type, product_id, num_feats)
            h = self.backbone(x)
            return F.normalize(self.contrastive_proj(h[:, -1, :]), dim=-1)
