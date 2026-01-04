# src/wayformer/transformer_wayformer.py
import torch
import torch.nn as nn

from src.models.pos_encoding import SinusoidalTimeEncoding


class WayformerBaseline(nn.Module):
    """
    Wayformer-style:
      - Encoder builds 'memory' from tokens
      - M learnable queries go through TransformerDecoder (cross-attn to memory)
      - Output trajectories: (B, M, Tf, 2)
      - Output scores/logits: (B, M)  (optional)
    """

    def __init__(
        self,
        T=25, Tf=25, K=8,
        ego_dim: int = 16,
        nb_dim: int = 9,
        use_neighbors: bool = True,
        use_slot_emb: bool = True,
        d_model=128, nhead=4,
        enc_layers=2, dec_layers=2,
        dropout=0.1,
        predict_delta: bool = False,
        M: int = 6,
        return_scores: bool = True,
    ):
        super().__init__()
        self.T, self.Tf, self.K = T, Tf, K
        self.use_neighbors = use_neighbors
        self.use_slot_emb = use_slot_emb
        self.predict_delta = predict_delta

        self.M = M
        self.return_scores = return_scores

        self.ego_proj = nn.Linear(ego_dim, d_model)
        self.nb_proj = nn.Linear(nb_dim, d_model)

        self.time_enc = SinusoidalTimeEncoding(d_model, max_len=T)
        self.slot_emb = nn.Embedding(1 + K, d_model) if use_slot_emb else None

        # Encoder
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=enc_layers)

        # Learnable queries (modes)
        self.query_emb = nn.Embedding(M, d_model)

        # Decoder
        dec_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=dec_layers)

        # Heads
        self.traj_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, Tf * 2),
        )
        self.score_head = nn.Linear(d_model, 1)

    def forward(self, x_ego, x_nb, nb_mask, style_prob=None, style_valid=None):
        """
        Keep signature compatible with TransformerStyleBaseline calls.
        style_prob/style_valid are ignored here.
        """
        B, T, _ = x_ego.shape
        ego_tok = self.ego_proj(x_ego)  # (B,T,d)

        if self.use_neighbors:
            nb_tok = self.nb_proj(x_nb)  # (B,T,K,d)
            tok = torch.cat([ego_tok.unsqueeze(2), nb_tok], dim=2)  # (B,T,1+K,d)
            tok = tok.reshape(B, T * (1 + self.K), -1)              # (B,L,d)

            valid = torch.ones((B, T, 1 + self.K), device=tok.device, dtype=torch.bool)
            valid[:, :, 1:] = nb_mask
            key_padding_mask = ~valid.reshape(B, -1)  # True=pad
        else:
            tok = ego_tok
            key_padding_mask = None

        # time PE
        if self.use_neighbors:
            t_idx = torch.arange(T, device=tok.device).repeat_interleave(1 + self.K)
        else:
            t_idx = torch.arange(T, device=tok.device)
        tok = tok + self.time_enc(t_idx).unsqueeze(0).repeat(B, 1, 1)

        # slot emb
        if self.use_neighbors and (self.slot_emb is not None):
            slot_ids = torch.arange(0, 1 + self.K, device=tok.device).repeat(T)
            tok = tok + self.slot_emb(slot_ids).unsqueeze(0).repeat(B, 1, 1)

        # encoder memory
        memory = self.encoder(tok, src_key_padding_mask=key_padding_mask)  # (B,L,d)

        # decoder queries
        q = self.query_emb.weight.unsqueeze(0).repeat(B, 1, 1)  # (B,M,d)
        dec_out = self.decoder(
            tgt=q,
            memory=memory,
            memory_key_padding_mask=key_padding_mask,
        )  # (B,M,d)

        traj = self.traj_head(dec_out).view(B, self.M, self.Tf, 2)  # (B,M,Tf,2)
        scores = self.score_head(dec_out).squeeze(-1)               # (B,M)

        return (traj, scores) if self.return_scores else traj
    
class WayformerStyle(nn.Module):
    """
    Wayformer-style + Style injection (방법 1):
      - Encoder input: add style embedding to ego tokens
      - Decoder input: add style embedding to query tokens
    """

    def __init__(
        self,
        T=25, Tf=25, K=8,
        ego_dim: int = 16,
        nb_dim: int = 9,
        use_neighbors: bool = True,
        use_slot_emb: bool = True,
        d_model=128, nhead=4,
        enc_layers=2, dec_layers=2,
        dropout=0.1,
        predict_delta: bool = False,
        M: int = 6,
        return_scores: bool = True,

        # --- style options ---
        style_dim_in: int = 3,          # 3/6/9 실험 가능
        use_style_enc: bool = True,     # encoder(ego_tok)에 style 주입
        use_style_dec: bool = True,     # decoder(query)에 style 주입
        style_dropout: float = 0.1,
        share_style_mlp: bool = False,  # True면 enc/dec가 같은 MLP 공유
    ):
        super().__init__()
        self.T, self.Tf, self.K = T, Tf, K
        self.use_neighbors = use_neighbors
        self.use_slot_emb = use_slot_emb
        self.predict_delta = predict_delta
        self.M = M
        self.return_scores = return_scores

        self.style_dim_in = int(style_dim_in)
        self.use_style_enc = bool(use_style_enc) and (self.style_dim_in > 0)
        self.use_style_dec = bool(use_style_dec) and (self.style_dim_in > 0)

        # projections
        self.ego_proj = nn.Linear(ego_dim, d_model)
        self.nb_proj = nn.Linear(nb_dim, d_model)

        # style mlps
        def make_style_mlp():
            return nn.Sequential(
                nn.Linear(self.style_dim_in, d_model),
                nn.ReLU(),
                nn.Dropout(style_dropout),
                nn.Linear(d_model, d_model),
            )

        if (self.use_style_enc or self.use_style_dec):
            if share_style_mlp:
                shared = make_style_mlp()
                self.style_mlp_enc = shared
                self.style_mlp_dec = shared
            else:
                self.style_mlp_enc = make_style_mlp() if self.use_style_enc else None
                self.style_mlp_dec = make_style_mlp() if self.use_style_dec else None
        else:
            self.style_mlp_enc = None
            self.style_mlp_dec = None

        # PE / slot
        self.time_enc = SinusoidalTimeEncoding(d_model, max_len=T)
        self.slot_emb = nn.Embedding(1 + K, d_model) if use_slot_emb else None

        # encoder
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=4 * d_model,
            dropout=dropout, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=enc_layers)

        # queries + decoder
        self.query_emb = nn.Embedding(M, d_model)

        dec_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=4 * d_model,
            dropout=dropout, batch_first=True
        )
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=dec_layers)

        # heads
        self.traj_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, Tf * 2)
        )
        self.score_head = nn.Linear(d_model, 1)

    def forward(self, x_ego, x_nb, nb_mask, style_prob=None, style_valid=None):
        """
        style_prob: (B, style_dim_in)
        style_valid: (B,) bool or {0,1}
        """
        B, T, _ = x_ego.shape
        ego_tok = self.ego_proj(x_ego)  # (B,T,D)

        # ---- (1) Encoder style injection: ego_tok += s_enc ----
        if self.use_style_enc and (style_prob is not None) and (style_valid is not None):
            s_enc = self.style_mlp_enc(style_prob)  # (B,D)
            s_enc = s_enc * style_valid.float().unsqueeze(-1)
            ego_tok = ego_tok + s_enc.unsqueeze(1)  # (B,T,D)

        # ---- build tokens (same as baseline) ----
        if self.use_neighbors:
            nb_tok = self.nb_proj(x_nb)  # (B,T,K,D)
            tok = torch.cat([ego_tok.unsqueeze(2), nb_tok], dim=2)  # (B,T,1+K,D)
            tok = tok.reshape(B, T * (1 + self.K), -1)              # (B,L,D)

            valid = torch.ones((B, T, 1 + self.K), device=tok.device, dtype=torch.bool)
            valid[:, :, 1:] = nb_mask
            key_padding_mask = ~valid.reshape(B, -1)  # True=pad
        else:
            tok = ego_tok
            key_padding_mask = None

        # time PE
        if self.use_neighbors:
            t_idx = torch.arange(T, device=tok.device).repeat_interleave(1 + self.K)
        else:
            t_idx = torch.arange(T, device=tok.device)
        tok = tok + self.time_enc(t_idx).unsqueeze(0).repeat(B, 1, 1)

        # slot emb
        if self.use_neighbors and (self.slot_emb is not None):
            slot_ids = torch.arange(0, 1 + self.K, device=tok.device).repeat(T)
            tok = tok + self.slot_emb(slot_ids).unsqueeze(0).repeat(B, 1, 1)

        # encoder memory
        memory = self.encoder(tok, src_key_padding_mask=key_padding_mask)  # (B,L,D)

        # ---- (2) Decoder style injection: q += s_dec ----
        q = self.query_emb.weight.unsqueeze(0).repeat(B, 1, 1)  # (B,M,D)
        if self.use_style_dec and (style_prob is not None) and (style_valid is not None):
            s_dec = self.style_mlp_dec(style_prob)  # (B,D)
            s_dec = s_dec * style_valid.float().unsqueeze(-1)
            q = q + s_dec.unsqueeze(1)              # (B,M,D)

        # decoder cross-attn
        dec_out = self.decoder(
            tgt=q,
            memory=memory,
            memory_key_padding_mask=key_padding_mask,
        )  # (B,M,D)

        traj = self.traj_head(dec_out).view(B, self.M, self.Tf, 2)  # (B,M,Tf,2)
        scores = self.score_head(dec_out).squeeze(-1)               # (B,M)

        return (traj, scores) if self.return_scores else traj