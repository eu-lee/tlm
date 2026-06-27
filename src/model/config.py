from dataclasses import dataclass


@dataclass
class ModelConfig:
    """Ternary transformer config preserving the original 10k architecture."""

    d_model: int = 1024
    n_heads: int = 16
    n_layers: int = 12
    d_ff: int = 2304
    vocab_size: int = 32000
    max_seq_len: int = 1024
    rope_theta: float = 10000.0
    norm_eps: float = 1e-6
    dropout: float = 0.0
    tie_embeddings: bool = True

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads

    @property
    def param_count(self) -> int:
        emb = self.vocab_size * self.d_model
        per_layer = (
            3 * self.d_model * self.d_model
            + self.d_model * self.d_model
            + 2 * self.d_model * self.d_ff
            + self.d_model * self.d_ff
        )
        total = emb + self.n_layers * per_layer
        if not self.tie_embeddings:
            total += emb
        return total

    @property
    def ternary_size_mb(self) -> float:
        emb_bytes = self.vocab_size * self.d_model
        ternary_params = self.param_count - self.vocab_size * self.d_model
        ternary_bytes = ternary_params * 2 / 8
        return (emb_bytes + ternary_bytes) / (1024 * 1024)
