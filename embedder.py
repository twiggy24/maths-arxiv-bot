# embedder.py  -- dual backend: 'onnx' (local) or 'openai')
from typing import List
import os
import numpy as np

BACKEND = os.getenv("EMBED_BACKEND", "onnx").lower()

# -----------------------------
# Backend A: OpenAI (optional)
# -----------------------------
if BACKEND == "openai":
    from openai import OpenAI
    from settings import settings  # reads .env via pydantic-settings

    EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-large")
    EMBED_DIM = 3072  # must match model above

    if not settings.OPENAI_API_KEY:
        raise RuntimeError(
            "OPENAI_API_KEY is missing. Set it in .env or export it. "
            "Or set EMBED_BACKEND=onnx to use the local model."
        )
    _client = OpenAI(api_key=settings.OPENAI_API_KEY)

    def embed_texts(texts: List[str]) -> List[List[float]]:
        """Embed document/passages (OpenAI models apparently do not require a 'passage:' prefix)."""
        resp = _client.embeddings.create(model=EMBED_MODEL, input=texts)
        return [d.embedding for d in resp.data]

    def embed_queries(texts: List[str]) -> List[List[float]]:
        """Keep the API uniform: for OpenAI we just call the same endpoint."""
        resp = _client.embeddings.create(model=EMBED_MODEL, input=texts)
        return [d.embedding for d in resp.data]

# -----------------------------
# Backend B: ONNX (local, CPU)
# -----------------------------
else:
    import glob
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer
    import onnxruntime as ort

    # An ONNX export of BGE-M3; produces dense 1024-dim embeddings
    ONNX_REPO_ID = os.getenv("ONNX_REPO_ID", "gpahal/bge-m3-onnx-int8")
    # Execution provider; CPU is default and universal
    ONNX_PROVIDER = os.getenv("ONNX_PROVIDER", "CPUExecutionProvider")

    EMBED_DIM = 1024  # bge-m3 dense embedding size

    #Download (or use cache) once to a local folder
    _model_dir = snapshot_download(
        repo_id=ONNX_REPO_ID,
        local_dir="models/bge_m3_onnx",
        ignore_patterns=["*.safetensors", "*.bin"]
    )

    # Tokeniser comes from the same snapshot (no torch needed)
    _tok = AutoTokenizer.from_pretrained(_model_dir)

    # Locate the actual ONNX file present in the snapshot
    _candidates = ["model.onnx", "model_quantized.onnx", "model_fp16.onnx", "bge-m3.onnx"]
    model_path = next(
        (os.path.join(_model_dir, name) for name in _candidates if os.path.exists(os.path.join(_model_dir, name))),
        None
    )
    if model_path is None:
        found = glob.glob(os.path.join(_model_dir, "*.onnx"))
        if not found:
            raise FileNotFoundError(f"No .onnx file found under {_model_dir}")
        model_path = found[0]

    # Create an inference session
    _session = ort.InferenceSession(model_path, providers=[ONNX_PROVIDER])

    def _mean_pool(last_hidden_state: np.ndarray, attention_mask: np.ndarray) -> np.ndarray:
        """
        Mean pool over tokens using the attention mask.
        last_hidden_state: (B, T, H)
        attention_mask:    (B, T)
        returns: (B, H)
        """
        mask = attention_mask[:, :, None].astype(last_hidden_state.dtype)  # (B,T,1)
        summed = (last_hidden_state * mask).sum(axis=1)                    # (B,H)
        counts = np.clip(mask.sum(axis=1), 1e-9, None)                     # (B,1)
        return summed / counts

    def _run_onnx(batch_texts: List[str]) -> List[List[float]]:
        """Tokenide -> ONNX forward -> (optional) mean-pool -> L2 normalise."""
        enc = _tok(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="np"
        )
        inputs = {
            "input_ids": enc["input_ids"].astype(np.int64),
            "attention_mask": enc["attention_mask"].astype(np.int64),
        }
        outputs = _session.run(None, inputs)
        first = outputs[0]
        pooled = _mean_pool(first, enc["attention_mask"]) if first.ndim == 3 else first
        norms = np.linalg.norm(pooled, axis=1, keepdims=True)
        pooled = pooled / np.clip(norms, 1e-12, None)
        return pooled.astype(np.float32).tolist()

    def _batch_map(texts: List[str], prefix: str) -> List[List[float]]:
        """Apply BGE-M3's required prefix and batch through ONNX."""
        out: List[List[float]] = []
        B = 64
        for i in range(0, len(texts), B):
            batch = [(prefix + t.strip()) for t in texts[i:i + B]]
            out.extend(_run_onnx(batch))
        return out

    def embed_texts(texts: List[str]) -> List[List[float]]:
        """Embed passages/chunks. BGE-M3 expects a 'passage: ' prefix."""
        return _batch_map(texts, "passage: ")

    def embed_queries(texts: List[str]) -> List[List[float]]:
        """Embed queries. BGE-M3 expects a 'query: ' prefix."""
        return _batch_map(texts, "query: ")




    