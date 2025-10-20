# -*- coding: utf-8 -*-
"""
Importable library for:
- VAD + chunking
- Embeddings (ECAPA/Resemblyzer)
- Clustering (fixed/thresh)
- Merge
- Whisper transcription
Exposes: Config, Pipeline, run_file, run_batch, build_config
"""

from __future__ import annotations
import os, sys, time, logging, shutil, tempfile, traceback, subprocess
from dataclasses import dataclass, asdict
from typing import List, Tuple, Optional, Sequence, Dict
import numpy as np

import torch
import librosa
import soundfile as sf
import webrtcvad
import whisper

from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import pairwise_distances

# ---- torch.load patch (aligned with the original script) ----
_orig_load = torch.load
def _patch_load(*args, **kwargs):
    if "weights_only" not in kwargs:
        kwargs["weights_only"] = False
    return _orig_load(*args, **kwargs)
torch.load = _patch_load

# ======================= Device Utils =======================
def resolve_device(requested: str) -> str:
    """Use CUDA only when it is actually available; otherwise fall back to CPU."""
    if requested == "cuda" and torch.cuda.is_available():
        return "cuda"
    return "cpu"

def describe_cuda() -> str:
    if not torch.cuda.is_available():
        return "CUDA: unavailable"
    try:
        dev_id = torch.cuda.current_device()
        return f"CUDA {torch.version.cuda} | {torch.cuda.get_device_name(dev_id)}"
    except Exception:
        return f"CUDA {torch.version.cuda} | <unknown device>"

def log_env(logger: logging.Logger):
    logger.info(f"Torch: {torch.__version__} | CUDA avail: {torch.cuda.is_available()} | Build CUDA: {torch.version.cuda}")
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")

# ======================= Config =======================
@dataclass
class Config:
    audio_dir: str = r"D:\02_Programme\KI\SpeechToText\Audio\neu"
    model_name: str = "tiny"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    target_sr: int = 16000

    # VAD
    vad_mode: int = 1
    frame_ms: int = 30
    min_speech_sec: float = 0.35
    merge_gap_sec: float = 0.20
    pad_edge_ms: int = 120

    # Sub-Chunks
    sub_max_len_sec: float = 4.0
    sub_hop_sec: float = 2.0

    # Embeddings
    emb_backend: str = "ecapa"      # "resemblyzer" | "ecapa"
    emb_min_sec: float = 0.30
    thresh_grid: Tuple[float, ...] = (0.35, 0.40, 0.45, 0.50, 0.55)

    # Clustering
    cluster_mode: str = "fixed"     # "fixed" | "thresh"
    n_speakers: int = 1
    clusterer: str = "kmeans"       # "kmeans" | "agglom"
    max_speakers: Optional[int] = 2
    smooth_window: int = 2

    # Merge final
    gap_merge_sec: float = 2.0

    # Output
    segment_min_sec: float = 0.35
    output_suffix: str = "_diarized"
    log_filename: str = "batch_voice_cluster.log"

    # Whisper
    whisper_kw: Dict = None
    language: Optional[str] = None

    log_level: str = "DEBUG"

    def __post_init__(self):
        if self.whisper_kw is None:
            self.whisper_kw = {
                "language": "de",
                "task": "transcribe",
                "temperature": 0.0,
                "beam_size": 5,
                "condition_on_previous_text": False,
                "no_speech_threshold": 0.6,
            }
        if self.language:
            self.whisper_kw["language"] = self.language

# ======================= Logging =======================
class LoggerFactory:
    @staticmethod
    def create(base_dir: str, level: str, log_filename: str) -> logging.Logger:
        os.makedirs(base_dir, exist_ok=True)
        log_path = os.path.join(base_dir, log_filename)
        logger = logging.getLogger(f"pipeline:{os.path.basename(base_dir)}")
        logger.setLevel(getattr(logging, level.upper(), logging.INFO))
        logger.handlers.clear()
        fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
        sh = logging.StreamHandler(sys.stdout); sh.setFormatter(fmt)
        fh = logging.FileHandler(log_path, encoding="utf-8"); fh.setFormatter(fmt)
        logger.addHandler(sh); logger.addHandler(fh)
        return logger

# ======================= Audio IO =======================
class AudioIO:
    def __init__(self, cfg: Config, logger: logging.Logger):
        self.cfg = cfg
        self.logger = logger

    def safe_basename(self, path: str) -> str:
        return os.path.splitext(os.path.basename(path))[0]

    def _ffmpeg_convert(self, in_path: str, out_path: str) -> bool:
        """Fallback conversion (exotic WAV/codecs -> PCM16, 16k, mono)."""
        cmd = [
            "ffmpeg", "-y", "-i", in_path,
            "-ac", "1", "-ar", str(self.cfg.target_sr),
            "-acodec", "pcm_s16le",
            out_path
        ]
        try:
            return subprocess.call(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) == 0
        except FileNotFoundError:
            # ffmpeg not installed -> fallback unavailable
            return False

    def convert_to_wav16k_mono(self, in_path: str, tmpdir: str) -> str:
        out = os.path.join(tmpdir, "input_16k_mono.wav")
        try:
            y, _ = librosa.load(in_path, sr=self.cfg.target_sr, mono=True)
            sf.write(out, y.astype(np.float32), self.cfg.target_sr, subtype="PCM_16")
            return out
        except Exception as e:
            self.logger.warning(f"librosa/soundfile could not read '{in_path}' ({e}). Trying ffmpeg fallback ...")
            ok = self._ffmpeg_convert(in_path, out)
            if not ok:
                raise
            return out

    def load_audio(self, path: str) -> Tuple[np.ndarray, int]:
        y, sr = librosa.load(path, sr=self.cfg.target_sr, mono=True)
        return y.astype(np.float32), sr

    def write_wav(self, path: str, y: np.ndarray, sr: int):
        sf.write(path, y, sr, subtype="PCM_16")

# ======================= VAD + Splitter =======================
class VADSegmenter:
    def __init__(self, cfg: Config, logger: logging.Logger):
        self.cfg = cfg
        self.logger = logger
        self.vad = webrtcvad.Vad(self.cfg.vad_mode)

    def segments(self, y: np.ndarray, sr: int) -> List[Tuple[int, int]]:
        frame_len = int(sr * self.cfg.frame_ms / 1000)
        pad = int(sr * self.cfg.pad_edge_ms / 1000)
        voiced = []
        for i in range(0, len(y) - frame_len + 1, frame_len):
            frame = y[i:i+frame_len]
            pcm16 = (np.clip(frame, -1, 1) * 32767).astype(np.int16).tobytes()
            voiced.append((i, i+frame_len, self.vad.is_speech(pcm16, sr)))

        segments = []
        cur_start, cur_end = None, None
        for (a, b, is_speech) in voiced:
            if is_speech and cur_start is None: cur_start, cur_end = a, b
            elif is_speech: cur_end = b
            elif cur_start is not None:
                if (b - cur_end) <= int(sr * self.cfg.merge_gap_sec): continue
                segments.append((max(0, cur_start - pad), min(len(y), cur_end + pad)))
                cur_start, cur_end = None, None

        if cur_start is not None:
            segments.append((max(0, cur_start - pad), min(len(y), cur_end + pad)))

        out = []
        for (a, b) in segments:
            if (b - a) / sr >= self.cfg.min_speech_sec:
                out.append((a, b))
        self.logger.debug(f"VAD segments (n={len(out)}): {[(round(a/sr,2), round(b/sr,2)) for a,b in out][:12]} ...")
        return out

class ChunkSplitter:
    def __init__(self, cfg: Config):
        self.cfg = cfg

    def split(self, chunks: List[Tuple[int, int]], sr: int) -> List[Tuple[int, int]]:
        max_len = int(self.cfg.sub_max_len_sec * sr)
        hop = int(self.cfg.sub_hop_sec * sr)
        out = []
        for (a, b) in chunks:
            L = b - a
            if L <= max_len:
                out.append((a, b)); continue
            i = a
            while i + max_len <= b:
                out.append((i, i + max_len))
                i += hop
            if b - i > int(1.0 * sr):
                out.append((i, b))
        return out

# ======================= Embedding (Strategy) =======================
class BaseEmbedder:
    def __init__(self, cfg: Config, logger: logging.Logger):
        self.cfg = cfg
        self.logger = logger
    def embed(self, wav_1d: np.ndarray) -> np.ndarray:
        raise NotImplementedError

class ECAPAEmbedder(BaseEmbedder):
    def __init__(self, cfg: Config, logger: logging.Logger):
        super().__init__(cfg, logger)
        import speechbrain as sb
        dev = "cuda" if cfg.device == "cuda" else "cpu"
        savedir = os.path.join(os.path.expanduser("~"), ".cache", "speechbrain_ecapa")

        def _load(savedir_arg):
            # speechbrain 1.0 uses sb.inference.*, older versions use sb.pretrained.*
            try:
                return sb.inference.EncoderClassifier.from_hparams(
                    source="speechbrain/spkrec-ecapa-voxceleb",
                    run_opts={"device": dev},
                    savedir=savedir_arg,
                    # Tip: newer SpeechBrain versions accept overrides:
                    # overrides={"pretrainer": {"collect_in": None}}
                )
            except AttributeError:
                return sb.pretrained.EncoderClassifier.from_hparams(
                    source="speechbrain/spkrec-ecapa-voxceleb",
                    run_opts={"device": dev},
                    savedir=savedir_arg,
                )

        try:
            # Faster: use collect_in/symlinks (Windows may raise error 1314)
            self.cls = _load(savedir)
        except OSError as e:
            # No symlink permissions -> retry without savedir (relies on HF cache, copies files)
            if getattr(e, "winerror", None) == 1314:
                logger.warning("No symlink permissions (WinError 1314). Loading ECAPA without symlinks ...")
                self.cls = _load(None)
            else:
                raise

    def embed(self, wav_1d: np.ndarray) -> np.ndarray:
        wav = torch.tensor(wav_1d, dtype=torch.float32).unsqueeze(0)
        if self.cfg.device == "cuda":
            wav = wav.cuda(non_blocking=True)
        with torch.no_grad():
            emb = self.cls.encode_batch(wav).squeeze(0).squeeze(0)
        return emb.detach().cpu().numpy()

class ResemblyzerEmbedder(BaseEmbedder):
    def __init__(self, cfg: Config, logger: logging.Logger):
        super().__init__(cfg, logger)
        from resemblyzer import VoiceEncoder
        dev = "cuda" if cfg.device == "cuda" else "cpu"
        self.enc = VoiceEncoder().to(dev)
    def embed(self, wav_1d: np.ndarray) -> np.ndarray:
        return self.enc.embed_utterance(wav_1d)

class EmbedderFactory:
    @staticmethod
    def create(cfg: Config, logger: logging.Logger) -> BaseEmbedder:
        if cfg.emb_backend.lower() == "ecapa":
            logger.info("Embedding backend: ECAPA (SpeechBrain)")
            return ECAPAEmbedder(cfg, logger)
        logger.info("Embedding backend: Resemblyzer")
        return ResemblyzerEmbedder(cfg, logger)

class EmbeddingExtractor:
    def __init__(self, cfg: Config, logger: logging.Logger, embedder: BaseEmbedder):
        self.cfg, self.logger, self.embedder = cfg, logger, embedder
    def for_chunks(self, y: np.ndarray, sr: int, chunks: List[Tuple[int, int]]) -> List[Optional[np.ndarray]]:
        embs: List[Optional[np.ndarray]] = []
        for i, (a, b) in enumerate(chunks):
            dur = (b - a) / sr
            if dur < self.cfg.emb_min_sec:
                embs.append(None); self.logger.debug(f"[emb {i}] short dur={dur:.2f}s -> None"); continue
            embs.append(self.embedder.embed(y[a:b])); self.logger.debug(f"[emb {i}] ok dur={dur:.2f}s")
        return embs

# ======================= Clustering =======================
def _cluster_fixed(X: np.ndarray, n_speakers: int, method="kmeans") -> np.ndarray:
    if method == "kmeans":
        cl = KMeans(n_clusters=max(1, n_speakers), n_init="auto", random_state=0)
        return cl.fit_predict(X)
    cl = AgglomerativeClustering(n_clusters=max(1, n_speakers), metric="euclidean", linkage="average")
    return cl.fit_predict(X)

def _cluster_thresh_sweep(X: np.ndarray, grid: Sequence[float], logger: logging.Logger) -> np.ndarray:
    D = pairwise_distances(X, metric="cosine")
    best_th, best_labs, best_k = None, None, -1
    for th in grid:
        cl = AgglomerativeClustering(n_clusters=None, distance_threshold=th, metric="precomputed", linkage="average")
        labs = cl.fit_predict(D)
        k = labs.max() + 1
        logger.debug(f"[auto] thresh={th:.2f} -> k={k}")
        if 2 <= k <= 5:
            logger.info(f"[auto] selected: thresh={th:.2f}, clusters={k}")
            return labs
        if k > best_k:
            best_th, best_labs, best_k = th, labs, k
    logger.info(f"[auto] fallback: thresh={best_th:.2f}, clusters={best_k}")
    return best_labs

def _compress_clusters_to_k(X_valid: np.ndarray, labs_valid: np.ndarray, k: int, method: str = "kmeans") -> np.ndarray:
    labs_valid = np.asarray(labs_valid)
    uniq = np.unique(labs_valid)
    if len(uniq) <= k:
        return labs_valid
    centroids = np.stack([X_valid[labs_valid == c].mean(axis=0) for c in uniq], axis=0)
    if method == "kmeans":
        km = KMeans(n_clusters=k, n_init="auto", random_state=0)
        group = km.fit_predict(centroids)
    else:
        ag = AgglomerativeClustering(n_clusters=k, metric="euclidean", linkage="average")
        group = ag.fit_predict(centroids)
    remap_old_to_new = {int(c): int(g) for c, g in zip(uniq, group)}
    return np.array([remap_old_to_new[int(c)] for c in labs_valid], dtype=int)

def _smooth_labels_sequence(labels: np.ndarray, window: int = 2) -> np.ndarray:
    import collections
    L = np.asarray(labels).tolist()
    out = L[:]
    for i in range(len(L)):
        a = max(0, i - window)
        b = min(len(L), i + window + 1)
        cnt = collections.Counter(L[a:b])
        out[i] = cnt.most_common(1)[0][0]
    return np.asarray(out, dtype=int)

class Clusterer:
    def __init__(self, cfg: Config, logger: logging.Logger):
        self.cfg, self.logger = cfg, logger
    def labels_from_embeddings(self, embs: List[Optional[np.ndarray]]) -> List[int]:
        idx = [i for i, e in enumerate(embs) if e is not None]
        if not idx: return [0] * len(embs)
        X_valid = np.stack([embs[i] for i in idx], axis=0)

        if self.cfg.cluster_mode == "fixed":
            labs_valid = _cluster_fixed(X_valid, self.cfg.n_speakers, method=self.cfg.clusterer)
            self.logger.debug(f"Clustering fixed: k={self.cfg.n_speakers}, uniq={np.unique(labs_valid).tolist()}")
        else:
            labs_valid = _cluster_thresh_sweep(X_valid, self.cfg.thresh_grid, self.logger)
            k0 = labs_valid.max() + 1
            self.logger.debug(f"Clustering thresh (raw): k={k0}, uniq={np.unique(labs_valid).tolist()}")
            if self.cfg.max_speakers is not None and k0 > self.cfg.max_speakers:
                labs_valid = _compress_clusters_to_k(X_valid, labs_valid, k=self.cfg.max_speakers, method="kmeans")
                self.logger.info(f"[cap] clusters compressed to K={self.cfg.max_speakers}")
            k1 = labs_valid.max() + 1
            if k1 < 2 and self.cfg.n_speakers >= 2:
                self.logger.info(f"[fallback] thresh produced {k1} cluster(s); switching to fixed={self.cfg.n_speakers}")
                labs_valid = _cluster_fixed(X_valid, self.cfg.n_speakers, method="kmeans")

        if self.cfg.smooth_window and self.cfg.smooth_window > 0:
            labs_valid = _smooth_labels_sequence(labs_valid, window=self.cfg.smooth_window)

        labels_full = [-1] * len(embs)
        for ii, lab in zip(idx, labs_valid):
            labels_full[ii] = int(lab)
        last = None
        for i in range(len(labels_full)):
            if labels_full[i] == -1:
                left = next((labels_full[j] for j in range(i-1, -1, -1) if labels_full[j] != -1), None)
                right = next((labels_full[j] for j in range(i+1, len(labels_full)) if labels_full[j] != -1), None)
                labels_full[i] = left if left is not None else (right if right is not None else (last if last is not None else 0))
            last = labels_full[i]
        remap, nxt, out = {}, 0, []
        for lab in labels_full:
            if lab not in remap:
                remap[lab] = nxt; nxt += 1
            out.append(remap[lab])
        return out

# ======================= Merger =======================
class Merger:
    def __init__(self, cfg: Config):
        self.cfg = cfg
    def merge_same_speaker_over_gaps(self, chunks: List[Tuple[int, int]], labels: List[int], sr: int) -> List[Tuple[float, float, int]]:
        if not chunks: return []
        out: List[Tuple[float, float, int]] = []
        cur_a, cur_b = chunks[0]
        cur_lab = labels[0]
        for i in range(1, len(chunks)):
            a, b = chunks[i]; lab = labels[i]
            gap = (a - cur_b) / sr
            if lab == cur_lab and gap <= self.cfg.gap_merge_sec:
                cur_b = b
            else:
                out.append((cur_a / sr, cur_b / sr, cur_lab))
                cur_a, cur_b, cur_lab = a, b, lab
        out.append((cur_a / sr, cur_b / sr, cur_lab))
        return out

# ======================= Whisper =======================
class WhisperTranscriber:
    def __init__(self, cfg: Config, logger: logging.Logger):
        self.cfg, self.logger = cfg, logger
        eff_dev = self.cfg.device  # hier ist das effektive Device bereits gesetzt
        self.logger.info("Loading Whisper model ...")
        t0 = time.time()
        self.model = whisper.load_model(self.cfg.model_name, device=eff_dev)
        took = time.time() - t0
        if eff_dev == "cuda":
            self.logger.info(f"Whisper loaded in {took:.2f}s on CUDA ({describe_cuda()}).")
        else:
            self.logger.info(f"Whisper loaded in {took:.2f}s on CPU.")
    def transcribe_wav(self, wav_path: str) -> str:
        res = self.model.transcribe(wav_path, **self.cfg.whisper_kw)
        return (res.get("text") or "").strip()

# ======================= Pipeline =======================
class Pipeline:
    def __init__(self, cfg: Config, logger: logging.Logger):
        self.cfg, self.logger = cfg, logger
        self.audio = AudioIO(cfg, logger)
        self.vad = VADSegmenter(cfg, logger)
        self.splitter = ChunkSplitter(cfg)
        self.embedder = EmbedderFactory.create(cfg, logger)
        self.emb_extractor = EmbeddingExtractor(cfg, logger, self.embedder)
        self.clusterer = Clusterer(cfg, logger)
        self.merger = Merger(cfg)
        self.whisper = WhisperTranscriber(cfg, logger)

    def process_file(self, in_path: str) -> Tuple[bool, Optional[str]]:
        base = self.audio.safe_basename(in_path)
        out_txt = os.path.join(self.cfg.audio_dir, f"{base}{self.cfg.output_suffix}.txt")
        tmpdir = tempfile.mkdtemp(prefix=f"{base}_", dir=self.cfg.audio_dir)
        seg_wav = os.path.join(tmpdir, "segment.wav")
        try:
            wav16 = self.audio.convert_to_wav16k_mono(in_path, tmpdir)
            y, sr = self.audio.load_audio(wav16)
            self.logger.debug(f"Audio len={len(y)}, dur={len(y)/sr:.2f}s")

            vad_raw = self.vad.segments(y, sr)
            self.logger.debug(f"Number of VAD chunks (pre split): {len(vad_raw)}")

            chunks = self.splitter.split(vad_raw, sr)
            self.logger.debug(f"Chunks after split (n={len(chunks)}): {[(round(a/sr,2), round(b/sr,2)) for a,b in chunks][:20]} ...")
            if not chunks: raise RuntimeError("No speech segments detected.")

            embs = self.emb_extractor.for_chunks(y, sr, chunks)
            valid = sum(e is not None for e in embs)
            self.logger.debug(f"Embeddings: valid={valid}/{len(embs)} (min_dur={self.cfg.emb_min_sec}s)")
            if valid < 2: self.logger.warning("Very few valid embeddings - clustering may collapse.")

            labels = self.clusterer.labels_from_embeddings(embs)
            uniq = sorted(set(labels)); counts = np.bincount(labels) if len(labels) else []
            self.logger.debug(f"Cluster IDs uniq: {uniq} (counts={counts.tolist() if len(counts)>0 else []})")

            merged = self.merger.merge_same_speaker_over_gaps(chunks, labels, sr)
            self.logger.debug(f"Merged segments (n={len(merged)}): {merged[:12]} ...")

            with open(out_txt, "w", encoding="utf-8") as f:
                for idx, (s, e, spk) in enumerate(merged):
                    a, b = int(s*sr), int(e*sr)
                    dur = (b-a)/sr
                    if dur < self.cfg.segment_min_sec:
                        self.logger.debug(f"[seg {idx}] too short ({dur:.2f}s) -> skip"); continue
                    clip = y[a:b]
                    self.audio.write_wav(seg_wav, clip, sr)
                    t_seg = time.time()
                    text = self.whisper.transcribe_wav(seg_wav)
                    dt_seg = time.time() - t_seg
                    self.logger.debug(f"[seg {idx}] transcribed in {dt_seg:.2f}s, spk={spk}, text_len={len(text)}")
                    f.write(f"SPEAKER_{spk}: {text}\n")

            return True, out_txt
        except Exception as ex:
            self.logger.error(f"Error processing {in_path}: {ex}")
            self.logger.debug(traceback.format_exc())
            return False, None
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def run_batch(self) -> None:
        ok = failed = 0
        for fname in os.listdir(self.cfg.audio_dir):
            if not fname.lower().endswith((".mp3",".wav",".mp4",".opus",".m4a",".flac")):
                continue
            in_path = os.path.join(self.cfg.audio_dir, fname)
            self.logger.info(f"--- File: {in_path}")
            t0 = time.time()
            success, out_txt = self.process_file(in_path)
            if success:
                self.logger.info(f"Completed: {out_txt} ({time.time()-t0:.2f}s)")
                ok += 1
            else:
                failed += 1
        self.logger.info(f"Batch done. OK={ok}, Failed={failed} (log at {os.path.join(self.cfg.audio_dir, self.cfg.log_filename)})")

# ======================= Mini-API (for imports) =======================
def build_config(**kwargs) -> Config:
    """
    Build a config from kwargs (everything optional).
    Beispiel: build_config(audio_dir="...", model_name="base", emb_backend="ecapa")
    """
    cfg = Config(**{**kwargs})
    return cfg

def _make_logger(cfg: Config) -> logging.Logger:
    return LoggerFactory.create(cfg.audio_dir, cfg.log_level, cfg.log_filename)

def run_file(in_path: str, **kwargs) -> Tuple[bool, Optional[str]]:
    """
    Simple function call for other scripts.
    Returns: (success, output_txt_path)
    """
    cfg = build_config(**kwargs)
    # Determine the effective device once
    cfg.device = resolve_device(cfg.device)
    logger = _make_logger(cfg)
    logger.debug(f"Config: {asdict(cfg)}")
    log_env(logger)
    if cfg.device == "cuda":
        logger.info(f"CUDA active: {describe_cuda()}")
    else:
        logger.info("CUDA not available - using CPU")
    pipe = Pipeline(cfg, logger)
    return pipe.process_file(in_path)

def run_batch(audio_dir: Optional[str] = None, **kwargs) -> None:
    """
    Batch processing for a directory.
    """
    cfg = build_config(**kwargs)
    if audio_dir is not None:
        cfg.audio_dir = audio_dir
    cfg.device = resolve_device(cfg.device)
    logger = _make_logger(cfg)
    logger.debug(f"Config: {asdict(cfg)}")
    log_env(logger)
    if cfg.device == "cuda":
        logger.info(f"CUDA active: {describe_cuda()}")
    else:
        logger.info("CUDA not available - using CPU")
    pipe = Pipeline(cfg, logger)
    pipe.run_batch()
