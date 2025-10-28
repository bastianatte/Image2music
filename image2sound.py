import os
import argparse
import math
import wave
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
from PIL import Image

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp"}


# ---------- utility audio ----------
def adsr_envelope(n_samples: int, sr: int,
                  attack=0.02, decay=0.08, sustain=0.7, release=0.1) -> np.ndarray:
    a = max(1, int(attack * sr))
    d = max(1, int(decay * sr))
    r = max(1, int(release * sr))
    s = max(0, n_samples - (a + d + r))

    env = np.zeros(n_samples, dtype=np.float32)
    env[:a] = np.linspace(0.0, 1.0, a, endpoint=True)
    env[a:a+d] = np.linspace(1.0, sustain, d, endpoint=True)
    env[a+d:a+d+s] = sustain
    env[a+d+s:a+d+s+r] = np.linspace(sustain, 0.0, r, endpoint=True)
    return env


def sine_wave(freq: float, duration: float, sr: int) -> np.ndarray:
    t = np.arange(int(duration * sr)) / sr
    return np.sin(2 * np.pi * freq * t).astype(np.float32)


def clamp_audio(x: np.ndarray) -> np.ndarray:
    return np.clip(x, -1.0, 1.0).astype(np.float32)


def to_int16_stereo(left: np.ndarray, right: np.ndarray) -> bytes:
    left_i  = np.int16(np.clip(left,  -1, 1) * 32767)
    right_i = np.int16(np.clip(right, -1, 1) * 32767)
    interleaved = np.empty(left_i.size + right_i.size, dtype=np.int16)
    interleaved[0::2] = left_i
    interleaved[1::2] = right_i
    return interleaved.tobytes()


def write_wav_stereo(path: str, left: np.ndarray, right: np.ndarray, sr: int = 44100):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    data = to_int16_stereo(left, right)
    with wave.open(path, "wb") as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)   # 16 bit
        wf.setframerate(sr)
        wf.writeframes(data)


def pan_stereo(signal: np.ndarray, pan: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Panning stereo equal-power: pan in [-1..+1]
    """
    pan = float(np.clip(pan, -1.0, 1.0))
    left_gain  = math.cos((pan + 1) * math.pi / 4)
    right_gain = math.sin((pan + 1) * math.pi / 4)
    return signal * left_gain, signal * right_gain


def concat_stereo_with_xfade(chunks: List[Tuple[np.ndarray, np.ndarray]], xfade_samps: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Concatena molti (L,R) con crossfade lineare di xfade_samps per evitare click.
    """
    if not chunks:
        return np.zeros(0, np.float32), np.zeros(0, np.float32)

    L_all = chunks[0][0].copy()
    R_all = chunks[0][1].copy()

    for L, R in chunks[1:]:
        if xfade_samps <= 0:
            L_all = np.concatenate([L_all, L])
            R_all = np.concatenate([R_all, R])
            continue

        xfade_samps = min(xfade_samps, len(L_all), len(L))
        if xfade_samps <= 0:
            L_all = np.concatenate([L_all, L])
            R_all = np.concatenate([R_all, R])
            continue

        fade_out = np.linspace(1.0, 0.0, xfade_samps, endpoint=True).astype(np.float32)
        fade_in  = 1.0 - fade_out

        L_overlap = L_all[-xfade_samps:] * fade_out + L[:xfade_samps] * fade_in
        R_overlap = R_all[-xfade_samps:] * fade_out + R[:xfade_samps] * fade_in

        L_all = np.concatenate([L_all[:-xfade_samps], L_overlap, L[xfade_samps:]])
        R_all = np.concatenate([R_all[:-xfade_samps], R_overlap, R[xfade_samps:]])

    return L_all, R_all


# ---------- mapping immagine ----------
def rgb_to_luma(img_rgb: np.ndarray) -> np.ndarray:
    r = img_rgb[..., 0].astype(np.float32) / 255.0
    g = img_rgb[..., 1].astype(np.float32) / 255.0
    b = img_rgb[..., 2].astype(np.float32) / 255.0
    return (0.2126 * r + 0.7152 * g + 0.0722 * b).astype(np.float32)


def map_brightness_to_freq(bright: float, fmin: float, fmax: float) -> float:
    bright = float(np.clip(bright, 0.0, 1.0))
    log_min, log_max = math.log2(fmin), math.log2(fmax)
    f = 2 ** (log_min + bright * (log_max - log_min))
    return float(np.clip(f, fmin, fmax))


def reduce_axis(arr: np.ndarray, axis: int, max_segments: int) -> np.ndarray:
    length = arr.shape[axis]
    if length <= max_segments:
        return arr
    k = math.ceil(length / max_segments)
    cut = (length // k) * k
    slicer = [slice(None)] * arr.ndim
    slicer[axis] = slice(0, cut)
    trimmed = arr[tuple(slicer)]
    new_shape = list(trimmed.shape)
    new_shape[axis] = cut // k
    new_shape.insert(axis + 1, k)
    reshaped = trimmed.reshape(new_shape)
    return reshaped.mean(axis=axis + 1)


# ---------- core ----------
@dataclass
class Config:
    input_dir: str
    output_dir: str
    mode: str                 # "single" | "per_col" | "per_row"
    sr: int
    unit_dur: float           # durata di ciascun suono (immagine o segmento)
    fmin: float
    fmax: float
    max_segments: int
    normalize: bool
    xfade: float              # secondi di crossfade tra suoni


def list_images(directory: str) -> List[str]:
    files = []
    for name in sorted(os.listdir(directory)):
        ext = os.path.splitext(name)[1].lower()
        if ext in IMG_EXTS:
            files.append(os.path.join(directory, name))
    return files


def load_image_rgb(path: str) -> np.ndarray:
    return np.array(Image.open(path).convert("RGB"))


def tone_from_brightness(bright: float, cfg: Config, pan: float = 0.0, contrast_amp: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
    freq = map_brightness_to_freq(bright, cfg.fmin, cfg.fmax)
    base = sine_wave(freq, cfg.unit_dur, cfg.sr)
    env  = adsr_envelope(base.size, cfg.sr, attack=0.02, decay=0.08, sustain=0.75, release=0.10)
    sig  = clamp_audio(base * env * (0.6 + 0.4 * np.tanh(3.0 * contrast_amp)))
    L, R = pan_stereo(sig, pan=pan)
    return L, R


def build_song_single(img_paths: List[str], cfg: Config) -> Tuple[np.ndarray, np.ndarray]:
    chunks = []
    for p in img_paths:
        img = load_image_rgb(p)
        luma = rgb_to_luma(img)
        bright = float(luma.mean())
        contrast = float(luma.std())
        L, R = tone_from_brightness(bright, cfg, pan=0.0, contrast_amp=contrast)
        chunks.append((L, R))
    L_all, R_all = concat_stereo_with_xfade(chunks, int(cfg.xfade * cfg.sr))
    return L_all, R_all


def build_song_axis(img_paths: List[str], cfg: Config, axis: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Crea una sequenza di segmenti: per ogni immagine, si scorre lungo axis (colonne=1 o righe=0)
    e si aggiunge un tono per segmento. Panning progressivo sull’immagine corrente.
    """
    chunks = []
    for p in img_paths:
        img = load_image_rgb(p)
        luma = rgb_to_luma(img)
        luma = reduce_axis(luma, axis=axis, max_segments=cfg.max_segments)
        seg_count = luma.shape[axis]
        contrast = float(luma.std())

        for i in range(seg_count):
            if axis == 1:
                seg_b = float(luma[:, i].mean())
                pan = -1.0 + 2.0 * (i / max(1, seg_count - 1))
            else:
                seg_b = float(luma[i, :].mean())
                pan = -1.0 + 2.0 * (i / max(1, seg_count - 1))
            L, R = tone_from_brightness(seg_b, cfg, pan=pan, contrast_amp=contrast)
            chunks.append((L, R))

    L_all, R_all = concat_stereo_with_xfade(chunks, int(cfg.xfade * cfg.sr))
    return L_all, R_all


def main():
    parser = argparse.ArgumentParser(description="Genera un'unica traccia WAV concatenando i suoni derivati dalle immagini.")
    parser.add_argument("--input",  type=str, default=r"C:\Users\spina\OneDrive\Immagini\Fk_account", help="Cartella immagini")
    parser.add_argument("--output", type=str, default=r"C:\Users\spina\Documents\Other_Codes\Image2sound\Trial", help="Cartella output per il WAV")
    parser.add_argument("--outfile",type=str, default=None, help="Nome file di output (es: song_single.wav). Default: auto in base alla modalità.")
    parser.add_argument("--mode",   type=str, choices=["single", "per_col", "per_row"], default="single",
                        help="single=1 suono per immagine; per_col/per_row=1 suono per colonna/riga; tutto in un'unica canzone")
    parser.add_argument("--sr",     type=int, default=44100, help="Sample rate")
    parser.add_argument("--unit_dur",    type=float, default=1.2, help="Durata (s) di OGNI suono (immagine o segmento)")
    parser.add_argument("--fmin",   type=float, default=200.0, help="Frequenza minima (Hz)")
    parser.add_argument("--fmax",   type=float, default=1200.0, help="Frequenza massima (Hz)")
    parser.add_argument("--max_segments", type=int, default=120, help="Max colonne/righe per immagine (riduce immagini grandi)")
    parser.add_argument("--no-normalize", action="store_true", help="(solo finale) non normalizzare la traccia")
    parser.add_argument("--xfade",  type=float, default=0.03, help="Crossfade tra suoni (s) per evitare click")
    args = parser.parse_args()

    cfg = Config(
        input_dir=args.input,
        output_dir=args.output,
        mode=args.mode,
        sr=args.sr,
        unit_dur=args.unit_dur,
        fmin=args.fmin,
        fmax=args.fmax,
        max_segments=args.max_segments,
        normalize=(not args.no_normalize),
        xfade=args.xfade,
    )

    img_paths = list_images(cfg.input_dir)
    if not img_paths:
        print(f"Nessuna immagine trovata in: {cfg.input_dir}")
        return

    if cfg.mode == "single":
        L, R = build_song_single(img_paths, cfg)
        default_name = "image2song_single.wav"
    elif cfg.mode == "per_col":
        L, R = build_song_axis(img_paths, cfg, axis=1)
        default_name = "image2song_percol.wav"
    else:
        L, R = build_song_axis(img_paths, cfg, axis=0)
        default_name = "image2song_perrow.wav"

    # normalizzazione finale
    if cfg.normalize and L.size > 0:
        peak = max(1e-6, float(max(np.abs(L).max(), np.abs(R).max())))
        L = L / peak * 0.95
        R = R / peak * 0.95

    os.makedirs(cfg.output_dir, exist_ok=True)
    out_path = os.path.join(cfg.output_dir, args.outfile if args.outfile else default_name)
    write_wav_stereo(out_path, L, R, sr=cfg.sr)

    duration_sec = (len(L) / cfg.sr) if len(L) else 0.0
    print(f"Salvato: {out_path}  (durata ~ {duration_sec:.1f}s)")


if __name__ == "__main__":
    main()
