import os
import argparse
import math
import wave
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
from PIL import Image

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp"}

# =======================
# UTIL AUDIO DI BASE
# =======================
def adsr_envelope(n_samples: int, sr: int,
                  attack=0.02, decay=0.08, sustain=0.7, release=0.1) -> np.ndarray:
    """
    ADSR robusta: se a+d+r > n, ridistribuisce proporzionalmente per evitare out-of-bounds.
    Garantisce sempre len(env) == n_samples.
    """
    n = int(max(1, n_samples))
    # campioni grezzi
    a = int(max(0, round(attack  * sr)))
    d = int(max(0, round(decay   * sr)))
    r = int(max(0, round(release * sr)))
    # se le code superano n, scala proporzionalmente (mantiene i rapporti)
    total_adr = a + d + r
    if total_adr > n:
        if total_adr == 0:
            a = d = r = 0
        else:
            scale = n / total_adr
            a = int(max(0, math.floor(a * scale)))
            d = int(max(0, math.floor(d * scale)))
            r = int(max(0, math.floor(r * scale)))
        # se per arrotondamenti mancano campioni, mettili al release
        while a + d + r < n:
            r += 1

    # sustain riempie il resto
    s = max(0, n - (a + d + r))

    env = np.zeros(n, dtype=np.float32)
    idx = 0
    if a > 0:
        env[idx:idx+a] = np.linspace(0.0, 1.0, a, endpoint=True, dtype=np.float32)
        idx += a
    if d > 0:
        env[idx:idx+d] = np.linspace(1.0, float(sustain), d, endpoint=True, dtype=np.float32)
        idx += d
    if s > 0:
        env[idx:idx+s] = float(sustain)
        idx += s
    if r > 0:
        # da sustain -> 0
        env[idx:idx+r] = np.linspace(float(sustain), 0.0, r, endpoint=True, dtype=np.float32)
        idx += r
    # sicurezza finale
    if idx < n:
        env[idx:] = 0.0
    return env


def sine_wave(freq: np.ndarray, sr: int) -> np.ndarray:
    """Accetta freq costante (float) o array per vibrato in Hz; restituisce float32."""
    if np.isscalar(freq):
        n = int(freq_duration_samples[0])  # placeholder if misused
    t = np.arange(freq.size, dtype=np.float32) / sr
    return np.sin(2 * np.pi * freq * t).astype(np.float32)

def sine_wave_const(freq: float, duration: float, sr: int) -> np.ndarray:
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
    """Equal-power; pan ∈ [-1..+1]"""
    pan = float(np.clip(pan, -1.0, 1.0))
    left_gain  = math.cos((pan + 1) * math.pi / 4)
    right_gain = math.sin((pan + 1) * math.pi / 4)
    return signal * left_gain, signal * right_gain

def concat_stereo_with_xfade(chunks: List[Tuple[np.ndarray, np.ndarray]], xfade_samps: int) -> Tuple[np.ndarray, np.ndarray]:
    if not chunks:
        return np.zeros(0, np.float32), np.zeros(0, np.float32)
    L_all = chunks[0][0].copy()
    R_all = chunks[0][1].copy()
    for L, R in chunks[1:]:
        if xfade_samps <= 0:
            L_all = np.concatenate([L_all, L])
            R_all = np.concatenate([R_all, R])
            continue
        xfade = min(xfade_samps, len(L_all), len(L))
        if xfade <= 0:
            L_all = np.concatenate([L_all, L])
            R_all = np.concatenate([R_all, R])
            continue
        fade_out = np.linspace(1.0, 0.0, xfade, endpoint=True).astype(np.float32)
        fade_in  = 1.0 - fade_out
        L_overlap = L_all[-xfade:] * fade_out + L[:xfade] * fade_in
        R_overlap = R_all[-xfade:] * fade_out + R[:xfade] * fade_in
        L_all = np.concatenate([L_all[:-xfade], L_overlap, L[xfade:]])
        R_all = np.concatenate([R_all[:-xfade], R_overlap, R[xfade:]])
    return L_all, R_all

# =======================
# IMMAGINE -> FEATURE
# =======================
def rgb_to_luma(img_rgb: np.ndarray) -> np.ndarray:
    r = img_rgb[..., 0].astype(np.float32) / 255.0
    g = img_rgb[..., 1].astype(np.float32) / 255.0
    b = img_rgb[..., 2].astype(np.float32) / 255.0
    return (0.2126 * r + 0.7152 * g + 0.0722 * b).astype(np.float32)

def rgb_to_hsv(rgb: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    # rgb [0..255] -> hsv [h(0..1), s(0..1), v(0..1)]
    r = rgb[..., 0].astype(np.float32) / 255.0
    g = rgb[..., 1].astype(np.float32) / 255.0
    b = rgb[..., 2].astype(np.float32) / 255.0
    mx = np.maximum(np.maximum(r, g), b)
    mn = np.minimum(np.minimum(r, g), b)
    df = mx - mn
    # hue
    h = np.zeros_like(mx)
    mask = df != 0
    idx = (mx == r) & mask
    h[idx] = ((g[idx] - b[idx]) / df[idx]) % 6
    idx = (mx == g) & mask
    h[idx] = ((b[idx] - r[idx]) / df[idx]) + 2
    idx = (mx == b) & mask
    h[idx] = ((r[idx] - g[idx]) / df[idx]) + 4
    h = h / 6.0
    s = np.zeros_like(mx)
    s[mx != 0] = df[mx != 0] / mx[mx != 0]
    v = mx
    return h, s, v

def sobel_edges(gray: np.ndarray) -> np.ndarray:
    # gray float32 [0..1]
    Kx = np.array([[1,0,-1],[2,0,-2],[1,0,-1]], dtype=np.float32)
    Ky = np.array([[1,2,1],[0,0,0],[-1,-2,-1]], dtype=np.float32)
    # pad con replicate
    g = np.pad(gray, ((1,1),(1,1)), mode='edge')
    Gx = (
        Kx[0,0]*g[:-2,:-2] + Kx[0,1]*g[:-2,1:-1] + Kx[0,2]*g[:-2,2:] +
        Kx[1,0]*g[1:-1,:-2] + Kx[1,1]*g[1:-1,1:-1] + Kx[1,2]*g[1:-1,2:] +
        Kx[2,0]*g[2:, :-2] + Kx[2,1]*g[2:, 1:-1] + Kx[2,2]*g[2:, 2:]
    )
    Gy = (
        Ky[0,0]*g[:-2,:-2] + Ky[0,1]*g[:-2,1:-1] + Ky[0,2]*g[:-2,2:] +
        Ky[1,0]*g[1:-1,:-2] + Ky[1,1]*g[1:-1,1:-1] + Ky[1,2]*g[1:-1,2:] +
        Ky[2,0]*g[2:, :-2] + Ky[2,1]*g[2:, 1:-1] + Ky[2,2]*g[2:, 2:]
    )
    mag = np.sqrt(Gx*Gx + Gy*Gy)
    mag = mag / max(1e-6, mag.max())
    return mag

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

# =======================
# MUSICAL MAPPING
# =======================
A4 = 440.0
def midi_to_hz(m: float) -> float:
    return 440.0 * (2 ** ((m - 69.0)/12.0))

SCALES = {
    "major":       [0,2,4,5,7,9,11],
    "minor":       [0,2,3,5,7,8,10],
    "pentatonic":  [0,2,4,7,9],
    "dorian":      [0,2,3,5,7,9,10],
    "mixolydian":  [0,2,4,5,7,9,10],
}

def quantize_to_scale(freq: float, root_hz: float, scale_name: str) -> float:
    """Quantizza freq alla nota più vicina nella scala. root_hz ~ C4/D4 ecc."""
    if scale_name not in SCALES:
        return freq
    # trova la midi note più vicina
    m = 69 + 12 * math.log2(freq / 440.0)
    # radice midi più vicina alla freq
    root_m = 69 + 12 * math.log2(root_hz / 440.0)
    # scorri vicino
    candidates = []
    for oct_shift in range(-5, 6):
        for deg in SCALES[scale_name]:
            mm = root_m + deg + 12*oct_shift
            candidates.append(mm)
    closest = min(candidates, key=lambda mm: abs(mm - m))
    return midi_to_hz(closest)

def chord_from_hue(hue_mean: float, mode_minor: bool=False) -> List[int]:
    """Triade I–IV–V (o i–iv–v) come gradi di scala; hue decide quale usare di più."""
    # Restituisce offsets in semitoni relativi alla radice (I triade)
    if mode_minor:
        I  = [0, 3, 7]
        IV = [5, 8, 12]
        V  = [7, 10, 14]
    else:
        I  = [0, 4, 7]
        IV = [5, 9, 12]
        V  = [7, 11, 14]
    h = hue_mean  # [0..1]
    if h < 1/3:
        return I
    elif h < 2/3:
        return IV
    else:
        return V

def soft_saw(freq: float, duration: float, sr: int, harmonics: int = 10) -> np.ndarray:
    """Saw ‘morbida’: somma 1/n di sin (rich arm.):"""
    n = int(duration * sr)
    t = np.arange(n, dtype=np.float32) / sr
    out = np.zeros(n, dtype=np.float32)
    for k in range(1, harmonics+1):
        out += (1.0/k) * np.sin(2*np.pi*freq*k*t)
    out *= (2.0/np.pi) * (0.7)  # normalizzazione blanda
    return out.astype(np.float32)

def soft_triangle(freq: float, duration: float, sr: int, harmonics: int = 8) -> np.ndarray:
    n = int(duration * sr)
    t = np.arange(n, dtype=np.float32) / sr
    out = np.zeros(n, dtype=np.float32)
    for i, k in enumerate(range(1, 2*harmonics, 2)):  # odd harmonics
        out += ((-1)**((k-1)//2)) * (1.0/(k*k)) * np.sin(2*np.pi*freq*k*t)
    out *= (8.0/(np.pi**2)) * 0.9
    return out.astype(np.float32)

def additive_stack(freq: float, duration: float, sr: int, harmonics: int, tilt: float=0.8) -> np.ndarray:
    """Additiva pura: ampiezza ~ (1/k)^tilt."""
    n = int(duration * sr)
    t = np.arange(n, dtype=np.float32) / sr
    out = np.zeros(n, dtype=np.float32)
    for k in range(1, harmonics+1):
        out += (1.0/(k**tilt)) * np.sin(2*np.pi*freq*k*t)
    out *= (1.0 / max(1e-6, np.max(np.abs(out)))) * 0.9
    return out.astype(np.float32)

def lfo_sine(n: int, sr: int, rate_hz: float, depth: float=1.0) -> np.ndarray:
    t = np.arange(n, dtype=np.float32) / sr
    return (np.sin(2*np.pi*rate_hz*t) * depth).astype(np.float32)

def apply_vibrato(sig: np.ndarray, sr: int, base_freq: float, vib_rate: float, vib_cents: float) -> np.ndarray:
    """Vibrato tramite resynth semplificato: FM leggera -> buona abbastanza per timbri stazionari."""
    if vib_cents <= 0 or vib_rate <= 0:
        return sig
    n = sig.size
    depth_ratio = 2 ** (vib_cents/1200.0) - 1.0
    t = np.arange(n, dtype=np.float32) / sr
    phase = 2*np.pi*base_freq*t + (2*np.pi*base_freq)*(depth_ratio/vib_rate)*(-np.cos(2*np.pi*vib_rate*t))
    return np.sin(phase).astype(np.float32)

def multitap_delay_stereo(L: np.ndarray, R: np.ndarray, sr: int,
                          times_ms=(120, 250, 380), fb=0.25, mix=0.2) -> Tuple[np.ndarray, np.ndarray]:
    """
    Delay multi-tap che mantiene la stessa lunghezza dell'input (n campioni).
    Ogni tap aggiunge una copia attenuata del segnale spostata di 'd' campioni.
    Il feedback è applicato in modo stabile senza accedere fuori range.
    """
    assert L.ndim == 1 and R.ndim == 1
    n = L.size
    if n == 0 or mix <= 0:
        return L, R

    taps = sorted({max(1, int(ms * sr / 1000.0)) for ms in times_ms if ms and ms > 0})
    if not taps:
        return L, R

    outL = L.astype(np.float32).copy()
    outR = R.astype(np.float32).copy()

    # Somma le copie ritardate rispettando i limiti (nessuna estensione della lunghezza)
    for d in taps:
        seg = n - d
        if seg <= 0:
            continue
        outL[d:n] += mix * L[:seg]
        outR[d:n] += mix * R[:seg]

    # Feedback semplice: riapplica una frazione del segnale già "delayed"
    if fb > 0:
        fb = float(np.clip(fb, 0.0, 0.95))
        for d in taps:
            seg = n - d
            if seg <= 0:
                continue
            outL[d:n] += fb * outL[:seg]
            outR[d:n] += fb * outR[:seg]

    # Soft clip per sicurezza
    outL = np.tanh(outL)
    outR = np.tanh(outR)
    return outL.astype(np.float32), outR.astype(np.float32)

# =======================
# CONFIG & IO
# =======================
@dataclass
class Config:
    input_dir: str
    output_dir: str
    mode: str                 # "single" | "per_col" | "per_row"
    sr: int
    unit_dur: float
    fmin: float
    fmax: float
    max_segments: int
    normalize: bool
    xfade: float
    # musical & timbre
    key_hz: float
    scale: str
    quantize: bool
    timbre: str               # "additive"|"saw"|"triangle"
    max_harm: int
    vib_rate: float
    vib_cents: float
    trem_rate: float
    trem_depth: float
    bpm: float
    grid: str                 # "off"|"1/4"|"1/8"|"1/16"
    delay_mix: float
    delay_fb: float

def list_images(directory: str) -> List[str]:
    files = []
    for name in sorted(os.listdir(directory)):
        ext = os.path.splitext(name)[1].lower()
        if ext in IMG_EXTS:
            files.append(os.path.join(directory, name))
    return files

def load_image_rgb(path: str) -> np.ndarray:
    return np.array(Image.open(path).convert("RGB"))

# =======================
# SINTESI “NOTE”
# =======================
def make_tone(freq: float, duration: float, cfg: Config, harmonics_hint: float = 0.5, pan: float = 0.0,
              attack=0.02, decay=0.08, sustain=0.75, release=0.10,
              edge_accent: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
    """Crea un tono ricco con preset scelto e LFO; edge_accent aumenta l’attacco."""
    if cfg.quantize:
        freq = quantize_to_scale(freq, cfg.key_hz, cfg.scale)

    n = int(duration * cfg.sr)
    harmonics = max(1, int(1 + harmonics_hint * (cfg.max_harm - 1)))

    # Timbro
    if cfg.timbre == "saw":
        core = soft_saw(freq, duration, cfg.sr, harmonics)
    elif cfg.timbre == "triangle":
        core = soft_triangle(freq, duration, cfg.sr, min(harmonics, 12))
    else:  # additive
        core = additive_stack(freq, duration, cfg.sr, harmonics, tilt=0.85)

    # LFO: vibrato (piccolo FM “musicale”)
    if cfg.vib_cents > 0 and cfg.vib_rate > 0:
        vib = apply_vibrato(core, cfg.sr, freq, cfg.vib_rate, cfg.vib_cents)
        core = 0.8*core + 0.2*vib

    # Tremolo (amp LFO)
    if cfg.trem_depth > 0 and cfg.trem_rate > 0:
        trem = 0.5 * (1.0 + lfo_sine(n, cfg.sr, cfg.trem_rate, 1.0))  # [0..1]
        trem = (1.0 - cfg.trem_depth) + cfg.trem_depth * trem
        core = core * trem

    # Envelope (edge accent alza l’attacco)
    atk = max(0.005, attack * (1.0 - 0.5*edge_accent))
    env = adsr_envelope(n, cfg.sr, attack=atk, decay=decay, sustain=sustain, release=release)
    sig = clamp_audio(core * env)

    # Panning
    L, R = pan_stereo(sig, pan)
    return L, R

def note_duration_from_grid(cfg: Config) -> float:
    if cfg.grid == "off" or cfg.bpm <= 0:
        return cfg.unit_dur
    beat = 60.0 / cfg.bpm  # quarter note
    if cfg.grid == "1/4":
        return beat
    if cfg.grid == "1/8":
        return beat / 2.0
    if cfg.grid == "1/16":
        return beat / 4.0
    return cfg.unit_dur

# =======================
# PIPELINE BRANO
# =======================
def build_song_single(img_paths: List[str], cfg: Config) -> Tuple[np.ndarray, np.ndarray]:
    chunks = []
    for p in img_paths:
        img = load_image_rgb(p)
        luma = rgb_to_luma(img)
        h, s, v = rgb_to_hsv(img)
        edges = sobel_edges(luma)

        b_mean = float(luma.mean())
        s_mean = float(s.mean())
        h_mean = float(h.mean())
        edge_mean = float(edges.mean())

        base_freq = map_brightness_to_freq(b_mean, cfg.fmin, cfg.fmax)
        dur = note_duration_from_grid(cfg)

        # Triade dalla tonalità percepita
        chord = chord_from_hue(h_mean, mode_minor=(cfg.scale == "minor"))
        # intensità armoniche in base alla saturazione
        chord_gain = np.linspace(1.0, 0.6, num=len(chord)) * (0.6 + 0.4*s_mean)
        # panning statico (centrato); aggiungi leggera aleatorietà con hue
        base_pan = (h_mean * 2.0 - 1.0) * 0.2

        # Root quantizzata
        root = base_freq
        if cfg.quantize:
            root = quantize_to_scale(base_freq, cfg.key_hz, cfg.scale)

        # costruiamo la triade
        Lmix = np.zeros(int(dur*cfg.sr), dtype=np.float32)
        Rmix = np.zeros_like(Lmix)
        for i, semis in enumerate(chord):
            f = midi_to_hz(69 + 12*math.log2(root/440.0) + semis)
            L, R = make_tone(f, dur, cfg,
                             harmonics_hint=s_mean,
                             pan=base_pan + (i-1)*0.15,
                             attack=0.02, decay=0.08, sustain=0.75, release=0.10,
                             edge_accent=edge_mean)
            Lmix[:L.size] += L * chord_gain[i]
            Rmix[:R.size] += R * chord_gain[i]
        # normalizza localmente triade
        peak = max(1e-6, float(max(np.abs(Lmix).max(), np.abs(Rmix).max())))
        Lmix *= 0.9/peak; Rmix *= 0.9/peak

        # piccolo delay per profondità
        if cfg.delay_mix > 0:
            Lmix, Rmix = multitap_delay_stereo(Lmix, Rmix, cfg.sr, fb=cfg.delay_fb, mix=cfg.delay_mix)

        chunks.append((Lmix, Rmix))

    L_all, R_all = concat_stereo_with_xfade(chunks, int(cfg.xfade * cfg.sr))
    return L_all, R_all

def build_song_axis(img_paths: List[str], cfg: Config, axis: int) -> Tuple[np.ndarray, np.ndarray]:
    chunks = []
    dur = note_duration_from_grid(cfg)
    for p in img_paths:
        img = load_image_rgb(p)
        luma = rgb_to_luma(img)
        h, s, v = rgb_to_hsv(img)
        edges = sobel_edges(luma)

        luma_r = reduce_axis(luma, axis=axis, max_segments=cfg.max_segments)
        h_r, s_r, v_r = [reduce_axis(x, axis=axis, max_segments=cfg.max_segments) for x in (h, s, v)]
        e_r = reduce_axis(edges, axis=axis, max_segments=cfg.max_segments)

        seg_count = luma_r.shape[axis]
        for i in range(seg_count):
            if axis == 1:
                b = float(luma_r[:, i].mean()); hh = float(h_r[:, i].mean()); ss=float(s_r[:, i].mean()); ee=float(e_r[:, i].mean())
                pan = -1.0 + 2.0*(i/max(1,seg_count-1))  # L->R
            else:
                b = float(luma_r[i, :].mean()); hh = float(h_r[i, :].mean()); ss=float(s_r[i, :].mean()); ee=float(e_r[i, :].mean())
                pan = -1.0 + 2.0*(i/max(1,seg_count-1))  # Top->Bottom mappato in pan

            f = map_brightness_to_freq(b, cfg.fmin, cfg.fmax)
            L, R = make_tone(f, dur, cfg, harmonics_hint=ss, pan=pan*0.8, edge_accent=ee)
            if cfg.delay_mix > 0:
                L, R = multitap_delay_stereo(L, R, cfg.sr, fb=cfg.delay_fb, mix=cfg.delay_mix)
            chunks.append((L, R))

    L_all, R_all = concat_stereo_with_xfade(chunks, int(cfg.xfade * cfg.sr))
    return L_all, R_all

# =======================
# MAIN
# =======================
def main():
    parser = argparse.ArgumentParser(description="Genera un'unica traccia WAV concatenando suoni derivati dalle immagini (versione musicale).")
    parser.add_argument("--input",  type=str, default=r"C:\Users\spina\OneDrive\Immagini\Fk_account", help="Cartella immagini")
    parser.add_argument("--output", type=str, default=r"C:\Users\spina\Documents\Other_Codes\Image2sound\Trial", help="Cartella output per il WAV")
    parser.add_argument("--outfile",type=str, default=None, help="Nome file di output (es: song_single.wav). Default: auto")
    parser.add_argument("--mode",   type=str, choices=["single", "per_col", "per_row"], default="single",
                        help="single=1 suono per immagine; per_col/per_row=1 suono per colonna/riga; tutto in un solo WAV")
    parser.add_argument("--sr",     type=int, default=44100, help="Sample rate")
    parser.add_argument("--unit_dur",    type=float, default=1.2, help="Durata (s) di OGNI suono (immagine o segmento) prima di grid quantization")
    parser.add_argument("--fmin",   type=float, default=150.0, help="Frequenza minima (Hz)")
    parser.add_argument("--fmax",   type=float, default=1800.0, help="Frequenza massima (Hz)")
    parser.add_argument("--max_segments", type=int, default=96, help="Max colonne/righe per immagine (riduzione automatica)")
    parser.add_argument("--no-normalize", action="store_true", help="Non normalizzare la traccia finale")
    parser.add_argument("--xfade",  type=float, default=0.04, help="Crossfade tra suoni (s)")

    # Musicalità & timbro
    parser.add_argument("--key_hz", type=float, default=261.63, help="Hz della tonica (C4=261.63, D4~293.66, A3~220, A4=440...)")
    parser.add_argument("--scale",  type=str, choices=list(SCALES.keys()), default="pentatonic", help="Scala musicale")
    parser.add_argument("--quantize", action="store_true", help="Quantizza le note alla scala/tonica")

    parser.add_argument("--timbre", type=str, choices=["additive","saw","triangle"], default="additive", help="Tipo di timbro")
    parser.add_argument("--max_harm", type=int, default=12, help="Massimo numero di armoniche")

    parser.add_argument("--vib_rate", type=float, default=5.5, help="Vibrato rate (Hz)")
    parser.add_argument("--vib_cents", type=float, default=8.0, help="Profondità vibrato (cent)")
    parser.add_argument("--trem_rate", type=float, default=4.0, help="Tremolo rate (Hz)")
    parser.add_argument("--trem_depth", type=float, default=0.25, help="Profondità tremolo [0..1]")

    parser.add_argument("--bpm", type=float, default=100.0, help="Tempo BPM per quantizzazione a griglia")
    parser.add_argument("--grid", type=str, choices=["off","1/4","1/8","1/16"], default="1/8", help="Griglia ritmica")

    parser.add_argument("--delay_mix", type=float, default=0.18, help="Delay mix [0..1]")
    parser.add_argument("--delay_fb", type=float, default=0.22, help="Delay feedback [0..1]")

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
        key_hz=args.key_hz,
        scale=args.scale,
        quantize=args.quantize,
        timbre=args.timbre,
        max_harm=args.max_harm,
        vib_rate=args.vib_rate,
        vib_cents=args.vib_cents,
        trem_rate=args.trem_rate,
        trem_depth=args.trem_depth,
        bpm=args.bpm,
        grid=args.grid,
        delay_mix=args.delay_mix,
        delay_fb=args.delay_fb,
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

    # Normalizzazione/limiter
    if cfg.normalize and L.size > 0:
        peak = max(1e-6, float(max(np.abs(L).max(), np.abs(R).max())))
        L = np.tanh(L/peak * 1.2) * 0.95
        R = np.tanh(R/peak * 1.2) * 0.95

    os.makedirs(cfg.output_dir, exist_ok=True)
    out_path = os.path.join(cfg.output_dir, args.outfile if args.outfile else default_name)
    write_wav_stereo(out_path, L, R, sr=cfg.sr)

    duration_sec = (len(L) / cfg.sr) if len(L) else 0.0
    print(f"Salvato: {out_path}  (durata ~ {duration_sec:.1f}s)")

if __name__ == "__main__":
    main()
