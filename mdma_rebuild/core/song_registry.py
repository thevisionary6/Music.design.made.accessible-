"""MDMA Song Registry System.

Permanent song database with quality assurance pipeline.

Features:
- Persistent song registry (survives restarts)
- Folder scanning with recursive search
- Quality analysis and grading (Low/Medium/High)
- Automatic WAV64 float conversion for high-quality playback
- Auto-tagging (BPM, key, energy, genre hints)
- AI-powered analysis for improved metadata
- Load by index or name from anywhere

Registry Structure:
    ~/Documents/MDMA/
    └── songs/
        ├── registry.json        # Master song database
        ├── cache/               # Converted WAV64 files
        │   └── <hash>.wav       # High-quality cached files
        └── analysis/            # Analysis cache
            └── <hash>.json      # Cached analysis results
"""

from __future__ import annotations

import json
import os
import hashlib
import struct
import wave
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
import numpy as np


# ============================================================================
# ENUMS & CONSTANTS
# ============================================================================

class QualityGrade(Enum):
    """Audio quality grades."""
    LOW = "low"           # < 128kbps MP3, mono, clipped
    MEDIUM = "medium"     # 128-256kbps, some issues
    HIGH = "high"         # 320kbps+ or lossless, clean


class FileFormat(Enum):
    """Supported audio formats."""
    WAV = "wav"
    MP3 = "mp3"
    FLAC = "flac"
    OGG = "ogg"
    M4A = "m4a"
    AIFF = "aiff"
    WAV64 = "w64"  # Sony Wave64


# Quality thresholds
QUALITY_THRESHOLDS = {
    'min_sample_rate': 44100,      # Minimum for HIGH
    'min_bit_depth': 16,           # Minimum for HIGH
    'max_clipping_ratio': 0.001,   # Max samples at clipping for HIGH
    'min_dynamic_range': 6.0,      # dB, minimum for HIGH
    'min_bitrate_mp3': 256,        # kbps for HIGH
    'min_bitrate_medium': 128,     # kbps for MEDIUM
}


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class SongAnalysis:
    """Audio analysis results."""
    # Technical
    sample_rate: int = 44100
    bit_depth: int = 16
    channels: int = 2
    duration: float = 0.0
    bitrate: int = 0  # kbps for compressed formats
    
    # Quality metrics
    peak_db: float = 0.0
    rms_db: float = -20.0
    dynamic_range_db: float = 10.0
    clipping_ratio: float = 0.0
    dc_offset: float = 0.0
    noise_floor_db: float = -60.0
    
    # Musical analysis
    bpm: float = 0.0
    bpm_confidence: float = 0.0
    key: str = ""
    key_confidence: float = 0.0
    energy: float = 0.5  # 0-1
    danceability: float = 0.5  # 0-1
    
    # Genre hints
    genre_hints: List[str] = field(default_factory=list)
    mood_hints: List[str] = field(default_factory=list)
    
    # AI description
    ai_description: str = ""
    
    # Analysis timestamp
    analyzed_at: str = ""


@dataclass
class SongEntry:
    """Song registry entry."""
    # Identity
    id: int = 0                    # Unique registry index
    hash: str = ""                 # File content hash (SHA256)
    
    # File info
    original_path: str = ""        # Original file location
    filename: str = ""             # Original filename
    format: str = ""               # File format
    file_size: int = 0             # Bytes
    
    # Quality
    quality_grade: str = "medium"  # low/medium/high
    quality_issues: List[str] = field(default_factory=list)
    
    # Converted cache
    cached_path: str = ""          # Path to WAV64 cache (if converted)
    cached_at: str = ""            # When converted
    
    # Analysis
    analysis: Optional[Dict] = None
    
    # User metadata
    title: str = ""
    artist: str = ""
    album: str = ""
    year: int = 0
    tags: List[str] = field(default_factory=list)
    rating: int = 0  # 0-5 stars
    play_count: int = 0
    last_played: str = ""
    
    # Registry metadata
    added_at: str = ""
    modified_at: str = ""
    
    # Flags
    favorite: bool = False
    hidden: bool = False


# ============================================================================
# SONG REGISTRY CLASS
# ============================================================================

class SongRegistry:
    """Permanent song database with quality assurance."""
    
    def __init__(self):
        self.songs: Dict[int, SongEntry] = {}
        self.hash_index: Dict[str, int] = {}  # hash -> id
        self.name_index: Dict[str, List[int]] = {}  # lowercase name -> ids
        self.next_id: int = 1
        self._registry_path: Optional[Path] = None
        self._cache_dir: Optional[Path] = None
        self._analysis_dir: Optional[Path] = None
        
        # Load registry
        self._init_paths()
        self.load()
        
        # Validate cache on startup
        self._validate_cache()
    
    def _init_paths(self):
        """Initialize registry paths."""
        from .user_data import get_mdma_root
        
        songs_dir = get_mdma_root() / 'songs'
        songs_dir.mkdir(parents=True, exist_ok=True)
        
        self._registry_path = songs_dir / 'registry.json'
        self._cache_dir = songs_dir / 'cache'
        self._cache_dir.mkdir(exist_ok=True)
        self._analysis_dir = songs_dir / 'analysis'
        self._analysis_dir.mkdir(exist_ok=True)
    
    def _validate_cache(self):
        """Validate cache matches registry. Re-cache missing files on startup."""
        if not self._cache_dir or not self.songs:
            return
        
        # Count cached files
        cached_files = set(f.stem for f in self._cache_dir.glob('*.wav'))
        
        # Count songs that should have cache
        songs_needing_cache = []
        for song in self.songs.values():
            if song.quality_grade == 'low':
                continue  # Don't cache low quality
            
            # Check if cached
            if song.hash in cached_files:
                # Update cached_path if not set
                if not song.cached_path:
                    song.cached_path = str(self._cache_dir / f"{song.hash}.wav")
            else:
                # Missing from cache
                songs_needing_cache.append(song)
        
        # Re-cache missing files
        if songs_needing_cache:
            print(f"MDMA: Validating cache... {len(songs_needing_cache)} files need caching")
            recached = 0
            for song in songs_needing_cache:
                original = Path(song.original_path)
                if original.exists():
                    try:
                        cached = self._convert_to_wav64(original, song.hash)
                        if cached:
                            song.cached_path = str(cached)
                            song.cached_at = datetime.now().isoformat()
                            recached += 1
                    except Exception as e:
                        print(f"  Failed to cache {song.filename}: {e}")
            
            if recached > 0:
                self.save()
                print(f"MDMA: Cached {recached} files")
    
    def load(self) -> bool:
        """Load registry from disk."""
        if not self._registry_path or not self._registry_path.exists():
            return False
        
        try:
            with open(self._registry_path, 'r') as f:
                data = json.load(f)
            
            self.next_id = data.get('next_id', 1)
            
            for song_data in data.get('songs', []):
                entry = SongEntry(**song_data)
                self.songs[entry.id] = entry
                
                # Build indexes
                if entry.hash:
                    self.hash_index[entry.hash] = entry.id
                self._index_name(entry)
            
            return True
        except Exception as e:
            print(f"Error loading registry: {e}")
            return False
    
    def save(self) -> bool:
        """Save registry to disk."""
        if not self._registry_path:
            return False
        
        try:
            data = {
                'next_id': self.next_id,
                'songs': [asdict(s) for s in self.songs.values()],
                'saved_at': datetime.now().isoformat(),
            }
            
            with open(self._registry_path, 'w') as f:
                json.dump(data, f, indent=2)
            
            return True
        except Exception as e:
            print(f"Error saving registry: {e}")
            return False
    
    def _index_name(self, entry: SongEntry):
        """Add entry to name index."""
        # Index by filename
        name_lower = entry.filename.lower()
        if name_lower not in self.name_index:
            self.name_index[name_lower] = []
        if entry.id not in self.name_index[name_lower]:
            self.name_index[name_lower].append(entry.id)
        
        # Index by title if different
        if entry.title:
            title_lower = entry.title.lower()
            if title_lower not in self.name_index:
                self.name_index[title_lower] = []
            if entry.id not in self.name_index[title_lower]:
                self.name_index[title_lower].append(entry.id)
    
    def _compute_hash(self, filepath: Path) -> str:
        """Compute SHA256 hash of file."""
        sha256 = hashlib.sha256()
        with open(filepath, 'rb') as f:
            for chunk in iter(lambda: f.read(65536), b''):
                sha256.update(chunk)
        return sha256.hexdigest()[:16]  # Use first 16 chars
    
    # ========================================================================
    # SCANNING
    # ========================================================================
    
    def scan_folder(
        self,
        folder: Union[str, Path],
        recursive: bool = True,
        analyze: bool = True,
        convert: bool = True,
        progress_callback=None
    ) -> Dict[str, Any]:
        """Scan folder for audio files and add to registry.
        
        Parameters
        ----------
        folder : path
            Folder to scan
        recursive : bool
            Scan subdirectories
        analyze : bool
            Run quality analysis on files
        convert : bool
            Convert to WAV64 if needed
        progress_callback : callable
            Called with (current, total, filename)
        
        Returns
        -------
        dict
            Scan results with counts and issues
        """
        folder = Path(folder)
        if not folder.exists():
            return {'error': f"Folder not found: {folder}"}
        
        # Find audio files
        audio_extensions = {'.wav', '.mp3', '.flac', '.ogg', '.m4a', '.aiff', '.w64'}
        files = []
        
        if recursive:
            for ext in audio_extensions:
                files.extend(folder.rglob(f'*{ext}'))
        else:
            for ext in audio_extensions:
                files.extend(folder.glob(f'*{ext}'))
        
        results = {
            'scanned': 0,
            'added': 0,
            'skipped': 0,
            'updated': 0,
            'errors': [],
            'quality': {'high': 0, 'medium': 0, 'low': 0},
        }
        
        total = len(files)
        
        for i, filepath in enumerate(files):
            if progress_callback:
                progress_callback(i + 1, total, filepath.name)
            
            results['scanned'] += 1
            
            try:
                # Check if already in registry
                file_hash = self._compute_hash(filepath)
                
                if file_hash in self.hash_index:
                    # Already registered - update path if needed
                    song_id = self.hash_index[file_hash]
                    song = self.songs[song_id]
                    if song.original_path != str(filepath):
                        song.original_path = str(filepath)
                        song.modified_at = datetime.now().isoformat()
                        results['updated'] += 1
                    else:
                        results['skipped'] += 1
                    continue
                
                # Add new song
                entry = self._create_entry(filepath, file_hash)
                
                # Analyze if requested
                if analyze:
                    analysis = self.analyze_file(filepath)
                    entry.analysis = asdict(analysis)
                    
                    # Grade quality
                    grade, issues = self._grade_quality(filepath, analysis)
                    entry.quality_grade = grade.value
                    entry.quality_issues = issues
                    results['quality'][grade.value] += 1
                
                # Convert if needed
                if convert and entry.quality_grade != 'low':
                    cached = self._convert_to_wav64(filepath, file_hash)
                    if cached:
                        entry.cached_path = str(cached)
                        entry.cached_at = datetime.now().isoformat()
                
                # Register
                self._register_entry(entry)
                results['added'] += 1
                
            except Exception as e:
                results['errors'].append(f"{filepath.name}: {str(e)}")
        
        # Save registry
        self.save()
        
        return results
    
    def _create_entry(self, filepath: Path, file_hash: str) -> SongEntry:
        """Create a new song entry."""
        stat = filepath.stat()
        
        entry = SongEntry(
            id=self.next_id,
            hash=file_hash,
            original_path=str(filepath),
            filename=filepath.name,
            format=filepath.suffix.lower().lstrip('.'),
            file_size=stat.st_size,
            added_at=datetime.now().isoformat(),
            modified_at=datetime.now().isoformat(),
        )
        
        # Try to extract metadata from filename
        name = filepath.stem
        if ' - ' in name:
            parts = name.split(' - ', 1)
            entry.artist = parts[0].strip()
            entry.title = parts[1].strip()
        else:
            entry.title = name
        
        return entry
    
    def _register_entry(self, entry: SongEntry):
        """Register entry in database."""
        self.songs[entry.id] = entry
        self.hash_index[entry.hash] = entry.id
        self._index_name(entry)
        self.next_id += 1
    
    # ========================================================================
    # QUALITY ANALYSIS
    # ========================================================================
    
    def analyze_file(self, filepath: Union[str, Path]) -> SongAnalysis:
        """Analyze audio file for quality and musical content.
        
        Returns comprehensive analysis including:
        - Technical specs (sample rate, bit depth, channels)
        - Quality metrics (peak, RMS, dynamic range, clipping)
        - Musical analysis (BPM, key, energy)
        - Genre/mood hints
        """
        filepath = Path(filepath)
        analysis = SongAnalysis(analyzed_at=datetime.now().isoformat())
        
        # Load audio
        try:
            audio, sr = self._load_audio(filepath)
        except Exception as e:
            analysis.ai_description = f"Error loading: {e}"
            return analysis
        
        analysis.sample_rate = sr
        analysis.channels = 1 if audio.ndim == 1 else audio.shape[1]
        analysis.duration = len(audio) / sr
        
        # Convert to mono for analysis
        if audio.ndim > 1:
            mono = np.mean(audio, axis=1)
        else:
            mono = audio
        
        # Technical analysis
        analysis.peak_db = 20 * np.log10(np.max(np.abs(mono)) + 1e-10)
        analysis.rms_db = 20 * np.log10(np.sqrt(np.mean(mono**2)) + 1e-10)
        analysis.dynamic_range_db = analysis.peak_db - analysis.rms_db
        
        # Clipping detection
        clip_threshold = 0.99
        clipped_samples = np.sum(np.abs(mono) > clip_threshold)
        analysis.clipping_ratio = clipped_samples / len(mono)
        
        # DC offset
        analysis.dc_offset = abs(np.mean(mono))
        
        # Noise floor (estimate from quietest section)
        frame_size = int(sr * 0.1)  # 100ms frames
        if len(mono) > frame_size * 10:
            frames = np.array_split(mono[:len(mono) - len(mono) % frame_size], 
                                   len(mono) // frame_size)
            frame_rms = [np.sqrt(np.mean(f**2)) for f in frames]
            noise_floor = np.percentile(frame_rms, 5)  # 5th percentile
            analysis.noise_floor_db = 20 * np.log10(noise_floor + 1e-10)
        
        # BPM detection
        analysis.bpm, analysis.bpm_confidence = self._detect_bpm(mono, sr)
        
        # Key detection
        analysis.key, analysis.key_confidence = self._detect_key(mono, sr)
        
        # Energy (overall loudness/intensity)
        analysis.energy = min(1.0, max(0.0, (analysis.rms_db + 20) / 20))
        
        # Danceability (based on rhythmic content)
        analysis.danceability = self._estimate_danceability(mono, sr)
        
        # Genre hints based on characteristics
        analysis.genre_hints = self._guess_genre(analysis)
        
        # Mood hints
        analysis.mood_hints = self._guess_mood(analysis)
        
        return analysis
    
    def _load_audio(self, filepath: Path) -> Tuple[np.ndarray, int]:
        """Load audio file to numpy array.
        
        Supports WAV, MP3, FLAC, OGG, M4A via soundfile or ffmpeg.
        """
        ext = filepath.suffix.lower()
        
        # Method 1: Try soundfile first (handles many formats)
        try:
            import soundfile as sf
            audio, sr = sf.read(str(filepath), dtype='float64')
            return audio, sr
        except ImportError:
            pass  # soundfile not installed
        except Exception:
            pass  # soundfile couldn't read this format
        
        # Method 2: For WAV files, use built-in wave module
        if ext in ('.wav', '.w64'):
            try:
                return self._load_wav(filepath)
            except Exception as e:
                # WAV loading failed, try ffmpeg as last resort
                pass
        
        # Method 3: Convert with ffmpeg for other formats
        try:
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
                tmp_path = tmp.name
            
            try:
                # Use ffmpeg to convert to WAV
                result = subprocess.run([
                    'ffmpeg', '-y', '-i', str(filepath),
                    '-acodec', 'pcm_s16le', '-ar', '48000',
                    '-loglevel', 'error',
                    tmp_path
                ], capture_output=True, timeout=60)
                
                if result.returncode == 0 and os.path.exists(tmp_path):
                    audio, sr = self._load_wav(Path(tmp_path))
                    return audio, sr
                else:
                    raise RuntimeError(f"ffmpeg failed: {result.stderr.decode()[:200]}")
            finally:
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)
                    
        except FileNotFoundError:
            # ffmpeg not installed
            pass
        except subprocess.TimeoutExpired:
            raise RuntimeError("ffmpeg timed out converting file")
        except Exception as e:
            raise RuntimeError(f"Could not load audio: {e}")
        
        # Method 4: Try pydub as last resort
        try:
            from pydub import AudioSegment
            
            audio_seg = AudioSegment.from_file(str(filepath))
            audio_seg = audio_seg.set_frame_rate(48000)
            
            samples = np.array(audio_seg.get_array_of_samples())
            if audio_seg.channels == 2:
                samples = samples.reshape((-1, 2))
            
            audio = samples.astype(np.float64) / 32768.0
            return audio, 48000
        except ImportError:
            pass
        except Exception:
            pass
        
        raise RuntimeError(
            f"Cannot load {ext} file. Install soundfile (pip install soundfile) "
            f"or ensure ffmpeg is available."
        )
    
    def _load_wav(self, filepath: Path) -> Tuple[np.ndarray, int]:
        """Load WAV file.
        
        Raises RuntimeError if file is not a valid WAV.
        """
        # First check if file starts with RIFF header
        try:
            with open(filepath, 'rb') as f:
                header = f.read(4)
                if header != b'RIFF':
                    raise RuntimeError(f"Not a WAV file (missing RIFF header): {filepath.name}")
        except Exception as e:
            raise RuntimeError(f"Cannot read file: {e}")
        
        try:
            with wave.open(str(filepath), 'rb') as wf:
                sr = wf.getframerate()
                n_channels = wf.getnchannels()
                n_frames = wf.getnframes()
                sample_width = wf.getsampwidth()
                
                raw_data = wf.readframes(n_frames)
        except wave.Error as e:
            raise RuntimeError(f"Invalid WAV file: {e}")
        
        # Convert to numpy
        if sample_width == 1:
            dtype = np.uint8
            max_val = 128.0
        elif sample_width == 2:
            dtype = np.int16
            max_val = 32768.0
        elif sample_width == 3:
            # 24-bit - handle specially
            dtype = np.int32
            max_val = 8388608.0
            # Unpack 24-bit to 32-bit
            raw_data = b''.join(
                bytes([0]) + raw_data[i:i+3] 
                for i in range(0, len(raw_data), 3)
            )
        elif sample_width == 4:
            dtype = np.int32
            max_val = 2147483648.0
        else:
            dtype = np.float32
            max_val = 1.0
        
        audio = np.frombuffer(raw_data, dtype=dtype).astype(np.float64)
        audio = audio / max_val
        
        if n_channels > 1:
            audio = audio.reshape(-1, n_channels)
        
        return audio, sr
    
    def _detect_bpm(self, audio: np.ndarray, sr: int) -> Tuple[float, float]:
        """Detect BPM using multi-method onset detection and autocorrelation.
        
        Uses:
        1. Spectral flux for onset detection
        2. Multi-band energy analysis
        3. Autocorrelation with harmonic weighting
        4. Tempo octave resolution
        """
        # Use ~30 seconds from middle of track for analysis
        analysis_len = min(len(audio), sr * 30)
        start = max(0, (len(audio) - analysis_len) // 2)
        audio = audio[start:start + analysis_len]
        
        # Parameters
        frame_size = int(sr * 0.023)  # ~23ms (1024 samples at 44.1kHz)
        hop_size = frame_size // 4     # 75% overlap
        
        n_frames = (len(audio) - frame_size) // hop_size
        if n_frames < 50:
            return 0.0, 0.0
        
        # Method 1: Energy-based onset detection (low frequency emphasis)
        # Apply simple low-pass by averaging
        envelope_low = np.array([
            np.sum(audio[i*hop_size:i*hop_size+frame_size]**2)
            for i in range(n_frames)
        ])
        
        # Method 2: High frequency onset detection (for hi-hats/snares)
        # Simple high-pass via differentiation
        diff_audio = np.diff(audio)
        envelope_high = np.array([
            np.sum(diff_audio[i*hop_size:min(i*hop_size+frame_size, len(diff_audio))]**2)
            for i in range(n_frames)
        ])
        
        # Combine envelopes
        envelope = envelope_low + 0.5 * envelope_high[:len(envelope_low)]
        
        # Normalize
        envelope = envelope / (np.max(envelope) + 1e-10)
        
        # Onset detection function (spectral flux style)
        onset = np.diff(envelope)
        onset = np.maximum(0, onset)  # Half-wave rectify
        
        # Apply adaptive threshold
        threshold = np.median(onset) + 0.5 * np.std(onset)
        onset = np.where(onset > threshold, onset, 0)
        
        if len(onset) < 100:
            return 0.0, 0.0
        
        # Autocorrelation
        # Pad for better resolution
        onset_padded = np.concatenate([onset, np.zeros(len(onset))])
        corr = np.correlate(onset_padded, onset_padded, mode='full')
        corr = corr[len(corr)//2:]
        
        # BPM range: 60-200 BPM
        min_bpm, max_bpm = 60, 200
        min_lag = int(60 * sr / hop_size / max_bpm)
        max_lag = int(60 * sr / hop_size / min_bpm)
        
        max_lag = min(max_lag, len(corr) - 1)
        if min_lag >= max_lag:
            return 0.0, 0.0
        
        # Weight by tempo likelihood (prefer 120-140 BPM range)
        search_range = corr[min_lag:max_lag].copy()
        lags = np.arange(min_lag, max_lag)
        bpms = 60 * sr / hop_size / lags
        
        # Gaussian weighting centered at 128 BPM
        tempo_prior = np.exp(-0.5 * ((bpms - 128) / 30)**2)
        search_range = search_range * tempo_prior
        
        # Find peaks (not just max)
        peaks = []
        for i in range(1, len(search_range) - 1):
            if search_range[i] > search_range[i-1] and search_range[i] > search_range[i+1]:
                peaks.append((search_range[i], i + min_lag))
        
        if not peaks:
            peak_idx = np.argmax(search_range) + min_lag
        else:
            # Sort by strength and take best
            peaks.sort(reverse=True)
            peak_idx = peaks[0][1]
        
        # Convert to BPM
        bpm = 60 * sr / hop_size / peak_idx
        
        # Handle tempo octave errors (double/half time)
        # For dance music, prefer tempos between 90-180 BPM
        # If detected tempo is below 90, consider doubling
        # If above 180, consider halving
        
        half_lag = peak_idx * 2
        double_lag = peak_idx // 2
        
        # Get correlation values for comparison
        peak_val = corr[peak_idx]
        
        # Check half-time (slower tempo)
        half_bpm = bpm / 2
        if half_lag < len(corr):
            half_val = corr[half_lag]
        else:
            half_val = 0
            
        # Check double-time (faster tempo)  
        double_bpm = bpm * 2
        if double_lag >= 1:
            double_val = corr[double_lag]
        else:
            double_val = 0
        
        # Decision logic:
        # 1. If detected BPM < 80, strongly prefer doubling if valid
        # 2. If detected BPM > 170, strongly prefer halving if valid
        # 3. Otherwise, prefer the candidate closest to 120-130 BPM
        
        candidates = [(bpm, peak_val, 'original')]
        
        if 60 <= half_bpm <= 200 and half_val > 0:
            candidates.append((half_bpm, half_val, 'half'))
        if 60 <= double_bpm <= 200 and double_val > 0:
            candidates.append((double_bpm, double_val, 'double'))
        
        # Score each candidate
        def score_tempo(tempo, corr_val):
            # Prefer tempos in 90-170 range
            if 90 <= tempo <= 170:
                range_bonus = 1.0
            elif 80 <= tempo <= 180:
                range_bonus = 0.8
            else:
                range_bonus = 0.5
            
            # Prefer tempos near 125 (common dance music tempo)
            center_score = np.exp(-0.5 * ((tempo - 125) / 40)**2)
            
            return corr_val * range_bonus * (0.5 + 0.5 * center_score)
        
        best_bpm = bpm
        best_score = 0
        for tempo, corr_val, label in candidates:
            s = score_tempo(tempo, corr_val)
            if s > best_score:
                best_score = s
                best_bpm = tempo
        
        bpm = best_bpm
        
        # Calculate confidence
        peak_val = corr[peak_idx]
        mean_val = np.mean(corr[min_lag:max_lag])
        std_val = np.std(corr[min_lag:max_lag])
        
        if std_val > 0:
            confidence = min(1.0, (peak_val - mean_val) / (3 * std_val + 1e-10))
        else:
            confidence = 0.5
        
        # Round to nearest 0.5 BPM for cleaner display
        bpm = round(bpm * 2) / 2
        
        return bpm, round(confidence, 2)
    
    def _detect_key(self, audio: np.ndarray, sr: int) -> Tuple[str, float]:
        """Detect musical key using chroma features and key profiles.
        
        Returns (key_string, confidence) where key_string is like "C major" or "A minor"
        """
        # Use middle 30 seconds
        analysis_len = min(len(audio), sr * 30)
        start = max(0, (len(audio) - analysis_len) // 2)
        audio = audio[start:start + analysis_len]
        
        # Parameters
        frame_size = 4096  # Longer for better frequency resolution
        hop_size = 2048
        n_fft = 8192
        
        n_frames = (len(audio) - frame_size) // hop_size
        if n_frames < 10:
            return "", 0.0
        
        # Key names
        key_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        
        # Krumhansl-Schmuckler key profiles
        major_profile = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
        minor_profile = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
        
        # Normalize profiles
        major_profile = major_profile / np.sum(major_profile)
        minor_profile = minor_profile / np.sum(minor_profile)
        
        # Compute chroma features using DFT
        chroma = np.zeros(12)
        
        # Frequency bins for each semitone (A4 = 440Hz reference)
        # We'll sum energy in bins corresponding to each pitch class
        
        for frame_idx in range(n_frames):
            frame_start = frame_idx * hop_size
            frame = audio[frame_start:frame_start + frame_size]
            
            # Apply Hann window
            window = 0.5 * (1 - np.cos(2 * np.pi * np.arange(len(frame)) / len(frame)))
            frame = frame * window
            
            # Zero-pad and FFT
            padded = np.zeros(n_fft)
            padded[:len(frame)] = frame
            spectrum = np.abs(np.fft.rfft(padded))
            
            # Map to chroma
            freqs = np.fft.rfftfreq(n_fft, 1/sr)
            
            for i, freq in enumerate(freqs):
                if freq < 65 or freq > 2000:  # C2 to B6 range
                    continue
                
                # Convert frequency to pitch class
                if freq > 0:
                    midi = 69 + 12 * np.log2(freq / 440)
                    pitch_class = int(round(midi)) % 12
                    chroma[pitch_class] += spectrum[i] ** 2
        
        # Normalize chroma
        chroma = chroma / (np.sum(chroma) + 1e-10)
        
        # Correlate with all key profiles
        best_key = ""
        best_score = -1
        scores = []
        
        for i in range(12):
            # Rotate chroma to match key
            rotated = np.roll(chroma, -i)
            
            # Correlate with major
            major_score = np.corrcoef(rotated, major_profile)[0, 1]
            scores.append((major_score, f"{key_names[i]} major"))
            
            # Correlate with minor  
            minor_score = np.corrcoef(rotated, minor_profile)[0, 1]
            scores.append((minor_score, f"{key_names[i]} minor"))
        
        # Find best match
        scores.sort(reverse=True)
        best_score, best_key = scores[0]
        
        # Confidence is difference between best and second best
        if len(scores) > 1:
            confidence = (best_score - scores[1][0]) / (abs(best_score) + 0.01)
            confidence = min(1.0, max(0.0, confidence + 0.5))  # Normalize
        else:
            confidence = 0.5
        
        return best_key, round(confidence, 2)
    
    def _estimate_danceability(self, audio: np.ndarray, sr: int) -> float:
        """Estimate danceability from rhythmic content."""
        # Use variance of onset envelope as proxy
        frame_size = int(sr * 0.02)
        hop_size = frame_size // 2
        
        n_frames = (len(audio) - frame_size) // hop_size
        if n_frames < 10:
            return 0.5
        
        envelope = np.array([
            np.sum(audio[i*hop_size:i*hop_size+frame_size]**2)
            for i in range(n_frames)
        ])
        
        # Normalize
        envelope = envelope / (np.max(envelope) + 1e-10)
        
        # High variance = more rhythmic = more danceable
        variance = np.var(envelope)
        
        # Scale to 0-1
        return min(1.0, variance * 5)
    
    def _guess_genre(self, analysis: SongAnalysis) -> List[str]:
        """Guess genre hints from analysis."""
        hints = []
        
        # BPM-based hints
        bpm = analysis.bpm
        if 70 <= bpm <= 90:
            hints.append('hip-hop')
        elif 90 <= bpm <= 110:
            hints.append('downtempo')
        elif 110 <= bpm <= 130:
            hints.append('house')
        elif 125 <= bpm <= 140:
            hints.append('techno')
        elif 140 <= bpm <= 160:
            hints.append('trance')
        elif 160 <= bpm <= 180:
            hints.append('drum-and-bass')
        
        # Energy-based hints
        if analysis.energy > 0.7:
            hints.append('energetic')
        elif analysis.energy < 0.3:
            hints.append('ambient')
        
        # Dynamic range hints
        if analysis.dynamic_range_db < 6:
            hints.append('compressed')
        elif analysis.dynamic_range_db > 15:
            hints.append('dynamic')
        
        return hints
    
    def _guess_mood(self, analysis: SongAnalysis) -> List[str]:
        """Guess mood hints from analysis."""
        hints = []
        
        if analysis.energy > 0.7:
            hints.append('intense')
        elif analysis.energy < 0.3:
            hints.append('calm')
        else:
            hints.append('moderate')
        
        if analysis.danceability > 0.6:
            hints.append('groovy')
        
        if analysis.bpm > 140:
            hints.append('fast')
        elif analysis.bpm < 100:
            hints.append('slow')
        
        return hints
    
    def _grade_quality(
        self, 
        filepath: Path, 
        analysis: SongAnalysis
    ) -> Tuple[QualityGrade, List[str]]:
        """Grade audio quality and identify issues."""
        issues = []
        
        # Check sample rate
        if analysis.sample_rate < QUALITY_THRESHOLDS['min_sample_rate']:
            issues.append(f"Low sample rate: {analysis.sample_rate}Hz")
        
        # Check clipping
        if analysis.clipping_ratio > QUALITY_THRESHOLDS['max_clipping_ratio']:
            issues.append(f"Clipping detected: {analysis.clipping_ratio*100:.2f}%")
        
        # Check dynamic range
        if analysis.dynamic_range_db < QUALITY_THRESHOLDS['min_dynamic_range']:
            issues.append(f"Low dynamic range: {analysis.dynamic_range_db:.1f}dB")
        
        # Check DC offset
        if analysis.dc_offset > 0.01:
            issues.append(f"DC offset: {analysis.dc_offset:.4f}")
        
        # Check for mono
        if analysis.channels == 1:
            issues.append("Mono audio")
        
        # Determine grade
        if len(issues) >= 3:
            grade = QualityGrade.LOW
        elif len(issues) >= 1:
            grade = QualityGrade.MEDIUM
        else:
            grade = QualityGrade.HIGH
        
        return grade, issues
    
    # ========================================================================
    # WAV64 CONVERSION
    # ========================================================================
    
    def _convert_to_wav64(self, filepath: Path, file_hash: str) -> Optional[Path]:
        """Convert audio to WAV64 float for high-quality playback."""
        if not self._cache_dir:
            return None
        
        cache_path = self._cache_dir / f"{file_hash}.wav"
        
        # Skip if already cached
        if cache_path.exists():
            return cache_path
        
        try:
            # Load audio
            audio, sr = self._load_audio(filepath)
            
            # Convert to float64
            if audio.dtype != np.float64:
                audio = audio.astype(np.float64)
            
            # Ensure stereo
            if audio.ndim == 1:
                audio = np.column_stack([audio, audio])
            
            # Resample to 48kHz if needed
            if sr != 48000:
                # Simple resampling (proper implementation would use scipy)
                ratio = 48000 / sr
                new_len = int(len(audio) * ratio)
                indices = np.linspace(0, len(audio) - 1, new_len)
                audio = np.array([
                    np.interp(indices, np.arange(len(audio)), audio[:, c])
                    for c in range(audio.shape[1])
                ]).T
                sr = 48000
            
            # Write as WAV (float32 for compatibility)
            self._write_wav_float(cache_path, audio.astype(np.float32), sr)
            
            return cache_path
            
        except Exception as e:
            print(f"Conversion error: {e}")
            return None
    
    def _write_wav_float(self, filepath: Path, audio: np.ndarray, sr: int):
        """Write float32 WAV file."""
        n_channels = audio.shape[1] if audio.ndim > 1 else 1
        n_frames = len(audio)
        
        with wave.open(str(filepath), 'wb') as wf:
            wf.setnchannels(n_channels)
            wf.setsampwidth(4)  # 32-bit
            wf.setframerate(sr)
            
            # Convert float to int32 range
            audio_int = (audio * 2147483647).astype(np.int32)
            wf.writeframes(audio_int.tobytes())
    
    # ========================================================================
    # RETRIEVAL
    # ========================================================================
    
    def get_by_id(self, song_id: int) -> Optional[SongEntry]:
        """Get song by registry ID."""
        return self.songs.get(song_id)
    
    def get_by_index(self, index: int) -> Optional[SongEntry]:
        """Get song by index (1-based for user convenience)."""
        return self.songs.get(index)
    
    def get_by_name(self, name: str) -> List[SongEntry]:
        """Get songs matching name (filename or title)."""
        name_lower = name.lower()
        
        # Exact match first
        if name_lower in self.name_index:
            return [self.songs[id] for id in self.name_index[name_lower]]
        
        # Partial match
        matches = []
        for key, ids in self.name_index.items():
            if name_lower in key:
                matches.extend(ids)
        
        return [self.songs[id] for id in set(matches)]
    
    def search(
        self,
        query: str = "",
        quality: Optional[str] = None,
        min_bpm: float = 0,
        max_bpm: float = 999,
        tags: List[str] = None,
        genre: str = "",
        favorites_only: bool = False,
        limit: int = 50
    ) -> List[SongEntry]:
        """Search songs with filters."""
        results = []
        
        for song in self.songs.values():
            # Skip hidden
            if song.hidden:
                continue
            
            # Quality filter
            if quality and song.quality_grade != quality:
                continue
            
            # Favorites filter
            if favorites_only and not song.favorite:
                continue
            
            # BPM filter
            if song.analysis:
                bpm = song.analysis.get('bpm', 0)
                if bpm < min_bpm or bpm > max_bpm:
                    continue
            
            # Tag filter
            if tags:
                if not any(t in song.tags for t in tags):
                    continue
            
            # Genre filter
            if genre and song.analysis:
                hints = song.analysis.get('genre_hints', [])
                if genre.lower() not in [h.lower() for h in hints]:
                    continue
            
            # Query filter (searches name, title, artist)
            if query:
                query_lower = query.lower()
                if not any(query_lower in field.lower() for field in [
                    song.filename, song.title, song.artist
                ]):
                    continue
            
            results.append(song)
            
            if len(results) >= limit:
                break
        
        return results
    
    def load_song(self, identifier: Union[int, str]) -> Tuple[Optional[np.ndarray], Optional[int], Optional[SongEntry]]:
        """Load song audio by ID or name from registry.
        
        Loading priority:
        1. Cached WAV64 in ~/Documents/MDMA/songs/cache/ (by hash)
        2. Stored cached_path  
        3. Original file path
        
        Returns (audio, sample_rate, entry) or (None, None, None) if not found.
        """
        # Find entry in registry
        if isinstance(identifier, int):
            entry = self.get_by_id(identifier)
        else:
            matches = self.get_by_name(identifier)
            entry = matches[0] if matches else None
        
        if not entry:
            return None, None, None
        
        # Try loading in order of preference
        load_path = None
        
        # 1. Try cache directory by hash first (canonical location)
        if self._cache_dir and entry.hash:
            cache_by_hash = self._cache_dir / f"{entry.hash}.wav"
            if cache_by_hash.exists():
                load_path = cache_by_hash
                # Update cached_path if not set
                if not entry.cached_path:
                    entry.cached_path = str(cache_by_hash)
        
        # 2. Try stored cached_path
        if load_path is None and entry.cached_path:
            cached = Path(entry.cached_path)
            if cached.exists():
                load_path = cached
        
        # 3. Fall back to original path
        if load_path is None:
            original = Path(entry.original_path)
            if original.exists():
                load_path = original
                
                # If we had to use original, re-cache it
                if entry.quality_grade != 'low':
                    try:
                        cached = self._convert_to_wav64(original, entry.hash)
                        if cached:
                            entry.cached_path = str(cached)
                            entry.cached_at = datetime.now().isoformat()
                    except Exception:
                        pass  # Caching failed, but we can still load
        
        # 4. File not found
        if load_path is None:
            entry._load_error = f"File not found. Tried:\n  - Cache: {self._cache_dir / f'{entry.hash}.wav' if self._cache_dir else '(no cache dir)'}\n  - Original: {entry.original_path}"
            return None, None, entry
        
        try:
            audio, sr = self._load_audio(load_path)
            
            # Update play stats
            entry.play_count += 1
            entry.last_played = datetime.now().isoformat()
            self.save()
            
            return audio, sr, entry
        except Exception as e:
            entry._load_error = f"Error loading {load_path}: {e}"
            return None, None, entry
    
    # ========================================================================
    # MANAGEMENT
    # ========================================================================
    
    def update_metadata(
        self,
        song_id: int,
        title: str = None,
        artist: str = None,
        album: str = None,
        tags: List[str] = None,
        rating: int = None,
        favorite: bool = None
    ) -> bool:
        """Update song metadata."""
        if song_id not in self.songs:
            return False
        
        entry = self.songs[song_id]
        
        if title is not None:
            entry.title = title
        if artist is not None:
            entry.artist = artist
        if album is not None:
            entry.album = album
        if tags is not None:
            entry.tags = tags
        if rating is not None:
            entry.rating = max(0, min(5, rating))
        if favorite is not None:
            entry.favorite = favorite
        
        entry.modified_at = datetime.now().isoformat()
        self.save()
        return True
    
    def remove_song(self, song_id: int, delete_cache: bool = True) -> bool:
        """Remove song from registry."""
        if song_id not in self.songs:
            return False
        
        entry = self.songs[song_id]
        
        # Remove cache if requested
        if delete_cache and entry.cached_path:
            try:
                Path(entry.cached_path).unlink(missing_ok=True)
            except Exception:
                pass
        
        # Remove from indexes
        if entry.hash in self.hash_index:
            del self.hash_index[entry.hash]
        
        # Remove from name index
        for key in list(self.name_index.keys()):
            if song_id in self.name_index[key]:
                self.name_index[key].remove(song_id)
                if not self.name_index[key]:
                    del self.name_index[key]
        
        # Remove entry
        del self.songs[song_id]
        
        self.save()
        return True
    
    def get_stats(self) -> Dict[str, Any]:
        """Get registry statistics."""
        quality_counts = {'high': 0, 'medium': 0, 'low': 0}
        total_duration = 0.0
        total_size = 0
        cached_in_registry = 0
        
        for song in self.songs.values():
            quality_counts[song.quality_grade] += 1
            total_size += song.file_size
            if song.analysis:
                total_duration += song.analysis.get('duration', 0)
            if song.cached_path:
                cached_in_registry += 1
        
        # Count actual cache files
        actual_cache_files = 0
        cache_size = 0
        if self._cache_dir and self._cache_dir.exists():
            for f in self._cache_dir.glob('*.wav'):
                actual_cache_files += 1
                cache_size += f.stat().st_size
        
        return {
            'total_songs': len(self.songs),
            'quality': quality_counts,
            'total_duration_hours': total_duration / 3600,
            'total_size_gb': total_size / (1024**3),
            'cached_songs': cached_in_registry,
            'actual_cache_files': actual_cache_files,
            'cache_size_gb': cache_size / (1024**3),
            'cache_synced': cached_in_registry == actual_cache_files,
        }
    
    def rebuild_cache(self, progress_callback=None) -> Dict[str, Any]:
        """Rebuild entire cache from registered songs.
        
        Returns stats about the rebuild.
        """
        results = {
            'total': len(self.songs),
            'cached': 0,
            'skipped': 0,
            'failed': 0,
            'errors': [],
        }
        
        for i, song in enumerate(self.songs.values()):
            if progress_callback:
                progress_callback(i + 1, results['total'], song.filename)
            
            # Skip low quality
            if song.quality_grade == 'low':
                results['skipped'] += 1
                continue
            
            # Check if already cached
            if self._cache_dir:
                cache_path = self._cache_dir / f"{song.hash}.wav"
                if cache_path.exists():
                    song.cached_path = str(cache_path)
                    results['cached'] += 1
                    continue
            
            # Try to cache
            original = Path(song.original_path)
            if original.exists():
                try:
                    cached = self._convert_to_wav64(original, song.hash)
                    if cached:
                        song.cached_path = str(cached)
                        song.cached_at = datetime.now().isoformat()
                        results['cached'] += 1
                    else:
                        results['failed'] += 1
                except Exception as e:
                    results['failed'] += 1
                    results['errors'].append(f"{song.filename}: {e}")
            else:
                results['failed'] += 1
                results['errors'].append(f"{song.filename}: original file not found")
        
        self.save()
        return results


# ============================================================================
# GLOBAL INSTANCE
# ============================================================================

_registry: Optional[SongRegistry] = None


def get_registry() -> SongRegistry:
    """Get or create global song registry."""
    global _registry
    if _registry is None:
        _registry = SongRegistry()
    return _registry


# ============================================================================
# COMMAND INTERFACE
# ============================================================================

def cmd_reg(args: list) -> str:
    """Song registry command interface.
    
    Usage:
      /reg                    Show registry stats
      /reg scan <folder>      Scan folder for songs
      /reg list [query]       List/search songs
      /reg info <id>          Show song details
      /reg load <id|name>     Load song to deck
      /reg tag <id> <tags>    Add tags to song
      /reg rate <id> <1-5>    Rate song
      /reg fav <id>           Toggle favorite
      /reg quality            Show quality breakdown
      /reg cache              Show cache status
      /reg cache rebuild      Rebuild all cache files
      /reg export             Export registry as CSV
    
    Examples:
      /reg scan ~/Music
      /reg list techno
      /reg load 42
      /reg load "aphex twin"
      /reg tag 42 idm ambient
      /reg rate 42 5
    """
    registry = get_registry()
    
    if not args:
        # Show stats
        stats = registry.get_stats()
        cache_status = "✓ synced" if stats['cache_synced'] else f"⚠ {stats['actual_cache_files']}/{stats['cached_songs']} files"
        return (f"=== SONG REGISTRY ===\n"
                f"  Total songs: {stats['total_songs']}\n"
                f"  Quality: {stats['quality']['high']} high, "
                f"{stats['quality']['medium']} medium, "
                f"{stats['quality']['low']} low\n"
                f"  Duration: {stats['total_duration_hours']:.1f} hours\n"
                f"  Size: {stats['total_size_gb']:.2f} GB\n"
                f"  Cache: {stats['cached_songs']} songs ({stats['cache_size_gb']:.2f} GB) {cache_status}")
    
    cmd = args[0].lower()
    
    # Cache management
    if cmd == 'cache':
        if len(args) > 1 and args[1].lower() == 'rebuild':
            print("Rebuilding cache...")
            
            def progress(current, total, name):
                if current % 5 == 0:
                    print(f"  Caching {current}/{total}: {name[:30]}")
            
            results = registry.rebuild_cache(progress_callback=progress)
            
            return (f"=== CACHE REBUILD COMPLETE ===\n"
                    f"  Total: {results['total']}\n"
                    f"  Cached: {results['cached']}\n"
                    f"  Skipped (low quality): {results['skipped']}\n"
                    f"  Failed: {results['failed']}")
        else:
            # Show cache status
            stats = registry.get_stats()
            
            lines = ["=== CACHE STATUS ==="]
            lines.append(f"  Registered songs: {stats['total_songs']}")
            lines.append(f"  Songs with cache path: {stats['cached_songs']}")
            lines.append(f"  Actual cache files: {stats['actual_cache_files']}")
            lines.append(f"  Cache size: {stats['cache_size_gb']:.2f} GB")
            
            if stats['cache_synced']:
                lines.append(f"  Status: ✓ Cache is synchronized")
            else:
                missing = stats['cached_songs'] - stats['actual_cache_files']
                lines.append(f"  Status: ⚠ {abs(missing)} files need attention")
                lines.append(f"  Run /reg cache rebuild to fix")
            
            return "\n".join(lines)
    
    if cmd == 'scan':
        if len(args) < 2:
            return "Usage: /reg scan <folder>"
        folder = ' '.join(args[1:])
        
        def progress(current, total, name):
            if current % 10 == 0:
                print(f"  Scanning {current}/{total}: {name[:30]}")
        
        results = registry.scan_folder(folder, progress_callback=progress)
        
        if 'error' in results:
            return f"ERROR: {results['error']}"
        
        return (f"=== SCAN COMPLETE ===\n"
                f"  Scanned: {results['scanned']}\n"
                f"  Added: {results['added']}\n"
                f"  Skipped: {results['skipped']}\n"
                f"  Updated: {results['updated']}\n"
                f"  Quality: {results['quality']['high']} high, "
                f"{results['quality']['medium']} medium, "
                f"{results['quality']['low']} low\n"
                f"  Errors: {len(results['errors'])}")
    
    elif cmd == 'list':
        query = ' '.join(args[1:]) if len(args) > 1 else ""
        songs = registry.search(query=query, limit=20)
        
        if not songs:
            return "No songs found"
        
        lines = ["=== SONGS ==="]
        for song in songs:
            quality = song.quality_grade[0].upper()
            bpm = song.analysis.get('bpm', 0) if song.analysis else 0
            key = song.analysis.get('key', '') if song.analysis else ''
            key_short = key.split()[0] if key else ''  # Just note name
            fav = '★' if song.favorite else ' '
            key_str = f" {key_short:3}" if key_short else "    "
            lines.append(f"  {song.id:4d} [{quality}] {fav} {song.filename[:35]}{key_str} {bpm:3.0f}bpm")
        
        return "\n".join(lines)
    
    elif cmd == 'info':
        if len(args) < 2:
            return "Usage: /reg info <id>"
        try:
            song_id = int(args[1])
        except ValueError:
            return "ERROR: Invalid song ID"
        
        song = registry.get_by_id(song_id)
        if not song:
            return f"Song {song_id} not found"
        
        lines = [f"=== SONG {song.id} ==="]
        lines.append(f"  Hash: {song.hash}")
        lines.append(f"  File: {song.filename}")
        lines.append(f"  Path: {song.original_path}")
        
        # Check file status
        original_exists = Path(song.original_path).exists() if song.original_path else False
        cache_exists = False
        if registry._cache_dir and song.hash:
            cache_path = registry._cache_dir / f"{song.hash}.wav"
            cache_exists = cache_path.exists()
        
        lines.append(f"  Original exists: {'✓' if original_exists else '✗'}")
        lines.append(f"  Cache exists: {'✓' if cache_exists else '✗'}")
        
        lines.append(f"  Title: {song.title or '(unknown)'}")
        lines.append(f"  Artist: {song.artist or '(unknown)'}")
        lines.append(f"  Quality: {song.quality_grade.upper()}")
        if song.quality_issues:
            lines.append(f"  Issues: {', '.join(song.quality_issues)}")
        if song.analysis:
            a = song.analysis
            lines.append(f"  BPM: {a.get('bpm', 0):.1f} (conf: {a.get('bpm_confidence', 0):.0%})")
            key = a.get('key', '')
            if key:
                lines.append(f"  Key: {key} (conf: {a.get('key_confidence', 0):.0%})")
            lines.append(f"  Duration: {a.get('duration', 0):.1f}s")
            lines.append(f"  Energy: {a.get('energy', 0):.0%}")
            lines.append(f"  Danceability: {a.get('danceability', 0):.0%}")
            if a.get('genre_hints'):
                lines.append(f"  Genre hints: {', '.join(a['genre_hints'])}")
            if a.get('mood_hints'):
                lines.append(f"  Mood hints: {', '.join(a['mood_hints'])}")
        lines.append(f"  Tags: {', '.join(song.tags) if song.tags else '(none)'}")
        lines.append(f"  Rating: {'★' * song.rating}{'☆' * (5-song.rating)}")
        lines.append(f"  Plays: {song.play_count}")
        
        return "\n".join(lines)
    
    elif cmd == 'load':
        if len(args) < 2:
            return "Usage: /reg load <id|name> [deck|buf]"
        
        identifier = args[1]
        deck_id = None
        load_to_buffer = False
        
        # Check if last arg is deck number or 'buf'/'buffer'
        if len(args) >= 3:
            last_arg = args[-1].lower()
            if last_arg in ('buf', 'buffer', 'b'):
                load_to_buffer = True
                identifier = ' '.join(args[1:-1])
            else:
                try:
                    deck_id = int(args[-1])
                    identifier = ' '.join(args[1:-1])
                except ValueError:
                    identifier = ' '.join(args[1:])
        
        try:
            identifier = int(identifier)
        except ValueError:
            pass  # Keep as string for name search
        
        audio, sr, entry = registry.load_song(identifier)
        
        if audio is None:
            if entry:
                # Get detailed error if available
                error_detail = getattr(entry, '_load_error', f"Could not load: {entry.filename}")
                return f"ERROR: {error_detail}"
            return f"ERROR: Song not found in registry: {identifier}\nUse /reg list to see available songs"
        
        import numpy as np
        
        # Convert to mono float64 for MDMA
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)
        audio = audio.astype(np.float64)
        
        # Normalize if needed
        max_val = np.max(np.abs(audio))
        if max_val > 1.0:
            audio = audio / max_val * 0.95
        
        # Get analysis info
        key_str = ""
        if entry.analysis and entry.analysis.get('key'):
            key_str = f"\n  Key: {entry.analysis['key']}"
        bpm = entry.analysis.get('bpm', 0) if entry.analysis else 0
        
        # Load to buffer if requested or no deck context
        if load_to_buffer:
            # Load directly to working buffer
            try:
                from ..commands.working_cmds import get_working_buffer
                wb = get_working_buffer()
                wb.set_pending(audio, f"reg:{entry.filename}", None)
            except:
                pass
            
            return (f"OK: Loaded song {entry.id} to working buffer\n"
                    f"  {entry.filename}\n"
                    f"  {len(audio)/sr:.1f}s @ {sr}Hz\n"
                    f"  BPM: {bpm:.1f}{key_str}\n"
                    f"  Use /a to commit to buffer")
        
        # Try to load into DJ deck
        try:
            from ..dsp.dj_mode import get_dj_engine, DJDeck
            
            dj = get_dj_engine(48000)
            
            # Check if deck is actually in use/selected
            has_deck_context = dj.enabled and dj.active_deck in dj.decks
            
            if not has_deck_context and deck_id is None:
                # No deck context - load to working buffer instead
                try:
                    from ..commands.working_cmds import get_working_buffer
                    wb = get_working_buffer()
                    wb.set_pending(audio, f"reg:{entry.filename}", None)
                except:
                    pass
                
                return (f"OK: Loaded song {entry.id} to working buffer (no deck selected)\n"
                        f"  {entry.filename}\n"
                        f"  {len(audio)/sr:.1f}s @ {sr}Hz\n"
                        f"  BPM: {bpm:.1f}{key_str}\n"
                        f"  Use /a to commit to buffer, or /deck to enable decks")
            
            if not dj.enabled:
                dj.enabled = True
            
            # Use specified deck or active deck
            if deck_id is None:
                deck_id = dj.active_deck
            
            # Create deck if needed
            if deck_id not in dj.decks:
                dj.decks[deck_id] = DJDeck(id=deck_id)
            
            deck = dj.decks[deck_id]
            
            # Convert to float64 for deck
            deck.buffer = audio.astype(np.float64)
            deck.position = 0.0
            deck.playing = False
            
            # Set tempo from analysis
            if entry.analysis and entry.analysis.get('bpm', 0) > 0:
                deck.tempo = entry.analysis['bpm']
            
            return (f"OK: Loaded song {entry.id} to deck {deck_id}\n"
                    f"  {entry.filename}\n"
                    f"  {len(audio)/sr:.1f}s @ {sr}Hz\n"
                    f"  BPM: {deck.tempo:.1f}{key_str}\n"
                    f"  Use /play to start playback")
        except ImportError:
            # No DJ mode available - load to working buffer
            try:
                from ..commands.working_cmds import get_working_buffer
                wb = get_working_buffer()
                wb.set_pending(audio, f"reg:{entry.filename}", None)
            except:
                pass
            
            return (f"OK: Loaded song {entry.id} to working buffer\n"
                    f"  {entry.filename}\n"
                    f"  {len(audio)/sr:.1f}s @ {sr}Hz\n"
                    f"  BPM: {bpm:.1f}{key_str}\n"
                    f"  Use /a to commit to buffer")
    
    elif cmd == 'tag':
        if len(args) < 3:
            return "Usage: /reg tag <id> <tags...>"
        try:
            song_id = int(args[1])
        except ValueError:
            return "ERROR: Invalid song ID"
        
        tags = args[2:]
        song = registry.get_by_id(song_id)
        if not song:
            return f"Song {song_id} not found"
        
        song.tags = list(set(song.tags + tags))
        registry.save()
        return f"OK: Added tags to song {song_id}: {', '.join(tags)}"
    
    elif cmd == 'rate':
        if len(args) < 3:
            return "Usage: /reg rate <id> <1-5>"
        try:
            song_id = int(args[1])
            rating = int(args[2])
        except ValueError:
            return "ERROR: Invalid song ID or rating"
        
        if registry.update_metadata(song_id, rating=rating):
            return f"OK: Song {song_id} rated {'★' * rating}"
        return f"Song {song_id} not found"
    
    elif cmd == 'fav':
        if len(args) < 2:
            return "Usage: /reg fav <id>"
        try:
            song_id = int(args[1])
        except ValueError:
            return "ERROR: Invalid song ID"
        
        song = registry.get_by_id(song_id)
        if not song:
            return f"Song {song_id} not found"
        
        registry.update_metadata(song_id, favorite=not song.favorite)
        return f"OK: Song {song_id} {'added to' if not song.favorite else 'removed from'} favorites"
    
    elif cmd == 'quality':
        stats = registry.get_stats()
        total = stats['total_songs']
        if total == 0:
            return "No songs in registry"
        
        high = stats['quality']['high']
        med = stats['quality']['medium']
        low = stats['quality']['low']
        
        return (f"=== QUALITY BREAKDOWN ===\n"
                f"  HIGH:   {high:4d} ({high*100//total}%) - Ready for performance\n"
                f"  MEDIUM: {med:4d} ({med*100//total}%) - Usable with minor issues\n"
                f"  LOW:    {low:4d} ({low*100//total}%) - Not recommended")
    
    elif cmd == 'high':
        # List only high quality songs
        songs = registry.search(quality='high', limit=30)
        if not songs:
            return "No high-quality songs in registry"
        
        lines = ["=== HIGH QUALITY SONGS ==="]
        for song in songs:
            bpm = song.analysis.get('bpm', 0) if song.analysis else 0
            fav = '★' if song.favorite else ' '
            lines.append(f"  {song.id:4d} {fav} {song.filename[:40]} ({bpm:.0f}bpm)")
        return "\n".join(lines)
    
    elif cmd == 'fix':
        if len(args) < 2:
            return "Usage: /reg fix <id>"
        try:
            song_id = int(args[1])
        except ValueError:
            return "ERROR: Invalid song ID"
        
        song = registry.get_by_id(song_id)
        if not song:
            return f"Song {song_id} not found"
        
        # Re-analyze
        filepath = Path(song.original_path)
        if not filepath.exists():
            return f"ERROR: File not found: {song.original_path}"
        
        analysis = registry.analyze_file(filepath)
        song.analysis = asdict(analysis)
        
        # Re-grade quality
        grade, issues = registry._grade_quality(filepath, analysis)
        old_grade = song.quality_grade
        song.quality_grade = grade.value
        song.quality_issues = issues
        
        # Re-convert to cache
        if song.quality_grade != 'low':
            cached = registry._convert_to_wav64(filepath, song.hash)
            if cached:
                song.cached_path = str(cached)
                song.cached_at = datetime.now().isoformat()
        
        registry.save()
        
        return (f"OK: Re-analyzed song {song_id}\n"
                f"  Quality: {old_grade} -> {song.quality_grade}\n"
                f"  BPM: {analysis.bpm:.1f}\n"
                f"  Issues: {len(issues)}")
    
    elif cmd == 'remove':
        if len(args) < 2:
            return "Usage: /reg remove <id>"
        try:
            song_id = int(args[1])
        except ValueError:
            return "ERROR: Invalid song ID"
        
        song = registry.get_by_id(song_id)
        if not song:
            return f"Song {song_id} not found"
        
        filename = song.filename
        if registry.remove_song(song_id):
            return f"OK: Removed song {song_id} ({filename})"
        return f"ERROR: Could not remove song {song_id}"
    
    elif cmd == 'rescan':
        # Re-scan all registered songs and update status
        updated = 0
        missing = 0
        
        for song_id, song in list(registry.songs.items()):
            filepath = Path(song.original_path)
            
            if not filepath.exists():
                song.hidden = True
                missing += 1
            elif song.hidden:
                song.hidden = False
                updated += 1
        
        registry.save()
        return f"OK: Rescan complete\n  Updated: {updated}\n  Missing: {missing}"
    
    elif cmd == 'bpm':
        # Filter by BPM range
        if len(args) < 2:
            return "Usage: /reg bpm <min>-<max> or /reg bpm <target>"
        
        bpm_arg = args[1]
        if '-' in bpm_arg:
            parts = bpm_arg.split('-')
            try:
                min_bpm = float(parts[0])
                max_bpm = float(parts[1])
            except ValueError:
                return "ERROR: Invalid BPM range"
        else:
            try:
                target = float(bpm_arg)
                min_bpm = target - 5
                max_bpm = target + 5
            except ValueError:
                return "ERROR: Invalid BPM value"
        
        songs = registry.search(min_bpm=min_bpm, max_bpm=max_bpm, limit=20)
        if not songs:
            return f"No songs in BPM range {min_bpm:.0f}-{max_bpm:.0f}"
        
        lines = [f"=== SONGS {min_bpm:.0f}-{max_bpm:.0f} BPM ==="]
        for song in songs:
            quality = song.quality_grade[0].upper()
            bpm = song.analysis.get('bpm', 0) if song.analysis else 0
            lines.append(f"  {song.id:4d} [{quality}] {song.filename[:35]} ({bpm:.0f}bpm)")
        return "\n".join(lines)
    
    elif cmd == 'genre':
        # Filter by genre hint
        if len(args) < 2:
            return "Usage: /reg genre <genre>"
        
        genre = args[1].lower()
        songs = registry.search(genre=genre, limit=20)
        
        if not songs:
            return f"No songs with genre hint '{genre}'"
        
        lines = [f"=== {genre.upper()} SONGS ==="]
        for song in songs:
            quality = song.quality_grade[0].upper()
            bpm = song.analysis.get('bpm', 0) if song.analysis else 0
            lines.append(f"  {song.id:4d} [{quality}] {song.filename[:35]} ({bpm:.0f}bpm)")
        return "\n".join(lines)
    
    elif cmd == 'key':
        # Filter by musical key
        if len(args) < 2:
            return "Usage: /reg key <key>\nExamples: /reg key C, /reg key Am, /reg key \"F# minor\""
        
        key_query = ' '.join(args[1:]).lower()
        
        # Normalize common key formats
        key_map = {
            'c': 'c major', 'cm': 'c minor', 'c#': 'c# major', 'c#m': 'c# minor',
            'd': 'd major', 'dm': 'd minor', 'd#': 'd# major', 'd#m': 'd# minor',
            'e': 'e major', 'em': 'e minor', 'eb': 'd# major', 'ebm': 'd# minor',
            'f': 'f major', 'fm': 'f minor', 'f#': 'f# major', 'f#m': 'f# minor',
            'g': 'g major', 'gm': 'g minor', 'g#': 'g# major', 'g#m': 'g# minor',
            'a': 'a major', 'am': 'a minor', 'a#': 'a# major', 'a#m': 'a# minor',
            'b': 'b major', 'bm': 'b minor', 'bb': 'a# major', 'bbm': 'a# minor',
        }
        
        # Normalize query
        key_search = key_map.get(key_query.replace(' ', ''), key_query)
        
        # Search for matching keys
        songs = []
        for song in registry.songs.values():
            if song.hidden:
                continue
            if song.analysis and song.analysis.get('key'):
                song_key = song.analysis['key'].lower()
                if key_search in song_key or song_key.startswith(key_search.split()[0]):
                    songs.append(song)
            if len(songs) >= 20:
                break
        
        if not songs:
            return f"No songs in key '{key_query}'"
        
        lines = [f"=== SONGS IN {key_query.upper()} ==="]
        for song in songs:
            quality = song.quality_grade[0].upper()
            bpm = song.analysis.get('bpm', 0) if song.analysis else 0
            key = song.analysis.get('key', '') if song.analysis else ''
            lines.append(f"  {song.id:4d} [{quality}] {song.filename[:30]} {key} ({bpm:.0f}bpm)")
        return "\n".join(lines)
    
    elif cmd == 'recent':
        # Show recently added songs
        songs = sorted(registry.songs.values(), 
                      key=lambda s: s.added_at, 
                      reverse=True)[:15]
        
        if not songs:
            return "No songs in registry"
        
        lines = ["=== RECENTLY ADDED ==="]
        for song in songs:
            quality = song.quality_grade[0].upper()
            bpm = song.analysis.get('bpm', 0) if song.analysis else 0
            lines.append(f"  {song.id:4d} [{quality}] {song.filename[:35]} ({bpm:.0f}bpm)")
        return "\n".join(lines)
    
    elif cmd == 'played':
        # Show most played songs
        songs = sorted(registry.songs.values(),
                      key=lambda s: s.play_count,
                      reverse=True)[:15]
        
        if not songs or songs[0].play_count == 0:
            return "No play history yet"
        
        lines = ["=== MOST PLAYED ==="]
        for song in songs:
            if song.play_count == 0:
                break
            quality = song.quality_grade[0].upper()
            lines.append(f"  {song.id:4d} [{quality}] {song.play_count:3d}x {song.filename[:30]}")
        return "\n".join(lines)
    
    elif cmd == 'favs':
        # Show favorite songs
        songs = [s for s in registry.songs.values() if s.favorite]
        
        if not songs:
            return "No favorites yet. Use /reg fav <id> to add."
        
        lines = ["=== FAVORITES ==="]
        for song in songs:
            quality = song.quality_grade[0].upper()
            bpm = song.analysis.get('bpm', 0) if song.analysis else 0
            lines.append(f"  {song.id:4d} [{quality}] ★ {song.filename[:35]} ({bpm:.0f}bpm)")
        return "\n".join(lines)
    
    elif cmd == 'verify':
        # Verify registry integrity and fix issues
        lines = ["=== REGISTRY VERIFICATION ==="]
        issues_found = 0
        issues_fixed = 0
        
        # Check 1: Verify all IDs are unique and sequential
        ids = sorted(registry.songs.keys())
        if ids:
            lines.append(f"  ID range: {min(ids)} - {max(ids)}")
            if len(ids) != len(set(ids)):
                lines.append("  ⚠ Duplicate IDs found!")
                issues_found += 1
        
        # Check 2: Verify hash index matches
        hash_issues = []
        for song_id, song in registry.songs.items():
            if song.hash:
                if song.hash not in registry.hash_index:
                    registry.hash_index[song.hash] = song_id
                    issues_fixed += 1
                elif registry.hash_index[song.hash] != song_id:
                    hash_issues.append(f"Hash collision: {song.hash}")
        if hash_issues:
            lines.append(f"  ⚠ {len(hash_issues)} hash index issues")
            issues_found += len(hash_issues)
        
        # Check 3: Verify cache files exist
        cache_ok = 0
        cache_missing = 0
        cache_orphan = 0
        
        # Get all cache files
        cache_files = set()
        if registry._cache_dir and registry._cache_dir.exists():
            cache_files = {f.stem for f in registry._cache_dir.glob('*.wav')}
        
        # Check each song
        for song in registry.songs.values():
            if song.hash in cache_files:
                cache_ok += 1
                # Ensure cached_path is set correctly
                expected_path = str(registry._cache_dir / f"{song.hash}.wav")
                if song.cached_path != expected_path:
                    song.cached_path = expected_path
                    issues_fixed += 1
            elif song.cached_path:
                # Has cached_path but no file
                if not Path(song.cached_path).exists():
                    song.cached_path = ""
                    cache_missing += 1
                    issues_fixed += 1
        
        # Check for orphan cache files
        registered_hashes = {s.hash for s in registry.songs.values() if s.hash}
        orphan_files = cache_files - registered_hashes
        cache_orphan = len(orphan_files)
        
        lines.append(f"  Cache files: {cache_ok} OK, {cache_missing} missing, {cache_orphan} orphan")
        
        # Check 4: Verify original files exist
        originals_ok = 0
        originals_missing = 0
        for song in registry.songs.values():
            if Path(song.original_path).exists():
                originals_ok += 1
            else:
                originals_missing += 1
        lines.append(f"  Original files: {originals_ok} OK, {originals_missing} missing")
        
        # Check 5: Verify next_id is correct
        if ids:
            expected_next = max(ids) + 1
            if registry.next_id < expected_next:
                registry.next_id = expected_next
                lines.append(f"  Fixed next_id: {registry.next_id}")
                issues_fixed += 1
        
        # Save if we fixed anything
        if issues_fixed > 0:
            registry.save()
            lines.append(f"\n  ✓ Fixed {issues_fixed} issues")
        
        if issues_found == 0 and issues_fixed == 0:
            lines.append("\n  ✓ Registry is healthy")
        
        return "\n".join(lines)
    
    else:
        return f"Unknown command: {cmd}. Use /reg for help."


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    'SongRegistry',
    'SongEntry',
    'SongAnalysis',
    'QualityGrade',
    'get_registry',
    'cmd_reg',
]
