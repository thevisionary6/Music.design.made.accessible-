"""MDMA Streaming Library - Audio streaming and library management.

Supports:
- SoundCloud streaming (via yt-dlp/soundcloud-dl)
- YouTube audio extraction
- Local file scanning
- Metadata extraction
- Pre-processing to engine native format (float64 wave)

All streamed audio is converted to engine-native format before use.
"""

from __future__ import annotations

import os
import json
import hashlib
import tempfile
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from enum import Enum
from urllib.parse import urlparse
import threading
import time


# ============================================================================
# TRACK METADATA
# ============================================================================

class TrackSource(Enum):
    LOCAL = "local"
    SOUNDCLOUD = "soundcloud"
    YOUTUBE = "youtube"
    BANDCAMP = "bandcamp"
    URL = "url"
    GENERATED = "generated"


@dataclass
class TrackInfo:
    """Metadata for a track in the library."""
    id: str
    title: str
    artist: str = ""
    duration: float = 0.0
    bpm: float = 0.0
    key: str = ""
    source: TrackSource = TrackSource.LOCAL
    url: str = ""
    local_path: str = ""
    thumbnail_url: str = ""
    tags: List[str] = field(default_factory=list)
    waveform_data: Optional[np.ndarray] = None
    analyzed: bool = False
    
    # Analysis results (populated after analysis)
    loudness: float = 0.0
    energy: float = 0.0
    danceability: float = 0.0


@dataclass
class StreamBuffer:
    """Buffer for streamed audio in engine-native format."""
    audio: np.ndarray  # float64, mono
    sample_rate: int = 48000
    track_info: Optional[TrackInfo] = None
    ready: bool = False
    error: Optional[str] = None
    registry_id: Optional[int] = None  # ID if registered
    
    @property
    def duration(self) -> float:
        if self.audio is None:
            return 0.0
        return len(self.audio) / self.sample_rate


# ============================================================================
# AUTO-REGISTRATION
# ============================================================================

def auto_register_stream(
    buffer: StreamBuffer,
    source_file: Optional[Path] = None,
    source_type: str = "stream"
) -> Optional[int]:
    """Automatically register streamed audio to song registry.
    
    Parameters
    ----------
    buffer : StreamBuffer
        The stream buffer with audio and metadata
    source_file : Path, optional
        The downloaded file path (for caching)
    source_type : str
        Source type (soundcloud, youtube, etc.)
    
    Returns
    -------
    int or None
        Registry ID if registered, None otherwise
    """
    if not buffer.ready or buffer.audio is None or len(buffer.audio) == 0:
        return None
    
    try:
        from ..core.song_registry import get_registry, SongEntry
        from ..core.user_data import get_mdma_root
        from dataclasses import asdict
        from datetime import datetime
        import wave
        
        registry = get_registry()
        
        # Create songs directory if needed
        songs_dir = get_mdma_root() / 'songs' / 'downloads'
        songs_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate filename from track info
        if buffer.track_info:
            safe_title = "".join(c for c in buffer.track_info.title if c.isalnum() or c in ' -_')[:50]
            safe_artist = "".join(c for c in buffer.track_info.artist if c.isalnum() or c in ' -_')[:30]
            if safe_artist:
                filename = f"{safe_artist} - {safe_title}.wav"
            else:
                filename = f"{safe_title}.wav"
        else:
            filename = f"{source_type}_{hashlib.md5(str(time.time()).encode()).hexdigest()[:8]}.wav"
        
        # Save as WAV to songs/downloads
        save_path = songs_dir / filename
        
        # Avoid overwriting
        counter = 1
        while save_path.exists():
            stem = save_path.stem.rsplit('_', 1)[0]
            save_path = songs_dir / f"{stem}_{counter}.wav"
            counter += 1
        
        # Convert to stereo float32 for saving
        audio = buffer.audio
        if audio.ndim == 1:
            audio = np.column_stack([audio, audio])
        
        # Write WAV file
        audio_int = (audio * 32767).astype(np.int16)
        with wave.open(str(save_path), 'wb') as wf:
            wf.setnchannels(2)
            wf.setsampwidth(2)
            wf.setframerate(buffer.sample_rate)
            wf.writeframes(audio_int.tobytes())
        
        # Compute hash
        file_hash = registry._compute_hash(save_path)
        
        # Check if already registered
        if file_hash in registry.hash_index:
            return registry.hash_index[file_hash]
        
        # Create entry
        entry = SongEntry(
            id=registry.next_id,
            hash=file_hash,
            original_path=str(save_path),
            filename=save_path.name,
            format='wav',
            file_size=save_path.stat().st_size,
            added_at=datetime.now().isoformat(),
            modified_at=datetime.now().isoformat(),
        )
        
        # Add metadata from track info
        if buffer.track_info:
            entry.title = buffer.track_info.title
            entry.artist = buffer.track_info.artist
            entry.tags = [source_type]
            if buffer.track_info.url:
                entry.tags.append('stream')
        
        # Run full analysis
        analysis = registry.analyze_file(save_path)
        entry.analysis = asdict(analysis)
        
        # Grade quality
        grade, issues = registry._grade_quality(save_path, analysis)
        entry.quality_grade = grade.value
        entry.quality_issues = issues
        
        # Convert to WAV64 cache if high/medium quality
        if entry.quality_grade != 'low':
            cached = registry._convert_to_wav64(save_path, file_hash)
            if cached:
                entry.cached_path = str(cached)
                entry.cached_at = datetime.now().isoformat()
        
        # Register
        registry._register_entry(entry)
        registry.save()
        
        return entry.id
        
    except Exception as e:
        # Registration failed, but streaming still works
        print(f"Auto-registration failed: {e}")
        return None


# ============================================================================
# AUDIO CONVERSION
# ============================================================================

def to_engine_format(
    audio: np.ndarray,
    source_rate: int,
    target_rate: int = 48000,
) -> np.ndarray:
    """Convert audio to engine-native format.
    
    Engine format:
    - float64
    - Mono
    - Target sample rate (default 48000)
    - Normalized to -0.95/+0.95
    
    Parameters
    ----------
    audio : np.ndarray
        Input audio (any format)
    source_rate : int
        Source sample rate
    target_rate : int
        Target sample rate
    
    Returns
    -------
    np.ndarray
        Engine-native format audio
    """
    # Convert to float64
    if audio.dtype == np.int16:
        audio = audio.astype(np.float64) / 32768.0
    elif audio.dtype == np.int32:
        audio = audio.astype(np.float64) / 2147483648.0
    elif audio.dtype == np.uint8:
        audio = (audio.astype(np.float64) - 128) / 128.0
    else:
        audio = audio.astype(np.float64)
    
    # Convert to mono
    if len(audio.shape) > 1:
        if audio.shape[0] == 2:  # Channels first
            audio = audio.mean(axis=0)
        elif audio.shape[1] == 2:  # Channels last
            audio = audio.mean(axis=1)
        else:
            audio = audio.flatten()
    
    # Resample if needed
    if source_rate != target_rate:
        ratio = target_rate / source_rate
        new_len = int(len(audio) * ratio)
        x_old = np.linspace(0, 1, len(audio))
        x_new = np.linspace(0, 1, new_len)
        audio = np.interp(x_new, x_old, audio)
    
    # Normalize
    peak = np.max(np.abs(audio))
    if peak > 0:
        audio = audio / peak * 0.95
    
    return audio.astype(np.float64)


def load_audio_file(
    path: Union[str, Path],
    target_rate: int = 48000,
) -> Tuple[np.ndarray, int]:
    """Load audio file and convert to engine format.
    
    Supports: wav, mp3, flac, ogg, m4a, aiff
    
    Returns (audio, sample_rate).
    
    Raises
    ------
    FileNotFoundError
        If file doesn't exist
    PermissionError
        If file cannot be read (access denied)
    RuntimeError
        If file format is not supported
    """
    path = Path(path)
    
    # Check file exists
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    
    # Check if it's a directory
    if path.is_dir():
        raise IsADirectoryError(f"Cannot load directory as audio: {path}")
    
    # Check read permission
    if not os.access(path, os.R_OK):
        raise PermissionError(f"Permission denied - cannot read file: {path}")
    
    errors = []
    
    # Try soundfile first (best for wav, flac, ogg)
    try:
        import soundfile as sf
        audio, sr = sf.read(str(path))
        return to_engine_format(audio, sr, target_rate), target_rate
    except PermissionError:
        raise  # Re-raise permission errors immediately
    except Exception as e:
        errors.append(f"soundfile: {e}")
    
    # Try librosa (handles mp3, m4a)
    try:
        import librosa
        audio, sr = librosa.load(str(path), sr=None, mono=False)
        return to_engine_format(audio, sr, target_rate), target_rate
    except PermissionError:
        raise
    except Exception as e:
        errors.append(f"librosa: {e}")
    
    # Try pydub as fallback
    try:
        from pydub import AudioSegment
        
        segment = AudioSegment.from_file(str(path))
        sr = segment.frame_rate
        
        # Convert to numpy
        samples = np.array(segment.get_array_of_samples())
        if segment.channels == 2:
            samples = samples.reshape((-1, 2))
        
        return to_engine_format(samples, sr, target_rate), target_rate
    except PermissionError:
        raise
    except Exception as e:
        errors.append(f"pydub: {e}")
    
    # Scipy wavfile as last resort
    try:
        from scipy.io import wavfile
        sr, audio = wavfile.read(str(path))
        return to_engine_format(audio, sr, target_rate), target_rate
    except PermissionError:
        raise
    except Exception as e:
        errors.append(f"scipy: {e}")
    
    # All methods failed
    raise RuntimeError(
        f"Could not load audio file '{path.name}'.\n"
        f"Tried: {', '.join(errors)}\n"
        f"Make sure the file is a supported audio format (wav, mp3, flac, ogg, m4a, aiff)"
    )


# ============================================================================
# STREAMING BACKENDS
# ============================================================================

def stream_soundcloud(
    url: str,
    target_rate: int = 48000,
    cache_dir: Optional[Path] = None,
) -> StreamBuffer:
    """Stream audio from SoundCloud.
    
    Uses yt-dlp for extraction.
    
    Parameters
    ----------
    url : str
        SoundCloud track URL
    target_rate : int
        Target sample rate
    cache_dir : Path, optional
        Cache directory for downloaded files
    
    Returns
    -------
    StreamBuffer
        Buffered audio in engine format
    """
    buffer = StreamBuffer(audio=np.array([]), sample_rate=target_rate)
    
    try:
        import yt_dlp
    except ImportError:
        buffer.error = "yt-dlp required: pip install yt-dlp"
        return buffer
    
    # Setup cache
    if cache_dir is None:
        cache_dir = Path(tempfile.gettempdir()) / 'mdma_stream_cache'
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate cache key
    url_hash = hashlib.md5(url.encode()).hexdigest()[:12]
    cache_path = cache_dir / f"sc_{url_hash}.audio"
    
    # Check cache
    if cache_path.exists():
        try:
            audio, sr = load_audio_file(cache_path, target_rate)
            buffer.audio = audio
            buffer.ready = True
            return buffer
        except Exception:
            try:
                cache_path.unlink()  # Remove corrupted cache
            except Exception:
                pass
    
    # Download
    temp_path = cache_dir / f"sc_{url_hash}_temp"
    
    # CRITICAL: Disable yt-dlp's cache to avoid pickle errors
    # The "unknown object type P for object float" error comes from pickle protocol issues
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': str(temp_path) + '.%(ext)s',
        'quiet': True,
        'no_warnings': True,
        'cachedir': False,  # DISABLE CACHE - fixes pickle errors
        'no_cache_dir': True,  # Belt and suspenders
    }
    
    downloaded_file = None
    
    # First attempt: download without FFmpeg postprocessing (more reliable)
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            
            # Get metadata - ensure all values are proper types
            title = info.get('title') or 'Unknown'
            artist = info.get('uploader') or info.get('channel') or ''
            duration = info.get('duration')
            thumbnail = info.get('thumbnail') or ''
            
            # Safely convert duration to float
            try:
                duration = float(duration) if duration is not None else 0.0
            except (TypeError, ValueError):
                duration = 0.0
            
            track_info = TrackInfo(
                id=url_hash,
                title=str(title),
                artist=str(artist),
                duration=duration,
                source=TrackSource.SOUNDCLOUD,
                url=url,
                thumbnail_url=str(thumbnail),
            )
            buffer.track_info = track_info
            
            # Get the actual extension from info
            actual_ext = info.get('ext', 'mp3')
            downloaded_file = Path(str(temp_path) + '.' + actual_ext)
        
        # Find downloaded file - check multiple possible extensions
        if not downloaded_file or not downloaded_file.exists():
            for ext in ['.mp3', '.m4a', '.wav', '.webm', '.opus', '.ogg', '.aac']:
                check_path = Path(str(temp_path) + ext)
                if check_path.exists():
                    downloaded_file = check_path
                    break
                    
    except Exception as e:
        buffer.error = f"SoundCloud stream failed: {e}"
        return buffer
    
    # Load the downloaded file
    if downloaded_file and downloaded_file.exists():
        try:
            audio, sr = load_audio_file(downloaded_file, target_rate)
            buffer.audio = audio
            buffer.ready = True
            
            # Save to cache
            try:
                import shutil
                shutil.copy2(downloaded_file, cache_path)
                downloaded_file.unlink()
            except Exception:
                pass  # Cache save failed, but audio is loaded
            
            # Auto-register to song registry
            try:
                reg_id = auto_register_stream(buffer, cache_path, "soundcloud")
                buffer.registry_id = reg_id
            except Exception:
                pass  # Registration failed, but streaming works
                
        except Exception as e:
            buffer.error = f"Failed to load downloaded audio: {e}"
    else:
        buffer.error = "Download completed but file not found"
    
    return buffer


def stream_youtube(
    url: str,
    target_rate: int = 48000,
    cache_dir: Optional[Path] = None,
) -> StreamBuffer:
    """Stream audio from YouTube.
    
    Uses yt-dlp for extraction.
    """
    buffer = StreamBuffer(audio=np.array([]), sample_rate=target_rate)
    
    try:
        import yt_dlp
    except ImportError:
        buffer.error = "yt-dlp required: pip install yt-dlp"
        return buffer
    
    if cache_dir is None:
        cache_dir = Path(tempfile.gettempdir()) / 'mdma_stream_cache'
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    url_hash = hashlib.md5(url.encode()).hexdigest()[:12]
    cache_path = cache_dir / f"yt_{url_hash}.audio"
    
    if cache_path.exists():
        try:
            audio, sr = load_audio_file(cache_path, target_rate)
            buffer.audio = audio
            buffer.ready = True
            return buffer
        except Exception:
            try:
                cache_path.unlink()
            except Exception:
                pass
    
    temp_path = cache_dir / f"yt_{url_hash}_temp"
    
    # CRITICAL: Disable yt-dlp's cache to avoid pickle errors
    # The "unknown object type P for object float" error comes from pickle protocol issues
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': str(temp_path) + '.%(ext)s',
        'quiet': True,
        'no_warnings': True,
        'cachedir': False,  # DISABLE CACHE - fixes pickle errors
        'no_cache_dir': True,  # Belt and suspenders
    }
    
    downloaded_file = None
    
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            
            # Get metadata - ensure all values are proper types
            title = info.get('title') or 'Unknown'
            artist = info.get('uploader') or info.get('channel') or ''
            duration = info.get('duration')
            thumbnail = info.get('thumbnail') or ''
            
            # Safely convert duration to float
            try:
                duration = float(duration) if duration is not None else 0.0
            except (TypeError, ValueError):
                duration = 0.0
            
            track_info = TrackInfo(
                id=url_hash,
                title=str(title),
                artist=str(artist),
                duration=duration,
                source=TrackSource.YOUTUBE,
                url=url,
                thumbnail_url=str(thumbnail),
            )
            buffer.track_info = track_info
            
            # Get the actual extension from info
            actual_ext = info.get('ext', 'webm')
            downloaded_file = Path(str(temp_path) + '.' + actual_ext)
        
        # Find downloaded file if not at expected path
        if not downloaded_file or not downloaded_file.exists():
            for ext in ['.webm', '.m4a', '.mp3', '.opus', '.ogg', '.aac', '.wav']:
                check_path = Path(str(temp_path) + ext)
                if check_path.exists():
                    downloaded_file = check_path
                    break
                    
    except Exception as e:
        buffer.error = f"YouTube stream failed: {e}"
        return buffer
    
    # Load the downloaded file
    if downloaded_file and downloaded_file.exists():
        try:
            audio, sr = load_audio_file(downloaded_file, target_rate)
            buffer.audio = audio
            buffer.ready = True
            
            # Save to cache
            try:
                import shutil
                shutil.copy2(downloaded_file, cache_path)
                downloaded_file.unlink()
            except Exception:
                pass
            
            # Auto-register to song registry
            try:
                reg_id = auto_register_stream(buffer, cache_path, "youtube")
                buffer.registry_id = reg_id
            except Exception:
                pass  # Registration failed, but streaming works
                
        except Exception as e:
            buffer.error = f"Failed to load downloaded audio: {e}"
    else:
        buffer.error = "Download completed but file not found"
    
    return buffer


def stream_url(
    url: str,
    target_rate: int = 48000,
) -> StreamBuffer:
    """Stream audio from a direct URL.
    
    Auto-detects source (SoundCloud, YouTube, direct file).
    """
    parsed = urlparse(url)
    domain = parsed.netloc.lower()
    
    if 'soundcloud.com' in domain:
        return stream_soundcloud(url, target_rate)
    elif 'youtube.com' in domain or 'youtu.be' in domain:
        return stream_youtube(url, target_rate)
    else:
        # Try as direct audio URL
        buffer = StreamBuffer(audio=np.array([]), sample_rate=target_rate)
        
        try:
            import urllib.request
            
            # Download to temp file
            with tempfile.NamedTemporaryFile(suffix='.audio', delete=False) as f:
                temp_path = f.name
                urllib.request.urlretrieve(url, temp_path)
            
            audio, sr = load_audio_file(temp_path, target_rate)
            buffer.audio = audio
            buffer.ready = True
            
            # Auto-register to song registry
            try:
                reg_id = auto_register_stream(buffer, Path(temp_path), "url")
                buffer.registry_id = reg_id
            except Exception:
                pass  # Registration failed, but streaming works
            
            os.unlink(temp_path)
            
        except Exception as e:
            buffer.error = f"URL stream failed: {e}"
        
        return buffer


# ============================================================================
# LIBRARY MANAGEMENT
# ============================================================================

class StreamingLibrary:
    """Library for managing streamed and local audio."""
    
    def __init__(self, sample_rate: int = 48000):
        self.sample_rate = sample_rate
        self.tracks: Dict[str, TrackInfo] = {}
        self.playlists: Dict[str, List[str]] = {}
        self.cache_dir = Path.home() / 'Documents' / 'MDMA' / 'stream_cache'
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Async loading
        self._loading: Dict[str, threading.Thread] = {}
        self._buffers: Dict[str, StreamBuffer] = {}
    
    def add_url(self, url: str, playlist: str = None) -> str:
        """Add a track from URL.
        
        Returns track ID.
        """
        url_hash = hashlib.md5(url.encode()).hexdigest()[:12]
        
        # Create initial track info
        track = TrackInfo(
            id=url_hash,
            title=f"Loading... ({url_hash})",
            source=self._detect_source(url),
            url=url,
        )
        self.tracks[url_hash] = track
        
        if playlist:
            if playlist not in self.playlists:
                self.playlists[playlist] = []
            self.playlists[playlist].append(url_hash)
        
        return url_hash
    
    def _detect_source(self, url: str) -> TrackSource:
        """Detect source type from URL."""
        domain = urlparse(url).netloc.lower()
        
        if 'soundcloud.com' in domain:
            return TrackSource.SOUNDCLOUD
        elif 'youtube.com' in domain or 'youtu.be' in domain:
            return TrackSource.YOUTUBE
        elif 'bandcamp.com' in domain:
            return TrackSource.BANDCAMP
        else:
            return TrackSource.URL
    
    def load_track(
        self,
        track_id: str,
        async_load: bool = True,
    ) -> Optional[StreamBuffer]:
        """Load a track's audio.
        
        Parameters
        ----------
        track_id : str
            Track ID
        async_load : bool
            If True, load in background thread
        
        Returns
        -------
        StreamBuffer or None
            Buffer if sync load, None if async
        """
        if track_id not in self.tracks:
            return None
        
        track = self.tracks[track_id]
        
        if async_load:
            # Start background load
            def _load():
                buffer = stream_url(track.url, self.sample_rate)
                if buffer.track_info:
                    # Update track info
                    track.title = buffer.track_info.title
                    track.artist = buffer.track_info.artist
                    track.duration = buffer.track_info.duration
                self._buffers[track_id] = buffer
            
            thread = threading.Thread(target=_load)
            self._loading[track_id] = thread
            thread.start()
            return None
        else:
            buffer = stream_url(track.url, self.sample_rate)
            if buffer.track_info:
                track.title = buffer.track_info.title
                track.artist = buffer.track_info.artist
                track.duration = buffer.track_info.duration
            self._buffers[track_id] = buffer
            return buffer
    
    def get_buffer(self, track_id: str) -> Optional[StreamBuffer]:
        """Get loaded buffer for a track."""
        return self._buffers.get(track_id)
    
    def is_loading(self, track_id: str) -> bool:
        """Check if a track is currently loading."""
        if track_id in self._loading:
            return self._loading[track_id].is_alive()
        return False
    
    def wait_for_load(self, track_id: str, timeout: float = 60.0) -> Optional[StreamBuffer]:
        """Wait for a track to finish loading."""
        if track_id in self._loading:
            self._loading[track_id].join(timeout)
        return self._buffers.get(track_id)
    
    def scan_directory(
        self,
        directory: Union[str, Path],
        recursive: bool = True,
    ) -> int:
        """Scan a directory for audio files.
        
        Returns count of tracks added.
        """
        directory = Path(directory)
        extensions = {'.wav', '.mp3', '.flac', '.ogg', '.m4a', '.aiff', '.aif'}
        
        count = 0
        pattern = '**/*' if recursive else '*'
        
        for path in directory.glob(pattern):
            if path.suffix.lower() in extensions:
                track_id = hashlib.md5(str(path).encode()).hexdigest()[:12]
                
                track = TrackInfo(
                    id=track_id,
                    title=path.stem,
                    source=TrackSource.LOCAL,
                    local_path=str(path),
                )
                self.tracks[track_id] = track
                count += 1
        
        return count
    
    def search(self, query: str) -> List[TrackInfo]:
        """Search library by title/artist."""
        query_lower = query.lower()
        results = []
        
        for track in self.tracks.values():
            if (query_lower in track.title.lower() or 
                query_lower in track.artist.lower() or
                any(query_lower in tag.lower() for tag in track.tags)):
                results.append(track)
        
        return results
    
    def create_playlist(self, name: str) -> bool:
        """Create a new playlist."""
        if name in self.playlists:
            return False
        self.playlists[name] = []
        return True
    
    def add_to_playlist(self, playlist: str, track_id: str) -> bool:
        """Add track to playlist."""
        if playlist not in self.playlists:
            return False
        if track_id not in self.tracks:
            return False
        
        self.playlists[playlist].append(track_id)
        return True
    
    def get_playlist(self, name: str) -> List[TrackInfo]:
        """Get tracks in a playlist."""
        if name not in self.playlists:
            return []
        
        return [self.tracks[tid] for tid in self.playlists[name] if tid in self.tracks]
    
    def list_tracks(self, limit: int = 50) -> List[TrackInfo]:
        """List all tracks."""
        return list(self.tracks.values())[:limit]
    
    def clear_cache(self) -> int:
        """Clear stream cache. Returns bytes freed."""
        total = 0
        for f in self.cache_dir.glob('*'):
            if f.is_file():
                total += f.stat().st_size
                f.unlink()
        return total


# ============================================================================
# SOUNDCLOUD API (Direct)
# ============================================================================

def soundcloud_search(
    query: str,
    limit: int = 10,
    client_id: str = None,
) -> List[Dict[str, Any]]:
    """Search SoundCloud for tracks.
    
    Note: Requires client_id or uses fallback scraping.
    
    Returns list of track info dicts.
    """
    try:
        import yt_dlp
        
        search_url = f"scsearch{limit}:{query}"
        
        ydl_opts = {
            'quiet': True,
            'no_warnings': True,
            'extract_flat': True,
        }
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            results = ydl.extract_info(search_url, download=False)
            
            if results and 'entries' in results:
                return [{
                    'title': e.get('title', ''),
                    'url': e.get('url', ''),
                    'duration': e.get('duration', 0),
                    'uploader': e.get('uploader', ''),
                    'id': e.get('id', ''),
                } for e in results['entries'] if e]
            
    except Exception as e:
        print(f"SoundCloud search error: {e}")
    
    return []


def youtube_search(
    query: str,
    limit: int = 10,
) -> List[Dict[str, Any]]:
    """Search YouTube for tracks/videos.
    
    Returns list of track info dicts.
    """
    try:
        import yt_dlp
        
        search_url = f"ytsearch{limit}:{query}"
        
        ydl_opts = {
            'quiet': True,
            'no_warnings': True,
            'extract_flat': True,
        }
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            results = ydl.extract_info(search_url, download=False)
            
            if results and 'entries' in results:
                return [{
                    'title': e.get('title', ''),
                    'url': e.get('url', e.get('webpage_url', '')),
                    'webpage_url': e.get('webpage_url', e.get('url', '')),
                    'duration': e.get('duration', 0),
                    'uploader': e.get('uploader', e.get('channel', '')),
                    'channel': e.get('channel', e.get('uploader', '')),
                    'id': e.get('id', ''),
                } for e in results['entries'] if e]
            
    except Exception as e:
        print(f"YouTube search error: {e}")
    
    return []


# ============================================================================
# GLOBAL LIBRARY
# ============================================================================

_library: Optional[StreamingLibrary] = None


def get_library(sample_rate: int = 48000) -> StreamingLibrary:
    """Get or create global streaming library."""
    global _library
    if _library is None:
        _library = StreamingLibrary(sample_rate)
    return _library
