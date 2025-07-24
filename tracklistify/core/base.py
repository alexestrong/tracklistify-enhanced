# tracklistify/core/app.py

# Standard library imports
import asyncio
import concurrent.futures
import traceback
from datetime import datetime
from pathlib import Path
from typing import List

# Local/package imports
from tracklistify.config.factory import get_config
from tracklistify.core.track import Track
from tracklistify.core.types import AudioSegment
from tracklistify.downloaders import DownloaderFactory
from tracklistify.exporters import TracklistOutput
from tracklistify.providers.factory import create_provider_factory
from tracklistify.utils.logger import get_logger
from tracklistify.utils.validation import validate_input
from tracklistify.utils.identification import IdentificationManager


logger = get_logger(__name__)


class AsyncApp:
    """Main application logic container"""

    def __init__(self, config=None):
        # Always refresh config
        self.config = config or get_config(force_refresh=True)
        self.provider_factory = create_provider_factory()
        self.downloader_factory = DownloaderFactory()
        self.logger = get_logger(__name__)
        self.shutdown_event = asyncio.Event()
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)

        # Always recreate identification_manager with fresh config
        self.identification_manager = IdentificationManager(
            config=self.config, provider_factory=self.provider_factory
        )

    def shutdown(self) -> None:
        """Shutdown the application gracefully."""
        self.logger.info("Shutting down...")
        self.shutdown_event.set()
        self.executor.shutdown(wait=True)

    async def process_input(self, input_path: str):
        """Process input URL or file path."""
        import time
        
        # Start timing
        start_time = time.time()
        
        try:
            # Validate input (local file or URL)
            validated_input = validate_input(input_path)
            if validated_input is None:
                raise ValueError("Invalid input: must be a valid audio file path or URL")

            self.logger.info(f"Validated input: {validated_input}")

            # Check if input is a local file or URL
            local_path = None
            if Path(validated_input).exists():
                # It's a local file, use it directly
                local_path = validated_input
                self.original_title = Path(local_path).stem
                self.logger.info(f"Using local audio file: {local_path}")
            else:
                # It's a URL, use downloader
                downloader = self.downloader_factory.create_downloader(validated_input)
                if downloader is None:
                    raise ValueError("Failed to create downloader")

                self.logger.info("Downloading audio...")

                # Download audio file
                local_path = await downloader.download(validated_input)
                if local_path is None:
                    raise ValueError("local_path cannot be None")

                self.logger.info(f"Downloaded audio to: {local_path}")

                # Store original title for output
                self.logger.debug(f"Storing original title: {local_path}")
                self.original_title = getattr(downloader, "title", Path(local_path).stem)

            self.logger.info("Processing audio...")

            # Process the downloaded file
            audio_segments = self.split_audio(local_path)
            if not audio_segments:
                raise ValueError("No audio segments were created")

            self.logger.info(f"Created {len(audio_segments)} audio segments")

            # Identify tracks in audio segments
            self.logger.info("Identifying tracks...")
            tracks = await self.identification_manager.identify_tracks(audio_segments)
            if not tracks:
                raise ValueError("No tracks were identified in the audio file")

            self.logger.info(f"Identified {len(tracks)} tracks")
            self.logger.debug(f"Tracks: {tracks}")

            # Calculate processing time
            end_time = time.time()
            processing_time = end_time - start_time

            # Only save if we have identified tracks
            self.logger.info("Saving output...")
            if len(tracks) > 0:
                await self.save_output(tracks, self.config.output_format, processing_time)
            else:
                raise ValueError(
                    "No tracks were successfully identified with sufficient confidence"
                )

        except Exception as e:
            self.logger.error(f"Failed to process input: {e}")
            if self.config.debug:
                self.logger.error(traceback.format_exc())
            raise
        finally:
            # Always clean up temporary files
            await self.cleanup()

    def split_audio(self, file_path: str) -> List[AudioSegment]:
        """Split audio file into overlapping segments for analysis."""
        self.logger.info(f"Splitting audio file: {file_path}")
        self.logger.debug(
            f"Config values: segment_length={self.config.segment_length}, "
            f"overlap_duration={self.config.overlap_duration}"
        )

        import os
        import subprocess
        from concurrent.futures import ThreadPoolExecutor
        from pathlib import Path
        
        # Try to read audio file with mutagen first
        duration = None
        try:
            from mutagen import File
            audio = File(file_path)
            if audio is not None:
                duration = getattr(audio.info, 'length', None)
                if duration and duration > 0:
                    self.logger.debug(f"Audio file duration from mutagen: {duration} seconds")
                else:
                    self.logger.warning(f"Mutagen returned invalid duration: {duration}")
                    duration = None

        except ImportError:
            self.logger.warning("Mutagen library not available, will try ffprobe")
        except Exception as e:
            self.logger.warning(f"Error reading audio file with mutagen: {e}, will try ffprobe")

        # Fallback to ffprobe if mutagen failed
        if duration is None or duration <= 0:
            self.logger.info("Trying ffprobe to get audio duration...")
            try:
                result = subprocess.run([
                    "ffprobe", "-v", "quiet", "-show_entries", "format=duration", 
                    "-of", "csv=p=0", file_path
                ], capture_output=True, text=True, check=True)
                
                duration_str = result.stdout.strip()
                if duration_str:
                    duration = float(duration_str)
                    self.logger.debug(f"Audio file duration from ffprobe: {duration} seconds")
                else:
                    self.logger.error("ffprobe returned empty duration")
                    return []
                    
            except subprocess.CalledProcessError as e:
                self.logger.error(f"ffprobe failed: {e.stderr}")
                return []
            except ValueError as e:
                self.logger.error(f"Could not parse duration from ffprobe: {e}")
                return []
            except FileNotFoundError:
                self.logger.error("ffprobe not found - please install ffmpeg")
                return []

        if duration is None or duration <= 0:
            self.logger.error(f"Could not determine valid audio duration: {duration}")
            return []

        # Get configuration for segmentation from instance
        segment_duration = self.config.segment_length
        overlap_duration = self.config.overlap_duration
        step = segment_duration - overlap_duration

        # Create temp directory for segments
        temp_dir = Path(self.config.temp_dir)
        temp_dir.mkdir(parents=True, exist_ok=True)

        # Optimize ffmpeg settings for faster processing
        base_cmd = [
            "ffmpeg",
            "-hide_banner",
            "-nostdin",  # Disable stdin processing (can break the shell)
            "-loglevel",
            "error",
            "-i",
            file_path,
            "-vn",  # Skip video processing
            "-ar",
            "44100",  # Standard sample rate
            "-ac",
            "2",  # Stereo
            "-c:a",
            "libmp3lame",  # Use MP3 encoder
            "-q:a",
            "5",  # Variable bitrate quality (0-9, lower is better)
            "-map",
            "0:a",  # Only process audio stream
            "-threads",
            str(os.cpu_count()),  # Use all CPU cores
        ]

        # Generate segment parameters
        segment_params = []
        current_time = 0

        while current_time < duration:
            segment_length = min(segment_duration, duration - current_time)
            segment_file = (
                temp_dir / f"segment_{current_time:.0f}_{segment_length:.0f}.mp3"
            )

            # Add small padding to improve recognition
            start_time = max(0, current_time - 0.5)
            end_time = min(duration, current_time + segment_length + 0.5)
            actual_length = end_time - start_time

            segment_params.append(
                {
                    "start_time": current_time,
                    "length": segment_length,
                    "file": segment_file,
                    "cmd": base_cmd
                    + [
                        "-ss",
                        str(start_time),
                        "-t",
                        str(actual_length),
                        "-y",
                        str(segment_file),
                    ],
                }
            )

            current_time += step

        def create_segment(params):
            """Create a single audio segment using ffmpeg."""
            try:
                if params["file"].exists():
                    # Skip if file already exists and has content
                    if params["file"].stat().st_size > 1000:
                        return AudioSegment(
                            file_path=str(params["file"]),
                            start_time=int(params["start_time"]),
                            duration=int(params["length"]),
                        )

                result = subprocess.run(
                    params["cmd"], capture_output=True, text=True, check=True
                )
                self.logger.debug(f"FFmpeg output: {result.stdout}")

                if params["file"].exists() and params["file"].stat().st_size > 1000:
                    return AudioSegment(
                        file_path=str(params["file"]),
                        start_time=int(params["start_time"]),
                        duration=int(params["length"]),
                    )
                else:
                    self.logger.error(
                        f"Failed to create segment at {params['start_time']}s: "
                        f"Output file is missing or too small"
                    )
                    return None

            except subprocess.CalledProcessError as e:
                self.logger.error(
                    f"Failed to create segment at {params['start_time']}s: {e.stderr}"
                )
                return None

            except Exception as e:
                self.logger.error(
                    f"Error creating segment at {params['start_time']}s: {e}"
                )
                return None

        # Process segments in parallel using ThreadPoolExecutor
        segments = []
        try:
            # Check if we have any segments to process
            if not segment_params:
                self.logger.error("No segments to process - segment_params is empty")
                self.logger.debug(f"Audio duration: {duration}, segment_duration: {segment_duration}, overlap_duration: {overlap_duration}")
                return []

            # Ensure os.cpu_count() does not return None
            cpu_count = os.cpu_count()
            if cpu_count is None:
                self.logger.warning("os.cpu_count() returned None, defaulting to 2 workers")
                cpu_count = 2

            # Use more workers for better parallelization, ensure max_workers > 0
            max_workers = min(max(1, cpu_count * 2), len(segment_params))
            self.logger.debug(f"Processing {len(segment_params)} segments with {max_workers} workers")

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Submit all tasks and gather results
                future_segments = list(executor.map(create_segment, segment_params))

                # Filter out failed segments
                segments = [seg for seg in future_segments if seg is not None]

            self.logger.info(f"Split audio into {len(segments)} segments (created {len(segment_params)} total)")
            return segments

        except Exception as e:
            self.logger.error(f"Failed to process segments: {e}")
            if self.config.debug:
                self.logger.error(traceback.format_exc())
            return []

    async def save_output(self, tracks: List["Track"], format: str, processing_time: float = None):
        """Save identified tracks to output files.

        Args:
            tracks: List of identified tracks
            format: Output format (json, markdown, m3u)
            processing_time: Time taken to process the audio in seconds

        Raises:
            ValueError: If tracks list is empty
        """
        if len(tracks) == 0:
            self.logger.error("Cannot save output: No tracks provided")
            return

        # Get title from the original title or use a default
        title = getattr(self, "original_title", None)
        if not title:
            # Try to construct a title from the first track
            if tracks and tracks[0].artist and tracks[0].song_name:
                title = f"{tracks[0].artist} - {tracks[0].song_name}"
            else:
                title = "Identified Mix"

        # Prepare mix info using the downloaded title
        mix_info = {
            "title": title,
            "date": datetime.now().strftime("%Y-%m-%d"),
            "track_count": len(tracks),
        }

        try:
            # Create output handler with processing time
            output = TracklistOutput(mix_info=mix_info, tracks=tracks, processing_time=processing_time)

            # Save in specified format
            if format == "all":
                saved_files = output.save_all()
                if saved_files:
                    for file in saved_files:
                        self.logger.debug(f"Saved tracklist to: {file}")
                else:
                    self.logger.error("Failed to save tracklist in any format")
            else:
                if saved_file := output.save(format):
                    self.logger.info(f"Saved {format} tracklist to: {saved_file}")
                else:
                    self.logger.error(f"Failed to save tracklist in format: {format}")

        except Exception as e:
            self.logger.error(f"Error saving tracklist: {e}")
            if self.config.debug:
                self.logger.error(traceback.format_exc())

    async def cleanup(self):
        """Cleanup resources"""
        try:
            # Clean up temp directory
            temp_dir = Path(self.config.temp_dir)
            if temp_dir.exists():
                # First try to remove all files
                for file in temp_dir.glob("*"):
                    try:
                        if file.is_file():
                            file.unlink()
                            self.logger.debug(f"Removed temporary file: {file}")
                        elif file.is_dir():
                            import shutil

                            shutil.rmtree(file)
                            self.logger.debug(f"Removed temporary directory: {file}")
                    except Exception as e:
                        self.logger.warning(f"Failed to remove {file}: {e}")

                # Try to remove the directory itself
                try:
                    # Use rmtree instead of rmdir to handle any remaining files
                    import shutil

                    shutil.rmtree(temp_dir)
                    self.logger.debug("Removed temporary directory")
                except Exception as e:
                    self.logger.debug(f"Could not remove temp directory: {e}")
                    # If rmtree fails, try to at least remove empty directory
                    try:
                        temp_dir.rmdir()
                    except Exception:
                        pass
        except Exception as e:
            self.logger.warning(f"Error during cleanup: {e}")

        # Clean up other resources
        try:
            if hasattr(self.identification_manager, "close"):
                await self.identification_manager.close()
        except Exception as e:
            self.logger.warning(f"Error cleaning up identification manager: {e}")

    async def close(self):
        """Cleanup resources."""
        await self.cleanup()


class ApplicationError(Exception):
    """Base application error."""

    pass
