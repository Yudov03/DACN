"""
ASR Module - Automatic Speech Recognition với Whisper
Chuyển đổi audio thành văn bản kèm timestamp
"""

import whisper
import torch
import json
from pathlib import Path
from typing import Dict, List, Optional, Union
from datetime import datetime
import warnings

warnings.filterwarnings("ignore")


class WhisperASR:
    """
    Lớp xử lý chuyển đổi âm thanh sang văn bản sử dụng OpenAI Whisper
    """

    def __init__(
        self,
        model_name: str = "base",
        device: Optional[str] = None,
        language: str = "vi"
    ):
        """
        Khởi tạo Whisper ASR

        Args:
            model_name: Tên model Whisper (tiny, base, small, medium, large)
            device: Device để chạy model (cuda/cpu), None để tự động detect
            language: Ngôn ngữ của audio (vi, en, etc.)
        """
        self.model_name = model_name
        self.language = language

        # Tự động detect device nếu không được chỉ định
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        print(f"Đang tải Whisper model '{model_name}' trên {self.device}...")
        self.model = whisper.load_model(model_name, device=self.device)
        print(f"Đã tải xong model Whisper '{model_name}'")

    def transcribe_audio(
        self,
        audio_path: Union[str, Path],
        return_timestamps: bool = True,
        verbose: bool = True
    ) -> Dict:
        """
        Chuyển đổi file audio thành văn bản kèm timestamp

        Args:
            audio_path: Đường dẫn đến file audio
            return_timestamps: Có trả về timestamps không
            verbose: Hiển thị progress bar

        Returns:
            Dict chứa transcript, segments với timestamps và metadata
        """
        audio_path = Path(audio_path)

        if not audio_path.exists():
            raise FileNotFoundError(f"Không tìm thấy file audio: {audio_path}")

        print(f"Đang transcribe file: {audio_path.name}...")

        # Thực hiện transcription
        result = self.model.transcribe(
            str(audio_path),
            language=self.language,
            verbose=verbose,
            word_timestamps=return_timestamps
        )

        # Xử lý kết quả
        transcript_data = {
            "audio_file": str(audio_path),
            "audio_filename": audio_path.name,
            "model": self.model_name,
            "language": result.get("language", self.language),
            "full_text": result["text"],
            "segments": self._process_segments(result.get("segments", [])),
            "transcribed_at": datetime.now().isoformat(),
            "duration": self._get_audio_duration(result)
        }

        if verbose:
            print(f"Hoàn thành! Tổng số segments: {len(transcript_data['segments'])}")

        return transcript_data

    def _process_segments(self, segments: List[Dict]) -> List[Dict]:
        """
        Xử lý các segments từ Whisper output

        Args:
            segments: Raw segments từ Whisper

        Returns:
            List các segments đã được xử lý
        """
        processed_segments = []

        for idx, segment in enumerate(segments):
            processed_segment = {
                "id": idx,
                "start": segment.get("start", 0.0),
                "end": segment.get("end", 0.0),
                "text": segment.get("text", "").strip(),
                "duration": segment.get("end", 0.0) - segment.get("start", 0.0)
            }
            processed_segments.append(processed_segment)

        return processed_segments

    def _get_audio_duration(self, result: Dict) -> float:
        """
        Lấy tổng thời lượng audio từ kết quả transcription

        Args:
            result: Kết quả từ Whisper

        Returns:
            Thời lượng audio (giây)
        """
        segments = result.get("segments", [])
        if segments:
            return segments[-1].get("end", 0.0)
        return 0.0

    def save_transcript(
        self,
        transcript_data: Dict,
        output_path: Union[str, Path],
        format: str = "json"
    ) -> Path:
        """
        Lưu transcript ra file

        Args:
            transcript_data: Dữ liệu transcript
            output_path: Đường dẫn file output
            format: Định dạng output (json, txt)

        Returns:
            Path của file đã lưu
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if format == "json":
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(transcript_data, f, ensure_ascii=False, indent=2)

        elif format == "txt":
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(f"Audio: {transcript_data['audio_filename']}\n")
                f.write(f"Duration: {transcript_data['duration']:.2f}s\n")
                f.write(f"Transcribed at: {transcript_data['transcribed_at']}\n")
                f.write("=" * 80 + "\n\n")
                f.write(transcript_data['full_text'] + "\n\n")
                f.write("=" * 80 + "\n")
                f.write("SEGMENTS WITH TIMESTAMPS:\n")
                f.write("=" * 80 + "\n\n")

                for segment in transcript_data['segments']:
                    start_time = self._format_timestamp(segment['start'])
                    end_time = self._format_timestamp(segment['end'])
                    f.write(f"[{start_time} --> {end_time}]\n")
                    f.write(f"{segment['text']}\n\n")

        print(f"Đã lưu transcript tại: {output_path}")
        return output_path

    def _format_timestamp(self, seconds: float) -> str:
        """
        Chuyển đổi giây thành format timestamp HH:MM:SS

        Args:
            seconds: Số giây

        Returns:
            Timestamp dạng string
        """
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = seconds % 60
        return f"{hours:02d}:{minutes:02d}:{secs:05.2f}"

    def transcribe_batch(
        self,
        audio_files: List[Union[str, Path]],
        output_dir: Union[str, Path],
        save_format: str = "json"
    ) -> List[Dict]:
        """
        Transcribe nhiều file audio cùng lúc

        Args:
            audio_files: List các đường dẫn file audio
            output_dir: Thư mục lưu output
            save_format: Định dạng output (json, txt)

        Returns:
            List các transcript data
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        all_transcripts = []

        for i, audio_file in enumerate(audio_files, 1):
            print(f"\n[{i}/{len(audio_files)}] Processing: {Path(audio_file).name}")

            try:
                transcript_data = self.transcribe_audio(audio_file)
                all_transcripts.append(transcript_data)

                # Lưu transcript
                output_filename = Path(audio_file).stem + f"_transcript.{save_format}"
                output_path = output_dir / output_filename
                self.save_transcript(transcript_data, output_path, format=save_format)

            except Exception as e:
                print(f"Lỗi khi xử lý {audio_file}: {str(e)}")
                continue

        print(f"\n✓ Hoàn thành! Đã transcribe {len(all_transcripts)}/{len(audio_files)} files")
        return all_transcripts


# Test function
if __name__ == "__main__":
    # Example usage
    asr = WhisperASR(model_name="base", language="vi")

    # Test với 1 file
    # transcript = asr.transcribe_audio("path/to/audio.mp3")
    # asr.save_transcript(transcript, "output/transcript.json")

    print("ASR Module initialized successfully!")
