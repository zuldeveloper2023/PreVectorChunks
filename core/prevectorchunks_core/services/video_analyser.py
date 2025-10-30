import os
import subprocess

import imageio_ffmpeg as ffmpeg
import cv2
import base64
import tempfile
from openai import OpenAI
from dotenv import load_dotenv
from PIL import Image
from io import BytesIO

load_dotenv(override=True)# must come firs
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
ffmpeg_path = ffmpeg.get_ffmpeg_exe()
class VideoAnalyzer:
    def __init__(self, video_path: str, frame_interval: int = 5):
        self.video_path = video_path
        self.frame_interval = frame_interval
        self.temp_audio_path = None
        self.frames = []

    # ---------------------------------------------------
    # 1️⃣ Extract audio using FFmpeg
    # ---------------------------------------------------
    def extract_audio(self):
        """Extracts audio from the video file and saves it as temp_audio.wav"""
        temp_dir = tempfile.gettempdir()
        self.temp_audio_path = os.path.join(temp_dir, "temp_audio.wav")

        ffmpeg_command = [
            ffmpeg_path,
            "-i", self.video_path,
            "-vn",  # No video
            "-ac", "1",  # Mono audio
            "-ar", "16000",  # 16 kHz sampling rate
            self.temp_audio_path
        ]
        # Run the ffmpeg command and check for errors
        result = subprocess.run(ffmpeg_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if result.returncode != 0:
            raise RuntimeError(f"FFmpeg failed: {result.stderr.decode('utf-8')}")

        print(f"🎧 Audio extracted to {self.temp_audio_path}")

        return self.temp_audio_path

    # ---------------------------------------------------
    # 2️⃣ Transcribe audio using OpenAI
    # ---------------------------------------------------
    def transcribe_audio(self):
        """Transcribe extracted audio using GPT-4o-transcribe"""
        if not self.temp_audio_path:
            raise ValueError("Audio not extracted. Call extract_audio() first.")



        audio_file = open(self.temp_audio_path, "rb")
        transcript = client.audio.transcriptions.create(
            model="gpt-4o-transcribe",
            file=audio_file,
            response_format="text",
            prompt="The following conversation is a lecture about the recent developments around OpenAI, GPT-4.5 and the future of AI."
        )

        print("✅ Audio transcription complete.")
        return transcript

    # ---------------------------------------------------
    # 3️⃣ Extract frames using OpenCV
    # ---------------------------------------------------
    def extract_frames(self):
        """Extract frames from the video every N seconds"""
        vidcap = cv2.VideoCapture(self.video_path)
        fps = vidcap.get(cv2.CAP_PROP_FPS)
        frame_gap = int(fps * self.frame_interval)
        success, image = vidcap.read()
        count = 0

        while success:
            if count % frame_gap == 0:
                img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(img_rgb)
                self.frames.append(pil_img)
            success, image = vidcap.read()
            count += 1

        vidcap.release()
        print(f"🖼️ Extracted {len(self.frames)} frames from video.")
        return self.frames

    # ---------------------------------------------------
    # 4️⃣ Analyze frames using GPT-4o Vision
    # ---------------------------------------------------
    def analyze_frames(self, max_frames: int = 5):
        """Send selected frames to GPT-4o for visual understanding"""
        selected_frames = self.frames[:max_frames]
        frame_contents = []

        for i, frame in enumerate(selected_frames):
            buffer = BytesIO()
            frame.save(buffer, format="JPEG")
            img_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
            frame_contents.append({
                "type": "input_image",
                "image_data": img_b64
            })

        print(f"🔍 Sending {len(selected_frames)} frames to GPT-4o for analysis...")

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "These are frames from a video. Describe what’s happening, list key actions, objects, and changes over time."},
                        *frame_contents
                    ]
                }
            ]
        )

        frame_analysis = response.choices[0].message.content
        print("✅ Frame analysis complete.")
        return frame_analysis

    # ---------------------------------------------------
    # 5️⃣ Combine transcript + visual analysis
    # ---------------------------------------------------
    def summarize_video(self, transcript, frame_analysis):
        """Combine the transcript and visual analysis for a full video summary"""
        print("🧠 Generating final summary from transcript + visuals...")

        summary_prompt = f"""
        Based on the following transcript and frame descriptions,
        summarise the video into structured markdown with sections:
        - **Overall Summary**
        - **Key Scenes**
        - **Main Actions**
        - **Dialogue / Transcript Highlights**
        - **Visual Elements (objects, motion, environment)**

        ### Transcript:
        {transcript}

        ### Frame Analysis:
        {frame_analysis}
        """

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": summary_prompt}]
        )

        final_summary = response.choices[0].message.content
        print("✅ Final video summary generated.")
        return final_summary



# ---------------------------------------------------
# 🧪 Example usage
# ---------------------------------------------------
if __name__ == "__main__":
    video_path = "C:\\test-sandbox\\be\\PreVectorDeps\\PreVectorChunks\\core\\prevectorchunks_core\\services\\13377816.mp4"  # replace with your video
    analyzer = VideoAnalyzer(video_path, frame_interval=5)

    audio_path = analyzer.extract_audio()
    transcript = analyzer.transcribe_audio()
    #frames = analyzer.extract_frames()
    #frame_analysis = analyzer.analyze_frames()
    #summary = analyzer.summarize_video(transcript, frame_analysis)

    print("\n\n===== FINAL VIDEO SUMMARY =====\n")
    print(transcript)
