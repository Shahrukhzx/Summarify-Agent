import subprocess
import whisper
import os

def extract_audio(video_path: str,audio_path:str = os.path.abspath("temp_audio.wav")) -> str :
    if os.path.exists(audio_path):
        os.remove(audio_path)   # if the audio file already exists, then remove it 

    command = [
            "ffmpeg",       # multi-media processing tool
            "-i",video_path,       #path to the input video file
            "-q:a","0",     # sets the audio quality to highest
            "-map","a",     # selects the audio track from the video
            audio_path,     # destination path for the output audio
            "-y"           # overwrites the existing file
        ]

    subprocess.run(command,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,check=True)
    return audio_path
    
def transcribe_audio(audio_path:str,model_size:str ="base") -> str:
    
    model = whisper.load_model(model_size)
    result = model.transcribe(audio_path)
    transcript = result["text"]
    return transcript