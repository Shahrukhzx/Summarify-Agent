import os
from transcriber import extract_audio,transcribe_audio
from summarizer import summarize_text
from utils import chunked_summarize

def video_to_summary(video_path:str,model_size:str = "base",summarizer_model_name:str ="facebook/bart-large-cnn",use_chunking:bool= False) ->str :
    # extract the audio from the video 
    audio_path = os.path.abspath("temp_audio.wav")
    extract_audio(video_path,audio_path)

    # transcribe audio
    transcript = transcribe_audio(audio_path,model_size=model_size)

    # summarizing transcript
    if use_chunking:
        # summarize in multiple chunks and then do a final summary
        final_summary = chunked_summarize(text=transcript,summarize_func=lambda txt: summarize_text(txt,model_name=summarizer_model_name),max_chunk_size=2000)
    else:
        # summarize in a single pass 
        final_summary = summarize_text(transcript,model_name=summarizer_model_name)
    if os.path.exists(audio_path):
        os.remove(audio_path)
    return final_summary

if __name__ == "__main__":
    # Example usage
    video_file = "example_video.mp4"
    summary_output = video_to_summary(video_file,model_size="base",summarizer_model_name="facebook/bart-large-cnn",use_chunking=True)
    print("--------------FINAL SUMMARY-----------------------")
    print(summary_output)