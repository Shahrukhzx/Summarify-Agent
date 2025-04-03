from transformers import pipeline


def summarize_text(text:str,model_name:str ="facebook/bart-large-cnn",max_length:int=300,min_length:int = 75) -> str:
    summarizer = pipeline("summarization", model=model_name)
    summary =summarizer(
        text,
        max_length=max_length,
        min_length=min_length,
        do_sample = False
    )
    return summary[0]["summary_text"]