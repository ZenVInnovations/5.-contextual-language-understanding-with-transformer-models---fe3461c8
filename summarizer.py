from transformers import pipeline

def summarize_text(file_path):
    with open(file_path, 'r') as f:
        text = f.read()

    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    summary = summarizer(text, max_length=130, min_length=30, do_sample=False)
    return summary[0]['summary_text']
