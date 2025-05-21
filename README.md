# Contextual Language Understanding with Transformer Models

This project demonstrates the use of transformer-based models like BERT and GPT for natural language understanding tasks including:
- Question Answering
- Text Summarization
- Document Comprehension

## Setup

Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### Question Answering

```bash
python main.py qa --context_file data/sample_qa.json
```

### Text Summarization

```bash
python main.py summarize --file data/sample_summary.txt
```

### Document Comprehension

```bash
python main.py comprehend --file data/sample_document.txt
```
