from datasets import load_dataset
from transformers import BertTokenizer
from transformers import BertForSequenceClassification, Trainer, TrainingArguments
import torch
from sklearn.metrics import accuracy_score
import os
import transformers
import datasets
import numpy
import packaging
from transformers import pipeline
import gradio as gr
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    # Load and prepare dataset
    logger.info("Loading emotion dataset...")
    dataset = load_dataset("dair-ai/emotion")
    dataset = dataset.rename_column("label", "labels")

    # Initialize tokenizer and model
    logger.info("Initializing BERT tokenizer and model...")
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=6)

    def tokenize(example):
        return tokenizer(example["text"], truncation=True, padding="max_length", max_length=128)

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = torch.argmax(torch.tensor(logits), dim=1)
        return {"accuracy": accuracy_score(labels, preds)}

    # Tokenize dataset
    logger.info("Tokenizing dataset...")
    tokenized = dataset.map(tokenize, batched=True)
    tokenized.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

    # Training configuration
    training_args = TrainingArguments(
        output_dir="./results",
        per_device_train_batch_size=16,
        num_train_epochs=1,
        learning_rate=2e-5,
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=100,
        no_cuda=not torch.cuda.is_available(),  # Use CPU if CUDA is not available
        report_to="none",  # Disable reporting to avoid unnecessary dependencies
        save_strategy="no",  # Don't save checkpoints
        eval_strategy="no",  # Disable evaluation during training
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        compute_metrics=compute_metrics,
    )

    # Train the model
    logger.info("Starting model training...")
    trainer.train()
    logger.info("Training completed!")

except Exception as e:
    logger.error(f"Error during model setup and training: {str(e)}")
    raise

# Print version information
logger.info("Environment information:")
print("Transformers:", transformers.__version__)
print("Datasets:", datasets.__version__)
print("NumPy:", numpy.__version__)
print("Packaging:", packaging.__version__)

# Initialize pipelines
logger.info("Initializing NLP pipelines...")
try:
    sentiment_pipeline = pipeline("sentiment-analysis")
    ner_pipeline = pipeline("ner", grouped_entities=True)
    summarizer = pipeline("summarization")
except Exception as e:
    logger.error(f"Error initializing pipelines: {str(e)}")
    raise

def analyze_text(text):
    try:
        # Analyze sentiment
        sentiment = sentiment_pipeline(text)

        # Analyze named entities
        ner_results = ner_pipeline(text)
        named_entities = [
            {
                "entity": item["entity_group"],
                "word": item["word"],
                "score": round(item["score"], 2)
            }
            for item in ner_results
        ]

        return {
            "Sentiment": sentiment[0],
            "Named Entities": named_entities
        }
    except Exception as e:
        logger.error(f"Error in analyze_text: {str(e)}")
        return {
            "Sentiment": {"label": "ERROR", "score": 0.0},
            "Named Entities": []
        }

def dashboard(text):
    try:
        result = analyze_text(text)
        sentiment = f"{result['Sentiment']['label']} (Score: {round(result['Sentiment']['score'], 2)})"
        entities = "\n".join([
            f"{item['word']} → {item['entity']} (Score: {item['score']})"
            for item in result['Named Entities']
        ])
        return sentiment, entities
    except Exception as e:
        logger.error(f"Error in dashboard: {str(e)}")
        return "Error processing text", "An error occurred during analysis"

# Create Gradio interface
iface = gr.Interface(
    fn=dashboard,
    inputs=gr.Textbox(lines=4, placeholder="Enter text here..."),
    outputs=[
        gr.Textbox(label="Sentiment Analysis"),
        gr.Textbox(label="Named Entity Recognition")
    ],
    title="Contextual Language Understanding with Transformers",
    description="Enter any English text and see sentiment and named entities identified by BERT."
)

if __name__ == "__main__":
    try:
        logger.info("Launching Gradio interface...")
        iface.launch()
    except Exception as e:
        logger.error(f"Error launching interface: {str(e)}")
        raise



