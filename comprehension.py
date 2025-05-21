from transformers import pipeline

def comprehend_document(file_path):
    with open(file_path, 'r') as f:
        document = f.read()

    generator = pipeline("text-generation", model="gpt2")
    completion = generator(document, max_length=100, num_return_sequences=1)
    return completion[0]['generated_text']
