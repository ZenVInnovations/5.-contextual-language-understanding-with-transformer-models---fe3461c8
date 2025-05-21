from transformers import pipeline
import json

def answer_question(context_path):
    with open(context_path, 'r') as f:
        data = json.load(f)

    nlp_qa = pipeline('question-answering', model='distilbert-base-uncased-distilled-squad')
    result = nlp_qa({
        'question': data['question'],
        'context': data['context']
    })
    return result['answer']
