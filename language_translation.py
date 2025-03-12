import os
from transformers import MarianMTModel, MarianTokenizer
from sacremoses import MosesTokenizer, MosesDetokenizer

# Set the model path
model_path = os.path.join(os.getcwd(),"models/en-de")

# Load the tokenizer and model (completely offline)
tokenizer = MarianTokenizer.from_pretrained(model_path, use_fast=False)
model = MarianMTModel.from_pretrained(model_path)
mt_en = MosesTokenizer(lang="en")
mt_de = MosesDetokenizer(lang="de")

def translate(text):
    tokens = mt_en.tokenize(text, return_str=True)

    # Encode & translate
    inputs = tokenizer(tokens, return_tensors="pt", padding=True, truncation=True)
    translated_tokens = model.generate(**inputs)

    # Decode and de-tokenize
    translated_text = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]
    return mt_de.detokenize(translated_text.split())


text = "My work is interesting, but very stressful"
translated_text = translate(text)
print("Translated:", translated_text)
