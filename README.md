# Language Translation using MarianMT (Offline)

## Overview
This project provides an **offline** translation system using the **MarianMT** model from Hugging Face. It supports English-to-German (en-de) translation without an internet connection, utilizing pre-downloaded models.

## Features
- **Completely offline translation** using MarianMT
- **Tokenization & Detokenization** with Moses tokenizer
- **Fast & efficient translation** for short and long texts

## Installation
### 1. Install Dependencies
Make sure you have Python installed, then install the required libraries:
```bash
pip install transformers sacremoses torch
```

### 2. Download the MarianMT Model
Since this project works **offline**, you need to download the model manually:
1. Visit [Hugging Face's MarianMT en-de model](https://huggingface.co/Helsinki-NLP/opus-mt-en-de).
2. Download the model files and store them in `models/en-de/`.

## Usage
### Running the Script
Run the Python script to translate text from English to German:
```bash
python translate.py
```

### Example Code
```python
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
```

### Expected Output
```
Translated: Meine Arbeit ist interessant, aber sehr stressig.
```

## Directory Structure
```
Language_Translation/
│── models/
│   ├── en-de/
│   │   ├── config.json
│   │   ├── pytorch_model.bin
│   │   ├── tokenizer.json
│   │   ├── ...
│── translate.py
│── README.md
```

## Troubleshooting
- **Error: Model not found** → Ensure that the MarianMT model is downloaded into `models/en-de/`
- **Missing dependencies** → Install all required libraries using `pip install transformers sacremoses torch`

## License
This project is open-source under the MIT License.

