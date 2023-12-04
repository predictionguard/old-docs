---
title: Translation
---

# Translation/ multi-lingual LLM app 
## Introduction

The translation functionality in our API allows users to seamlessly translate text between different languages. This section provides an overview of the translation process, the models used, and details about language support.

## Supported Languages

Our translation API supports a wide range of languages, including but not limited to English,Hindi, French, Spanish, German, and more. Refer to the language codes to identify specific languages.

![List of supported langauges !!!](./languages.png)

# Code Overview

## Install packages
```python
import os

from pydantic import BaseModel
import predictionguard as pg
import pandas as pd
from langchain import PromptTemplate
from comet import download_model, load_from_checkpoint
import deepl
from fastapi import FastAPI, HTTPException
import uvicorn
import pandas as pd
from io import BytesIO
import nltk
import translators as ts
import translators.server as tss

# Download the NLTK data for word tokenization
nltk.download('punkt')
```

## Login Tokens
```python
PREDICTIONGUARD_TOKEN=os.getenv("PREDICTIONGUARD_TOKEN")
OPENAI_API_KEY=os.getenv("OPENAI_API_KEY")
deepl_translator=deepl.Translator(auth_key=os.getenv("DEEPL_API_KEY"))
```
The code initializes login tokens for various APIs, including PredictionGuard, OpenAI, and DeepL.

## Translation Templates
```python
trans_template=""" ### Instruction:
Translate the following {source_language} input to {target_language}.

### Input:
{source_language} {input}

### Response:
"""

trans_prompt = PromptTemplate(template=trans_template, input_variables=["input","source_language","target_language"])
```
Defines a prompt template for translation instructions which takes in the input text, source language and target language.

## ISO Code Languages Data
```python
# Reads language data from an S3 bucket
df = pd.read_csv(BytesIO(csv_content))
```
Reads language data from an AWS S3 bucket and creates a DataFrame. This dataset has iso639 language names, codes, supporting models codes.

## Quality Check
```python
model_path = download_model("Unbabel/wmt20-comet-qe-da")
model = load_from_checkpoint(model_path)
```
Downloads and loads a quality estimation model for translation quality check. This state of the art scoring model gives out scores with higher scores showing higher translation quality and vice-e-versa.

## MT API Models
Functions for translation using different models like "Nous-Hermes-Llama2-13B" , "openai-text-davinci-003", DeepL, and Google Translate.

Below shows modelling structure for LLama2 , similar fashion for open-ai.

```python
def nous_hermes_llama2_translation(text, source_language, target_language):
    # Calculate the number of words in the paragraph_text
    num_words = len(nltk.word_tokenize(text))
    print("num_words", num_words)
    dynamic_max_tokens = max(num_words * 2, min_tokens)

    result = pg.Completion.create(
        model="Nous-Hermes-Llama2-13B",
        prompt=trans_prompt.format(input=text, 
        source_language=source_language, target_language=target_language),
        temperature=0.1,
        max_tokens=600
    )

    response = result['choices'][0]['text']

    qa_input = QAInput(source=text, translation=response)
    score = get_quality_score(qa_input).score

    return {"Response": response, "Score": score}


```
For DeepL and google translate:
```python
def deepl_translation(text, source_language, target_language):
    response = deepl_translator.translate_text(text, target_lang=target_language).text

    # Check if the response is None or empty/whitespace
    if response is None or not response.strip():
        response = " "  # Set a default response or handle it as needed

    qa_input = QAInput(source=text, translation=response)
    score = get_quality_score(qa_input).score

    return {"Response": response, "Score": score}

def google_translation(text, source_language, target_language):
    translation = tss.google(
        input.text, 
        from_language=source_language, 
        to_language=target_language
    )
    

    qa_input = QAInput(source=text, translation=response)
    score = get_quality_score(qa_input).score

    return {"Response": response, "Score": score}
```

Defines translation models and loads language mapping from the DataFrame.
```python
# Define the translation models
translation_models = {
    'openai': openai_translation,
    'deepl': deepl_translation,
    'google': google_translation,
    'nous_hermes_llama2': nous_hermes_llama2_translation,
}

# Load the language-to-model mapping from the provided DataFrame
language_mapping_df = df
```

## Translate and Score Function
Below function takes in the input as user-input text, source language and target language. Translation functions run based on whichever models support them, scores these translations and returns the best translation & score.
```python
def translate_and_score(text, source_language_iso639, target_language_iso639):
    available_models = []

    # Check the target language column for each model
    for model in translation_models.keys():
        target_column = model + '_code'
        if not pd.isna(language_mapping_df[language_mapping_df['code'] == target_language_iso639][target_column].values[0]):
            available_models.append(model)

    best_translation = None
    best_score = -1
    print("available models",available_models)

    for model in available_models:
        translation_function = translation_models[model]
        target_column = model + '_code'
        target_language_code = language_mapping_df[language_mapping_df['code'] == target_language_iso639][target_column].values[0]
        source_language_code= language_mapping_df[language_mapping_df['code'] == source_language_iso639][target_column].values[0]
        translation_result = translation_function(text, source_language_code, target_language_code)

        if translation_result['Score'] > best_score:
            best_translation = translation_result['Response']
            best_score = translation_result['Score']
    if best_translation is None:
        best_translation = "Could not translate"

    return {"Best translation": best_translation, "Score": best_score}
```
## API
Defines a FastAPI application with two endpoints:  
1. GET "/" returns a simple "Hello World" message.  
2. POST /translate takes a translation request, validates language codes, and returns the best translation and its score.

```python
#---------------------#
# API                   #
#---------------------#
app = FastAPI()

@app.get("/")
def read_root():
    return {"Hello": "World"}

class TranslateRequest(BaseModel):
    text: str
    source_lang: str
    target_lang: str

def is_valid_language(language_code):
    return language_code in df["code"].to_list()

@app.post("/translate")
def update_item(req: TranslateRequest):
    if not is_valid_language(req.source_lang) or not is_valid_language(req.target_lang):
        raise HTTPException(status_code=400, detail="Invalid language code(s)")

    # Now you can proceed with the translation
    return {"response": translate_and_score(req.text, req.source_lang, req.target_lang)}
```

## Run the FastAPI Application
```python
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, port=8080, host="0.0.0.0")
```
Runs the FastAPI application on port 8080.

This code creates an API for translating text using multiple translation models, performs quality checks, and returns the best translation along with its quality score.


### `/translate`

Endpoint for translating text from one language to another. (Input source and target lang should be in ISO639 language format i.e English : eng, Russian: rus, Hindi: hin etc)

#### Request

- **Method:** POST
- **Path:** `/translate`

##### Request Body

```json
{
  "text": "Hello, world!",
  "source_lang": "eng",
  "target_lang": "fra"
}
```
##### Response
```json
{
  "response": "Bonjour, le monde!",
  "score": 0.95
}
```
