---
title: Data Extraction
description: Data extract using PG with factuality checks
---

# Data extraction example with factuality check
This document demonstrates the extraction of patient information from doctor-patient transcripts using PredictionGuard (PG) hosted LLMs and their wrappers for factuality checks. The example focuses on the first 5 rows of a dataset containing these transcripts.

## Load the data 

Donwload the data from the [json file](pages/usingllms/transcript_data/transcripts.json).   
Below code snippet loads the necessary libraries and the dataset from the above mentioned JSON file. It converts the data into a Pandas DataFrame and selects the first 5 rows for testing.

```python
import json
import pandas as pd
import itertools
from langchain import PromptTemplate
import predictionguard as pg

data = []
with open('transcripts.json') as f:
    for line in itertools.islice(f, 5):
        line = line.strip()
        if not line: continue
        data.append(json.loads(line))

df = pd.DataFrame(data)

#transform rows to columns
df = df.transpose()

# Reset the index and assign an index column
df = df.reset_index()
df.columns = ['id', 'transcript']

# Reset the index (optional)
df.reset_index(drop=True, inplace=True)

#Just to test with starting 5 rows of the dataframe
df=df.head(5)
```

## Summarize the data
This code snippet defines a template for summarization, creates a prompt template using langchain, and uses PredictionGuard hosted state-of-the-art LLM - "Nous-Hermes-Llama2-13B" to generate summaries for each transcript. The generated summaries are added as a new column in the DataFrame and saved to a CSV file.


```python
summarize_template = """### Instruction:
Summarize the input transcript below.

### Input:
{transcript}

### Response:
"""

summary_prompt = PromptTemplate(template=summarize_template,
    input_variables=["context"],
)
summaries = []
for i,row in df.iterrows():
    result=pg.Completion.create(
        model="Nous-Hermes-Llama2-13B",
        prompt=summary_prompt.format(
            transcript=row['transcript']
        ),
        max_tokens=200,
        temperature=0.1
    )
    print(result['choices'][0]['text'])
    summaries.append(result['choices'][0]['text'])

df['summary']=summaries
print(df.head(5))
df.to_csv("summarized_transcripts.csv")
```
## Extract Information and Perform Factuality Checks
Load the data from the csv and add a prompt with the relevant questions for the data that we are interested to collect - symptoms, Patient name, when the symptom started, level of pain the patient is experiencing

```python
df=pd.read_csv("summarized_transcripts.csv")

questions=["What symptoms is the patient experiencing",
           "What is the Patient's name?",
           "When did the symptoms start?",
           "On a scale of 1 to 10, what level of pain is the patient experiencing?"]

question_answer_template = """### Instruction:
Answer the following question {question} using the below doctor-patient transcript summary.

### Input:
{transcript_summary}

### Response:
"""

q_and_a_prompt = PromptTemplate(template=question_answer_template,
    input_variables=["question", "transcript_summary"],
)
```
## Factuality Checks

Factuality checks are crucial for evaluating the accuracy of information provided by language models, especially when dealing with potential hallucinations or inaccuracies. PredictionGuard leverages state-of-the-art models for factuality checks, ensuring the reliability of outputs against the context of the prompts.

### Scoring and Thresholds
Factuality scores are obtained during the factuality check, indicating the degree of accuracy in the generated responses. Navigating the landscape of language models can be challenging, and factuality scores provide a quantitative measure to assess the reliability of the model's output.

Using the defined template, we prompt the model again with each question and evaluates the responses against the corresponding transcript summaries. Factuality scores are generated to assess the accuracy of the answers.

```python
answers = {q: [] for q in questions}
fact_scores = {q: [] for q in questions}

for i, row in df.iterrows():
    for q in questions:
        result = pg.Completion.create(
            model="Nous-Hermes-Llama2-13B",
            prompt=q_and_a_prompt.format(
                question=q, transcript_summary=row["summary"]
            ),
            max_tokens=200,
            temperature=0.1,

        )
        fact_score = pg.Factuality.check(
            reference=row['summary'],
            text=result['choices'][0]['text']
        )
        fact_scores[q].append(fact_score['checks'][0]['score'])
        answers[q].append(result["choices"][0]["text"])

# Add the answers and fact scores as new columns to the original DataFrame
for q in questions:
    df[f"{q}_answer"] = answers[q]
    df[f"{q}_fact_score"] = fact_scores[q]

print(df.head(2))
df.to_csv("answers_with_fact_scores.csv", index=False)
```

|    | id   | transcript   | summary   | What symptoms is the patient experiencing_answer   | What symptoms is the patient experiencing_fact_score   | What is the Patient's name?_answer   | What is the Patient's name?_fact_score   | When did the symptoms start?_answer   | When did the symptoms start?_fact_score   | On a scale of 1 to 10, what level of pain is the patient experiencing?_answer   | On a scale of 1 to 10, what level of pain is the patient experiencing?_fact_score   |
|---:|:-----|:-------------|:----------|:--------------------------------------------------|:------------------------------------------------------|:-------------------------------------|:-----------------------------------------|:-------------------------------------|:-----------------------------------------|:-----------------------------------------------------------------------------|:----------------------------------------------------------------------------------|
|  0 | 2055 | During the... | During a... | The patient, Mr. Don Hicks, is experiencing sym... | 0.08922114223                                        | The patient's name is Mr. Don Hicks. | 0.451582998                                 | The symptoms started when Mr. Don ... | 0.1504420638                               | The transcript summary does not conta... | 0.5611280203                                                                   |
|  1 | 291  | During the... | During a... | The patient, Tina Will, is experiencing sympt... | 0.3320894539                                        | The patient's name is Tina Will.     | 0.8268791437                               | The symptoms started when Tina pre... | 0.7537286878                               | I am sorry to hear that Tina is expe... | 0.2882582843                                                                   |  
|  2 | 102  | "D: Good mo... | The patien... | The patient, Tommie, has been experiencing sy... | 0.1203972548                                        | I'm sorry, the question "What is t... | 0.6292911172                               | The symptoms started when?             | 0.7372002602                               | "I'm sorry to hear that Tommie has b... | 0.1583527327                                                                   |
|  3 | 2966 | "D: Good mo... | The patien... | The patient, Chris, is experiencing symptoms... | 0.03648262098                                       | The patient's name is Chris.         | 0.8302355409                               | The symptoms started when Chris exp... | 0.8345838189                               | I'm sorry to hear that Chris is expe... | 0.7252672315                                                                   | 
|  4 | 2438 | "D: Hi Erne... | Ernest visi... | The patient, Ernest, is experiencing bladder... | 0.149951458                                        | The patient's name is Ernest.        | 0.6766917109                               | The symptoms started when Ernest st... | 0.1891670823                               | Based on the information provided, i... | 0.6463367343                                                                   |


### Standalone Factuality Functionality

If needed, you can directly access the factuality checking functionality using the /factuality endpoint. This allows you to configure thresholds and score arbitrary inputs independently. Adding factuality=True or utilizing the /factuality endpoint provides an additional layer of control and customization for factuality checks.