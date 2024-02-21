---
title: Data Chat with LLMs
---

# Using LLMs for Data Analysis and SQL Query Generation

(Run this example in Google Colab [here](https://colab.research.google.com/drive/15nOCaau9tXpov3MHg3fnuWWyfavxdaSA?usp=sharing))

Large language models (LLMs) like 'deepseek-coder-6.7B-instruct' have demonstrated impressive capabilities for understanding natural language and generating SQL. We can leverage these skills for data analysis by having them automatically generate SQL queries against known database structures. And then rephrase these sql outputs using state of the art text/chat completion models like 'Neural-Chat-7B' to get well written answers to user questions.

Unlike code generation interfaces that attempt to produce executable code from scratch, our approach focuses strictly on generating industry-standard SQL from plain English questions. This provides two major benefits:

- SQL is a well-established language supported across environments, avoiding the need to execute less secure auto-generated code. 

- Mapping natural language questions to SQL over known schemas is more robust than attempting to generate arbitrary code for unfamiliar data structures.  

By combining language model understanding of questions with a defined database schema, the system can translate simple natural language queries into precise SQL for fast and reliable data analysis. This makes surfacing insights more accessible compared to manual SQL writing or hopelessly broad code generation.

For this demo we have selecteed a public dataset from Kaggle - Jobs and Salaries in Data Science (Find the dataset [here](https://www.kaggle.com/datasets/hummaamqaasim/jobs-in-data?resource=download))

## Installation and Setup
- Install the Python SDK with `pip install predictionguard`
- Get a Prediction Guard access token (as described [here](https://docs.predictionguard.com/)) and set it as the environment variable `PREDICTIONGUARD_TOKEN`.

## Setup

First, import the necessary libraries:

```python
import time
import os

import pandas as pd
from langchain import PromptTemplate
import predictionguard as pg
import requests
import duckdb
import re
import json
import numpy as np
from getpass import getpass
```

## Data Loading
Load data into a pandas DataFrame.

```python
df=pd.read_csv('jobs_in_data.csv')
df.head()
```

## Data Cleaning and preprocessing
After having a look at the dataset , make data cleaning/preprocessing decisions if needed.

```python
df = df[df['work_year'].isin(['2023', '2022', '2021', '2020'])]
df = df.astype({
    'work_year': 'int',
    'salary': 'int',
    'salary_in_usd': 'int'
})
```

## Connect to the Database 
```python
# Here we will use DuckDB (an in process database) as a SQL database
# interface to our data. In real world use cases, you could still use DuckDB
# to connect to your database, or you could connect via another client.

# Create an in memory database with DuckDB
# see: https://duckdb.org/docs/api/python/dbapi.html
conn = duckdb.connect(database=':memory:')

# make our data frame available as a view in duckdb
conn.register('jobs', df)
```

## Define the Schema
```python
# Here we will create an example SQL schema based on the data in this dataset.
# In a real use case, you likely already have this sort of CREATE TABLE statement.
# Performance can be improved by manually curating the descriptions.

columns_info = []

# Iterate through each column in the DataFrame
for col in df.columns:
    # Determine the SQL data type based on the first non-null value in the column
    first_non_null = df[col].dropna().iloc[0]
    if isinstance(first_non_null, np.int64):
        kind = "INTEGER"
    elif isinstance(first_non_null, float):
        kind = "DECIMAL(10,2)"
    elif isinstance(first_non_null, str):
        kind = "VARCHAR(255)"  # Assuming a default max length of 255
    else:
        kind = "VARCHAR(255)"  # Default to VARCHAR for other types or customize as needed

    # Sample a few example values
    example_values = ', '.join([str(x) for x in df[col].dropna().unique()[0:4]])

    # Append column info to the list
    columns_info.append(f"{col} {kind}, -- Example values are {example_values}")

# Construct the CREATE TABLE statement
create_table_statement = "CREATE TABLE jobs (\n  " + ",\n  ".join(columns_info) + "\n);"

# Adjust the statement to handle the final comma, primary keys, or other specifics
create_table_statement = create_table_statement.replace(",\n);", "\n);")

print(create_table_statement)
```

Schema for this example dataset will look like this :

```python
CREATE TABLE df (
  work_year INTEGER, -- Example values are 2023, 2022, 2020, 2021,
  job_title VARCHAR(255), -- Example values are Data DevOps Engineer, Data Architect, Data Scientist, Machine Learning Researcher,
  job_category VARCHAR(255), -- Example values are Data Engineering, Data Architecture and Modeling, Data Science and Research, Machine Learning and AI,
  salary_currency VARCHAR(255), -- Example values are EUR, USD, GBP, CAD,
  salary INTEGER, -- Example values are 88000, 186000, 81800, 212000,
  salary_in_usd INTEGER, -- Example values are 95012, 186000, 81800, 212000,
  employee_residence VARCHAR(255), -- Example values are Germany, United States, United Kingdom, Canada,
  experience_level VARCHAR(255), -- Example values are Mid-level, Senior, Executive, Entry-level,
  employment_type VARCHAR(255), -- Example values are Full-time, Part-time, Contract, Freelance,
  work_setting VARCHAR(255), -- Example values are Hybrid, In-person, Remote,
  company_location VARCHAR(255), -- Example values are Germany, United States, United Kingdom, Canada,
  company_size VARCHAR(255), -- Example values are L, M, S
);
```
## Prompt Templates
Define prompt templates for generating SQL queries and chatbot responses using Prediction Guard:

```python
qa_template = """### System:
You are a data chatbot who answers the user question. To answer these questions we need to run SQL queries on our data and its output is given below in context. You just have to frame your answer using that context. Give a short and crisp response.Don't add any notes or any extra information after your response.

### User:
Question: {question}

context: {context}

### Assistant:
"""

qa_prompt = PromptTemplate(template=qa_template,input_variables=["thread", "question", "context"])

sql_template = """<|begin_of_sentence|>You are a SQL expert and you only generate SQL queries which are executable. You provide no extra explanations.
You respond with a SQL query that answers the user question in the below instruction by querying a database with the schema provided in the below instruction.
Always start your query with SELECT statement and end with a semicolon.

### Instruction:
User question: \"{question}\"

Database schema:
{schema}

### Response:
"""
sql_prompt=PromptTemplate(template=sql_template, input_variables=["question","schema"])
```

## Generating and Refining SQL Queries
Generate SQL queries based on user questions using PredictionGuard and process the queries:

```python
def generate_sql_query(question):

  prompt_filled = sql_prompt.format(question=question,schema=create_table_statement)

  try:
      result = pg.Completion.create(
          model="deepseek-coder-6.7b-instruct",
          prompt=prompt_filled,
          max_tokens=300,
          temperature=0.1
      )
      sql_query = result["choices"][0]["text"]
      return sql_query

  except Exception as e:
      return None


def extract_and_refine_sql_query(sql_query):

  # Extract SQL query using a regular expression
  match = re.search(r"(SELECT.*?);", sql_query, re.DOTALL)
  if match:

      refined_query = match.group(1)

      # Check for and remove any text after a colon
      colon_index = refined_query.find(':')
      if colon_index != -1:
          refined_query = refined_query[:colon_index]

      # Ensure the query ends with a semicolon
      if not refined_query.endswith(';'):
          refined_query += ';'
      return refined_query

  else:
      return ""


def get_answer_from_sql(question):

  sql_query = generate_sql_query(question)
  sql_query = extract_and_refine_sql_query(sql_query)

  try:

      # print("Executing SQL Query:", sql_query)
      result = conn.execute(sql_query).fetchall()

      # print("Result:", result)
      return result, sql_query

  except Exception as e:

      print(f"Error executing SQL query: {e}")
      return "There was an error executing the SQL query."
```

## Return natural language responses
Generate responses to user questions based on SQL query results:

```python
def get_answer(question,context):
  try:

      prompt_filled = qa_prompt.format(question=question, context=context)

      # Respond to the user
      output = pg.Completion.create(
          model="Neural-Chat-7B",
          prompt=prompt_filled,
          max_tokens=200,
          temperature=0.1
      )
      completion = output['choices'][0]['text']

      return completion

  except Exception as e:
      completion = "There was an error executing the SQL query."
      return completion
```

## Test it out
```python
question = "What is the average salary of a data scientist?"

context, sql_query = get_answer_from_sql(question)
answer = get_answer(question, context)

# Convert context and answer to string if they are not already
context_str = ', '.join([str(item) for item in context]) if isinstance(context, list) else str(context)
answer_str = str(answer)

print('Question:')
print('------------------------')
print(question)
print('')
print('Generated SQL Query:')
print('------------------------')
print(sql_query)
print('')
print('SQL result:')
print('------------------------')
print(context)
print('')
print('Generate NL answer:')
print('------------------------')
print(answer)
```

Output :

![Query to Response !!](./datachat-3.png)

## For Multiple Tables
In this case we will be using a RAG based approach which will involve semantic comparison of user questions with the tables we have in our database. Once we have shortlisted the most relavent table having the data , we then process that table with the similar process as above to generate the answer.

You can go through this code to replicate for your various use cases :
(Run this example in Google Colab [here](https://colab.research.google.com/drive/1zj9sic1H-tAw3BRvPGZAwGdMg4l5Y2FU?usp=sharing))

## For an interactive UI
In this case we will be using a streamlit based web application to create an appealing chat interface.

You can go through this code to replicate for your various use cases :
(Find relavent codes and details for this in our github repo[here](https://github.com/predictionguard/datachat-streamlit))

Chatbot in action :

  <iframe
    src="https://www.loom.com/embed/7f35885e57f14c1daf0af1c71e019c06"
    width="100%"
    height="500px"
    title="SWR-States"
  />

## Conclusion

This document outlines the structure of a application designed for interactive data analysis through a chat interface. It leverages several advanced Python libraries and techniques, including vector embeddings with LanceDB, executing SQL queries on pandas dataframes, and generating dynamic responses using LangChain and PredictionGuard.

