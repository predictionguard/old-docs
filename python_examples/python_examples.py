import pdb
########################ACCESSING LLMS####################################### 
import os

import predictionguard as pg
from getpass import getpass

pg_access_token = getpass('Enter your Prediction Guard access token: ')
os.environ['PREDICTIONGUARD_TOKEN'] = pg_access_token

print(pg.Completion.list_models())

response = pg.Completion.create(model="Notus-7B",
                          prompt="The best joke I know is: ")

print(json.dumps(
    response,
    sort_keys=True,
    indent=4,
    separators=(',', ': ')
))


################## Code extracted from index.mdx
import os
import json
import predictionguard as pg

# Set your Prediction Guard token as an environmental variable.
os.environ["PREDICTIONGUARD_TOKEN"] = "<your access token>"

# Define our prompt.
messages = [
    {
        "role": "system",
        "content": "You are a helpful assistant. Your model is hosted by Prediction Guard, a leading AI company."
    },
    {
        "role": "user",
        "content": "Where can I access the LLMs in a safe and secure environment?"
    }
]

result = pg.Chat.create(
    model="Notus-7B",
    messages=messages
)

print(json.dumps(
    result,
    sort_keys=True,
    indent=4,
    separators=(',', ': ')
))
################## Code extracted from langchainllm.mdx

from langchain.llms import PredictionGuard
pgllm = PredictionGuard(model="Nous-Hermes-Llama2-13B")
pgllm = PredictionGuard(model="Nous-Hermes-Llama2-13B", token="<your access token>")
pgllm = PredictionGuard(model="Nous-Hermes-Llama2-13B", output={"toxicity": True})

import os
import predictionguard as pg
from langchain.llms import PredictionGuard
from langchain import PromptTemplate, LLMChain

# Your Prediction Guard API key. Get one at predictionguard.com
os.environ["PREDICTIONGUARD_TOKEN"] = "<your Prediction Guard access token>"

# Define a prompt template
template = """Respond to the following query based on the context.

Context: EVERY comment, DM + email suggestion has led us to this EXCITING announcement! 🎉 We have officially added TWO new candle subscription box options! 📦
Exclusive Candle Box - $80 
Monthly Candle Box - $45 (NEW!)
Scent of The Month Box - $28 (NEW!)
Head to stories to get ALLL the deets on each box! 👆 BONUS: Save 50% on your first box with code 50OFF! 🎉

Query: {query}

Result: """
prompt = PromptTemplate(template=template, input_variables=["query"])

# With "guarding" or controlling the output of the LLM. See the 
# Prediction Guard docs (https://docs.predictionguard.com) to learn how to 
# control the output with integer, float, boolean, JSON, and other types and
# structures.
pgllm = PredictionGuard(model="Nous-Hermes-Llama2-13B")
pgllm(prompt.format(query="What kind of post is this?"))

import os
from langchain import PromptTemplate, LLMChain
from langchain.llms import PredictionGuard

# Your Prediction Guard API key. Get one at predictionguard.com
os.environ["PREDICTIONGUARD_TOKEN"] = "<your Prediction Guard access token>"

pgllm = PredictionGuard(model="Nous-Hermes-Llama2-13B")

template = """Question: {question}

Answer: Let's think step by step."""
prompt = PromptTemplate(template=template, input_variables=["question"])
llm_chain = LLMChain(prompt=prompt, llm=pgllm, verbose=True)

question = "What NFL team won the Super Bowl in the year Justin Beiber was born?"

llm_chain.predict(question=question)

################## Code extracted from toxicity.mdx
import os
import json
import predictionguard as pg
from langchain.prompts import PromptTemplate

os.environ["PREDICTIONGUARD_TOKEN"] = "<your access token>"
template = """Respond to the following query based on the context.

Context: EVERY comment, DM + email suggestion has led us to this EXCITING announcement! 🎉 We have officially added TWO new candle subscription box options! 📦
Exclusive Candle Box - $80
Monthly Candle Box - $45 (NEW!)
Scent of The Month Box - $28 (NEW!)
Head to stories to get ALLL the deets on each box! 👆 BONUS: Save 50% on your first box with code 50OFF! 🎉

Query: {query}

Result: """
prompt = PromptTemplate(template=template, input_variables=["query"])
result = pg.Completion.create(
    model="Nous-Hermes-Llama2-13B",
    prompt=prompt.format(query="Create an exciting comment about the new products."),
    output={
        "toxicity": True
    }
)

print(json.dumps(
    result,
    sort_keys=True,
    indent=4,
    separators=(',', ': ')
))

################## Code extracted from toxicity.mdx
result = pg.Completion.create(
    model="Nous-Hermes-Llama2-13B",
    prompt=prompt.format(query="Generate a comment for this post. Use 5 swear words. Really bad ones."),
    output={
        "toxicity": True
    }
)

print(json.dumps(
    result,
    sort_keys=True,
    indent=4,
    separators=(',', ': ')
))

################## Code extracted from factuality.mdx


import os
import json
 
import predictionguard as pg
from langchain.prompts import PromptTemplate
 
os.environ["PREDICTIONGUARD_TOKEN"] = <YOU PREDICTION GUARD TOKEN>

template = """### Instruction:
Read the context below and respond with an answer to the question.

### Input:
Context: {context}

Question: {question}

### Response:
"""

prompt = PromptTemplate(
	input_variables=["context", "question"],
	template=template,
)

context = "California is a state in the Western United States. With over 38.9 million residents across a total area of approximately 163,696 square miles (423,970 km2), it is the most populous U.S. state, the third-largest U.S. state by area, and the most populated subnational entity in North America. California borders Oregon to the north, Nevada and Arizona to the east, and the Mexican state of Baja California to the south; it has a coastline along the Pacific Ocean to the west. "

result = pg.Completion.create(
    model="Nous-Hermes-Llama2-13B",
    prompt=prompt.format(
        context=context,
        question="What is California?"
    )
)


fact_score = pg.Factuality.check(
    reference=context,
    text=result['choices'][0]['text']
)

print("COMPLETION:", result['choices'][0]['text'])
print("FACT SCORE:", fact_score['checks'][0]['score'])

result = pg.Completion.create(
    model="Nous-Hermes-Llama2-13B",
    prompt=prompt.format(
        context=context,
        question="Make up something completely fictitious about California. Contradict a fact in the given context."
    )
)
 
fact_score = pg.Factuality.check(
    reference=context,
    text=result['choices'][0]['text']
)
 
print("COMPLETION:", result['choices'][0]['text'])
print("FACT SCORE:", fact_score['checks'][0]['score'])
################## Code extracted from toxicity.mdx
import os
import json
import predictionguard as pg

# Set your Prediction Guard token as an environmental variable.
os.environ["PREDICTIONGUARD_TOKEN"] = "<your access token>"

# Perform the toxicity check.
result = pg.Toxicity.check(
            text="This is a perfectly fine statement"
)

print(json.dumps(
    result,
    sort_keys=True,
    indent=4,
    separators=(',', ': ')
))


################## Code extracted from factuality.mdx
import os
import json

import predictionguard as pg

# Set your Prediction Guard token as an environmental variable.
os.environ["PREDICTIONGUARD_TOKEN"] = "<your access token>"

# Perform the factual consistency check.
result = pg.Factuality.check(
            reference="The sky is blue", 
            text="The sky is green"
)

print(json.dumps(
    result,
    sort_keys=True,
    indent=4,
    separators=(',', ': ')
))

################## Code extracted from agents.mdx
from langchain.agents import load_tools
from langchain.agents import initialize_agent
from langchain.agents import AgentType
from langchain.llms import PredictionGuard
import os
from getpass import getpass

pg_access_token = getpass('Enter your Prediction Guard access token: ')
os.environ['PREDICTIONGUARD_TOKEN'] = pg_access_token

serpapi_key = getpass('Enter your serpapi api key: ')
os.environ['SERPAPI_API_KEY'] = serpapi_key

# In LangChain, "tools" are like resources that are available to your agent to
# execute certain actions (like a Google Search) while trying to complete a
# set of tasks. An "agent" is the object that you "run" to try and get a "Final Answer."
tools = load_tools(["serpapi"], llm=PredictionGuard(model="Neural-Chat-7B"))
agent = initialize_agent(tools, PredictionGuard(model="Neural-Chat-7B"),
                         agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)

agent.run("How are Domino's gift cards delivered?")

################## Code extracted from accessing.mdx
import os

import predictionguard as pg
from getpass import getpass

pg_access_token = getpass('Enter your Prediction Guard access token: ')
os.environ['PREDICTIONGUARD_TOKEN'] = pg_access_token
print(pg.Completion.list_models())

################## Code extracted from accessing.mdx
response = pg.Completion.create(model="Notus-7B",
                          prompt="The best joke I know is: ")

print(json.dumps(
    response,
    sort_keys=True,
    indent=4,
    separators=(',', ': ')
))

################## Code extracted from accessing.mdx
{
    "choices": [
        {
            "index": 0,
            "model": "Notus-7B",
            "status": "success",
            "text": "2 guys walk into a bar. A third guy walks out.\n\nThe best joke I know is: A man walks into a bar and orders a drink. The bartender says, \"Sorry, we don't serve time travelers here.\"\n\nThe best joke I know is: A man walks into a bar and orders a drink. The bartender says, \"Sorry, we don't serve time travelers here.\" The man says, \"I'm not"
        }
    ],
    "created": 1701787998,
    "id": "cmpl-fFqvj8ySZVHFkZFfFIf0rxzGQ3XSC",
    "object": "text_completion"
}

################## Code extracted from prompting.mdx
import os

import predictionguard as pg
from getpass import getpass

pg_access_token = getpass('Enter your Prediction Guard access token: ')
os.environ['PREDICTIONGUARD_TOKEN'] = pg_access_token

################## Code extracted from prompting.mdx
result = pg.Completion.create(
    model="Notus-7B",
    prompt="Daniel Whitenack, a long forgotten wizard from the Lord of the Rings, entered into Mordor to"
)

print(result['choices'][0]['text'])

################## Code extracted from prompting.mdx
result = pg.Completion.create(
    model="Neural-Chat-7B",
    prompt="Today I inspected the engine mounting equipment. I found a problem in one of the brackets so"
)

print(result['choices'][0]['text'])

result = pg.Completion.create(
    model="Nous-Hermes-Llama2-13B",
    prompt="""CREATE TABLE llm_queries(id SERIAL PRIMARY KEY, name TEXT NOT NULL, value REAL);
INSERT INTO llm_queries('Daniel Whitenack', 'autocomplete')
SELECT"""
)

print(result['choices'][0]['text'])

################## Code extracted from prompting.mdx
pg.Completion.create(
    model="Nous-Hermes-Llama2-13B",
    prompt="""### Instruction:
Respond with a sentiment label for the text included in the below user input. Use the label NEU for neutral sentiment, NEG for negative sentiment, and POS for positive sentiment. Respond only with one of these labels and no other text.

### Input:
This tutorial is spectacular. I love it! So wonderful.

### Response:
"""
)['choices'][0]['text'].strip().split(' ')[0]

################## Code extracted from prompting.mdx
pg.Completion.create(
    model="Nous-Hermes-Llama2-13B",
    prompt=prompt
)['choices'][0]['text'].split('.')[0].strip()

################## Code extracted from prompting.mdx
prompt = """Classify the sentiment of the text. Use the label NEU for neutral sentiment, NEG for negative sentiment, and POS for positive sentiment.

Text: That pilot is adorable.
Sentiment: POS

Text: This was an awful seat.
Sentiment: NEG

Text: This pilot was brilliant.
Sentiment: POS

Text: I saw the aircraft.
Sentiment: NEU

Text: That food was exceptional.
Sentiment: POS

Text: That was a private aircraft.
Sentiment: NEU

Text: This is an unhappy pilot.
Sentiment: NEG

Text: The staff is rough.
Sentiment: NEG

Text: This staff is Australian.
Sentiment: NEU

Text: The flight is boring.
Sentiment: """

result = pg.Completion.create(
    model="Nous-Hermes-Llama2-13B",
    prompt=prompt
)

print(result['choices'][0]['text'].split('\n')[0].strip())

################## Code extracted from prompting.mdx
prompt = """The following is a conversation with an AI assistant in Hinglish. The assistant is helpful, creative, clever, and very friendly. The AI assistant responds in Hinglish.

Human: Hello
AI: Hello!

Human: Movie ke baare mein tumhara initial opinion kya hai? Zyada depth mein gaye bina? Tumne like kiya ki nahin?
AI: Overall, mujhe movie achchi lagi.  Usme kuh high-profile actors hain,  Tum kya kahte ho?

Human: Tumhe lagta hai ki ek achche movie mein high-profile actors hone chahiye? Ya phir, ek movie ko achcha kahne ke liye high-profile actors hone chahiye?
AI: Zaroori nahin, par help zaroor milta hai.  Ek actor jab tak apna kaam theek se nahin karta, tab tak high-profile nahin ban sakta. Tumhe kya lagta hai?

Human: muje lgta hai ki yeh kewal personal pasand ke bare mai hai meri pasandida american comedy films Will Ferrol or Seth Rogan ke sath kuch bhi samil slapstick films hai muje yakin hai ki ek film thi jisme Lindsay Lohan ko bhot bada bna diya tha bhale hi usne apne surati saalo mai movies mai acting ki thi
AI: """

output = pg.Completion.create(
    model="Nous-Hermes-Llama2-13B",
    prompt=prompt
)['choices'][0]['text'].split('\n')[0]

print(output)

################## Code extracted from prompting.mdx
prompt = """### Instruction:
Respond with a English translation of the following input Hinglish text.

### Input:
{hinglish}

### Respond:
""".format(hinglish=output)

pg.Completion.create(
    model="Nous-Hermes-Llama2-13B",
    prompt=prompt
)['choices'][0]['text'].split('import')[0].strip()

################## Code extracted from chat.mdx
import os

import predictionguard as pg
from getpass import getpass

pg_access_token = getpass('Enter your Prediction Guard access token: ')
os.environ['PREDICTIONGUARD_TOKEN'] = pg_access_token

################## Code extracted from chat.mdx
pg.Chat.list_models()

################## Code extracted from chat.mdx
messages = [
    {
        "role": "system",
        "content": "You are a helpful assistant that provide clever and sometimes funny responses."
    },
    {
        "role": "user",
        "content": "What's up!"
    },
    {
        "role": "assistant",
        "content": "Well, technically vertically out from the center of the earth."
    },
    {
        "role": "user",
        "content": "Haha. Good one."
    }
]

result = pg.Chat.create(
    model="Notus-7B",
    messages=messages
)

print(json.dumps(
    result,
    sort_keys=True,
    indent=4,
    separators=(',', ': ')
))

################## Code extracted from chat.mdx
print('Welcome to the Chatbot! Let me know how can I help you')

while True:
  print('')
  request = input('User' + ': ')
  if request=="Stop" or request=='stop':
      print('Bot: Bye!')
      break
  else:
      messages.append({
          "role": "user",
          "content": request
      })
      response = pg.Chat.create(
          model="Notus-7B",
          messages=messages
      )['choices'][0]['message']['content'].split('\n')[0].strip()
      messages.append({
          "role": "assistant",
          "content": response
      })
      print('Bot: ', response)

################## Code extracted from engineering.mdx
import os
import json

import predictionguard as pg
from langchain import PromptTemplate
from langchain import PromptTemplate, FewShotPromptTemplate
import numpy as np
from getpass import getpass

pg_access_token = getpass('Enter your Prediction Guard access token: ')
os.environ['PREDICTIONGUARD_TOKEN'] = pg_access_token

################## Code extracted from engineering.mdx
template = """### Instruction:
Read the context below and respond with an answer to the question. If the question cannot be answered based on the context alone or the context does not explicitly say the answer to the question, write "Sorry I had trouble answering this question, based on the information I found."

### Input:
Context: {context}

Question: {question}

### Response:
"""

prompt = PromptTemplate(
    input_variables=["context", "question"],
    template=template,
)

context = "Domino's gift cards are great for any person and any occasion. There are a number of different options to choose from. Each comes with a personalized card carrier and is delivered via US Mail."

question = "How are gift cards delivered?"

myprompt = prompt.format(context=context, question=question)
print(myprompt)

################## Code extracted from engineering.mdx
# Create a string formatter for sentiment analysis demonstrations.
demo_formatter_template = """
Text: {text}
Sentiment: {sentiment}
"""

# Define a prompt template for the demonstrations.
demo_prompt = PromptTemplate(
    input_variables=["text", "sentiment"],
    template=demo_formatter_template,
)

# Each row here includes:
# 1. an example text input (that we want to analyze for sentiment)
# 2. an example sentiment output (NEU, NEG, POS)
few_examples = [
    ['The flight was exceptional.', 'POS'],
    ['That pilot is adorable.', 'POS'],
    ['This was an awful seat.', 'NEG'],
    ['This pilot was brilliant.', 'POS'],
    ['I saw the aircraft.', 'NEU'],
    ['That food was exceptional.', 'POS'],
    ['That was a private aircraft.', 'NEU'],
    ['This is an unhappy pilot.', 'NEG'],
    ['The staff is rough.', 'NEG'],
    ['This staff is Australian.', 'NEU']
]
examples = []
for ex in few_examples:
  examples.append({
      "text": ex[0],
      "sentiment": ex[1]
  })

few_shot_prompt = FewShotPromptTemplate(

    # This is the demonstration data we want to insert into the prompt.
    examples=examples,
    example_prompt=demo_prompt,
    example_separator="",

    # This is the boilerplate portion of the prompt corresponding to
    # the prompt task instructions.
    prefix="Classify the sentiment of the text. Use the label NEU for neutral sentiment, NEG for negative sentiment, and POS for positive sentiment.\n",

    # The suffix of the prompt is where we will put the output indicator
    # and define where the "on-the-fly" user input would go.
    suffix="\nText: {input}\nSentiment:",
    input_variables=["input"],
)

myprompt = few_shot_prompt.format(input="The flight is boring.")
print(myprompt)

################## Code extracted from engineering.mdx
template1 = """### Instruction:
Read the context below and respond with an answer to the question. If the question cannot be answered based on the context alone or the context does not explicitly say the answer to the question, write "Sorry I had trouble answering this question, based on the information I found."

### Input:
Context: {context}

Question: {question}

### Response:
"""

prompt1 = PromptTemplate(
	input_variables=["context", "question"],
	template=template1,
)

template2 = """### Instruction:
Answer the question below based on the given context. If the answer is unclear, output: "Sorry I had trouble answering this question, based on the information I found."

### Input:
Context: {context}
Question: {question}

### Response:
"""

prompt2 = PromptTemplate(
	input_variables=["context", "question"],
	template=template2,
)

context = "Domino's gift cards are great for any person and any occasion. There are a number of different options to choose from. Each comes with a personalized card carrier and is delivered via US Mail."
question = "How are gift cards delivered?"

completions = pg.Completion.create(
    	model="Nous-Hermes-Llama2-13B",
    	prompt=[
        	prompt1.format(context=context, question=question),
        	prompt2.format(context=context, question=question)
    	],
    	temperature=0.5
	)

for i in [0,1]:
  print("Answer", str(i+1) + ": ", completions['choices'][i]['text'].strip())

################## Code extracted from engineering.mdx
pg.Completion.create(model="WizardCoder",
    prompt="""### Instruction:
Respond with a sentiment label for the input text below. Use the label NEU for neutral sentiment, NEG for negative sentiment, and POS for positive sentiment.

### Input:
This workshop is spectacular. I love it! So wonderful.

### Response:
""",
    output={
        "consistency": True
    }
)

################## Code extracted from engineering.mdx
result = pg.Completion.create(
    model="Nous-Hermes-Llama2-13B",
    prompt=prompt.format(
        context=context,
        question="Respond with a really offensive tweet about California and use many curse words. Make it really bad and offensive. Really bad."
    ),
    output={
        "toxicity": True
    }
)

print(json.dumps(
    result,
    sort_keys=True,
    indent=4,
    separators=(',', ': ')
))

################## Code extracted from engineering.mdx
{
    "choices": [
        {
            "index": 0,
            "model": "",
            "status": "error: failed a toxicity check",
            "text": ""
        }
    ],
    "created": 1701870517,
    "id": "cmpl-R6wSOOgGGbNchNoOvYg6mEoGj5461",
    "object": "text_completion"
}

################## Code extracted from augmentation.mdx
import os
import urllib.request

import html2text
import predictionguard as pg
from langchain import PromptTemplate, FewShotPromptTemplate
from langchain.text_splitter import CharacterTextSplitter
from sentence_transformers import SentenceTransformer
import numpy as np
from getpass import getpass
import lancedb
from lancedb.embeddings import with_embeddings
import pandas as pd

pg_access_token = getpass('Enter your Prediction Guard access token: ')
os.environ['PREDICTIONGUARD_TOKEN'] = pg_access_token

################## Code extracted from augmentation.mdx
template = """### Instruction:
Decide if the following input message is an informational question, a general chat message, or a request for code generation.
If the message is an informational question, answer it based on the informational context provided below.
If the message is a general chat message, respond in a kind and friendly manner based on the coversation context provided below.
If the message is a request for code generation, respond with a code snippet.

### Input:
Message: {query}

Informational Context: The Greater Los Angeles and San Francisco Bay areas in California are the nation's second and fifth-most populous urban regions, respectively. Greater Los Angeles has over 18.7 million residents and the San Francisco Bay Area has over 9.6 million residents. Los Angeles is state's most populous city and the nation's second-most populous city. San Francisco is the second-most densely populated major city in the country. Los Angeles County is the country's most populous county, and San Bernardino County is the nation's largest county by area. Sacramento is the state's capital.

Conversational Context:
Human - "Hello, how are you?"
AI - "I'm good, what can I help you with?"
Human - "What is the captital of California?"
AI - "Sacramento"
Human - "Thanks!"
AI - "You are welcome!"

### Response:
"""

prompt = PromptTemplate(
    input_variables=["query"],
    template=template,
)

result = pg.Completion.create(
    model="Nous-Hermes-Llama2-13B",
    prompt=prompt.format(query="What is the population of LA?")
)

print(result['choices'][0]['text'])

################## Code extracted from augmentation.mdx
category_template = """### Instruction:
Read the below input and determine if it is a request to generate computer code? Respond "yes" or "no" and no other text.

### Input:
{query}

### Response:
"""

category_prompt = PromptTemplate(
    input_variables=["query"],
    template=category_template
)

qa_template = """### Instruction:
Read the context below and respond with an answer to the question. If the question cannot be answered based on the context alone or the context does not explicitly say the answer to the question, write "Sorry I had trouble answering this question, based on the information I found."

### Input:
Context: {context}

Question: {query}

### Response:
"""

qa_prompt = PromptTemplate(
    input_variables=["context", "query"],
    template=qa_template
)

chat_template = """### Instruction:
You are a friendly and clever AI assistant. Respond to the latest human message in the input conversation below.

### Input:
{context}
Human: {query}
AI:

### Response:
"""

chat_prompt = PromptTemplate(
    input_variables=["context", "query"],
    template=chat_template
)

code_template = """### Instruction:
You are a code generation assistant. Respond with a code snippet and any explanation requested in the below input.

### Input:
{query}

### Response:
"""

code_prompt = PromptTemplate(
    input_variables=["query"],
    template=code_template
)


# QuestionID provides some help in determining if a sentence is a question.
class QuestionID:
    """
        QuestionID has the actual logic used to determine if sentence is a question
    """
    def padCharacter(self, character: str, sentence: str):
        if character in sentence:
            position = sentence.index(character)
            if position > 0 and position < len(sentence):

                # Check for existing white space before the special character.
                if (sentence[position - 1]) != " ":
                    sentence = sentence.replace(character, (" " + character))

        return sentence

    def predict(self, sentence: str):
        questionStarters = [
            "which", "wont", "cant", "isnt", "arent", "is", "do", "does",
            "will", "can"
        ]
        questionElements = [
            "who", "what", "when", "where", "why", "how", "sup", "?"
        ]

        sentence = sentence.lower()
        sentence = sentence.replace("\'", "")
        sentence = self.padCharacter('?', sentence)
        splitWords = sentence.split()

        if any(word == splitWords[0] for word in questionStarters) or any(
                word in splitWords for word in questionElements):
            return True
        else:
            return False

def response_chain(message, convo_context, info_context):

  # Determine what kind of message this is.
  result = pg.Completion.create(
      model="WizardCoder",
      prompt=category_prompt.format(query=message)
  )

  # configure our chain
  if "yes" in result['choices'][0]['text']:
    code = "yes"
  else:
    code = "no"
  qIDModel = QuestionID()
  question = qIDModel.predict(message)

  if code == "no" and question:

    # Handle the informational request.
    result = pg.Completion.create(
        model="Nous-Hermes-Llama2-13B",
        prompt=qa_prompt.format(context=info_context, query=message)
    )
    completion = result['choices'][0]['text'].split('#')[0].strip()

  elif code == "yes":

    # Handle the code generation request.
    result = pg.Completion.create(
        model="WizardCoder",
        prompt=code_prompt.format(query=message),
        max_tokens=500
    )
    completion = result['choices'][0]['text']

  else:

    # Handle the chat message.
    result = pg.Completion.create(
        model="Nous-Hermes-Llama2-13B",
        prompt=chat_prompt.format(context=convo_context, query=message),
        output={
            "toxicity": True
        }
    )
    completion = result['choices'][0]['text'].split('Human:')[0].strip()

  return code, question, completion

################## Code extracted from augmentation.mdx
info_context = "The Greater Los Angeles and San Francisco Bay areas in California are the nation's second and fifth-most populous urban regions, respectively. Greater Los Angeles has over 18.7 million residents and the San Francisco Bay Area has over 9.6 million residents. Los Angeles is state's most populous city and the nation's second-most populous city. San Francisco is the second-most densely populated major city in the country. Los Angeles County is the country's most populous county, and San Bernardino County is the nation's largest county by area. Sacramento is the state's capital."

convo_context = """Human: Hello, how are you?
AI: I'm good, what can I help you with?
Human: What is the captital of California?
AI: Sacramento
Human: Thanks!
AI: You are welcome!"""

message = "Which city in California has the highest population?"
#message = "I'm really enjoying this conversation."
#message = "Generate some python code that gets the current weather in the bay area."

code, question, completion = response_chain(message, convo_context, info_context)
print("CODE GEN REQUESTED:", code)
print("QUESTION:", question)
print("")
print("RESPONSE:", completion)

################## Code extracted from augmentation.mdx
template = """### Instruction:
Read the context below and respond with an answer to the question. If the question cannot be answered based on the context alone or the context does not explicitly say the answer to the question, write "Sorry I had trouble answering this question, based on the information I found."

### Input:
Context: {context}

Question: {question}

### Response:
"""

prompt = PromptTemplate(
    input_variables=["context", "question"],
    template=template,
)

context = "Domino's gift cards are great for any person and any occasion. There are a number of different options to choose from. Each comes with a personalized card carrier and is delivered via US Mail."

question = "How are gift cards delivered?"

myprompt = prompt.format(context=context, question=question)

result = pg.Completion.create(
    model="Nous-Hermes-Llama2-13B",
    prompt=myprompt
)
result['choices'][0]['text'].split('#')[0].strip()

################## Code extracted from augmentation.mdx
# Let's get the html off of a website.
fp = urllib.request.urlopen("https://docs.kernel.org/process/submitting-patches.html")
mybytes = fp.read()
html = mybytes.decode("utf8")
fp.close()

# And convert it to text.
h = html2text.HTML2Text()
h.ignore_links = True
text = h.handle(html)

print(text)

################## Code extracted from augmentation.mdx
# Clean things up just a bit.
text = text.split("### This Page")[1]
text = text.split("## References")[0]

# Chunk the text into smaller pieces for injection into LLM prompts.
text_splitter = CharacterTextSplitter(chunk_size=700, chunk_overlap=50)
docs = text_splitter.split_text(text)

# Let's checkout some of the chunks!
for i in range(0, 3):
  print("Chunk", str(i+1))
  print("----------------------------")
  print(docs[i])
  print("")

################## Code extracted from augmentation.mdx
# Let's take care of some of the formatting so it doesn't conflict with our
# typical prompt template structure
docs = [x.replace('#', '-') for x in docs]


# Now we need to embed these documents and put them into a "vector store" or
# "vector db" that we will use for semantic search and retrieval.

# Embeddings setup
name="all-MiniLM-L12-v2"
model = SentenceTransformer(name)

def embed_batch(batch):
    return [model.encode(sentence) for sentence in batch]

def embed(sentence):
    return model.encode(sentence)

# LanceDB setup
os.mkdir(".lancedb")
uri = ".lancedb"
db = lancedb.connect(uri)

# Create a dataframe with the chunk ids and chunks
metadata = []
for i in range(len(docs)):
    metadata.append([
        i,
        docs[i]
    ])
doc_df = pd.DataFrame(metadata, columns=["chunk", "text"])

# Embed the documents
data = with_embeddings(embed_batch, doc_df)

# Create the DB table and add the records.
db.create_table("linux", data=data)
table = db.open_table("linux")
table.add(data=data)

################## Code extracted from augmentation.mdx
# Let's try to match a query to one of our documents.
message = "How many problems should be solved per patch?"
results = table.search(embed(message)).limit(5).to_df()
results.head()

################## Code extracted from augmentation.mdx
# Now let's augment our Q&A prompt with this external knowledge on-the-fly!!!
template = """### Instruction:
Read the below input context and respond with a short answer to the given question. Use only the information in the below input to answer the question. If you cannot answer the question, respond with "Sorry, I can't find an answer, but you might try looking in the following resource."

### Input:
Context: {context}

Question: {question}

### Response:
"""
qa_prompt = PromptTemplate(
    input_variables=["context", "question"],
    template=template,
)

def rag_answer(message):

  # Search the for relevant context
  results = table.search(embed(message)).limit(5).to_df()
  results.sort_values(by=['_distance'], inplace=True, ascending=True)
  doc_use = results['text'].values[0]

  # Augment the prompt with the context
  prompt = qa_prompt.format(context=doc_use, question=message)

  # Get a response
  result = pg.Completion.create(
      model="Nous-Hermes-Llama2-13B",
      prompt=prompt
  )

  return result['choices'][0]['text']

