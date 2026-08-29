import os
import json
from dotenv import load_dotenv
from groq import Groq
from pydantic import BaseModel

# Load environment variables
load_dotenv()

# Get API key
my_api_key = os.getenv("GROQ_API_KEY")

if not my_api_key:
    raise ValueError("API key kaha hai bhai")

# Create Groq client
client = Groq(api_key=my_api_key)

# Model
model = "openai/gpt-oss-120b"

# -----------------------------
# Pydantic Schema
# -----------------------------
class Ticket(BaseModel):
    name: str
    email: str
    issue: str

schema = Ticket.model_json_schema()

# Tell the model to return JSON
response_format = {
    "type": "json_object"
}
# -----------------------------
# System Prompt
# -----------------------------

system_prompt = f"""
Extract the information from the customer ticket strictly
according to this schema and return only JSON.

Schema:
{schema}
"""
message_system = {
    "role": "system",
    "content": system_prompt
}

# -----------------------------
# Customer Ticket
# -----------------------------

text = """
Hello, my name is Pratyush.
Yesterday I broke up with my girlfriend Sheetal.
I have an iPhone which is not working at all.
My address is Delhi.
My email is abc@gmail.com.
My contact number is 82134.
"""

prompt = f"""
This is a customer ticket
Please extract the relevant information from this ticket.
Ticket:
{text}
"""
message = {
    "role": "user",
    "content": prompt
}

messages = [message_system, message]

# -----------------------------
# Groq API Call
# -----------------------------

response = client.chat.completions.create(
    model=model,
    messages=messages,
    response_format=response_format
)

# -----------------------------
# Get JSON response
# -----------------------------

answer = response.choices[0].message.content

print("AI Response:")
print(answer)

print("#######################################")

# -------------------------------
# JSON string → Python dictionary
# -------------------------------
data_file = json.loads(answer)
# -----------------------------
# Dictionary → Pydantic object
# -----------------------------
ticket = Ticket(**data_file)
# -----------------------------
# Access structured data
# -----------------------------
print("Name:", ticket.name)
print("Email:", ticket.email)
print("Issue:", ticket.issue)