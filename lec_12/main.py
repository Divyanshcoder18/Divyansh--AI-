# ============================================================
# LECTURE 12 - SIMPLE RAG SYSTEM
# ============================================================
#
# RAG ka full form hai:
# Retrieval-Augmented Generation
#
# Isme 2 main kaam hote hain:
#
# 1. RETRIEVAL:
#    User ke question ke liye sabse relevant document find karna.
#
# 2. GENERATION:
#    Retrieved document ko LLM ko dena aur usse answer generate karwana.
#
# Flow:
#
# Documents
#     ↓
# Embeddings
#     ↓
# User Question
#     ↓
# Question ka Embedding
#     ↓
# Similarity Check
#     ↓
# Best Document
#     ↓
# Groq LLM
#     ↓
# Final Answer
#
# ============================================================


# ------------------------------------------------------------
# IMPORTS
# ------------------------------------------------------------

import os

# .env file se API key read karne ke liye
from dotenv import load_dotenv

# Groq LLM ko use karne ke liye
from groq import Groq

# Mathematical operations ke liye
# Hum cosine similarity calculate karenge
import numpy as np

# Text ko embeddings/vectors mein convert karne ke liye
from sentence_transformers import SentenceTransformer


# ============================================================
# 1. EMBEDDING MODEL LOAD KARNA
# ============================================================

# Yeh model text ko numerical vector mein convert karega.
#
# all-MiniLM-L6-v2 ek embedding model hai.
#
# Yeh answer generate nahi karta.
# Iska kaam sirf text -> vector karna hai.
#
# Is model ka embedding size = 384
#
# Example:
#
# "I like football"
#       ↓
# [0.12, -0.45, 0.67, .........]
#       ↓
#    384 numbers
#
# ============================================================

model = SentenceTransformer("all-MiniLM-L6-v2")


# ============================================================
# 2. GROQ API KEY LOAD KARNA
# ============================================================

# .env file ko load karenge
#
# Example .env:
#
# GROQ_API_KEY=your_api_key
#
# API key ko code ke andar directly nahi likhna chahiye.
# ============================================================

load_dotenv()


# Environment variable se API key nikal rahe hain
my_api_key = os.getenv("GROQ_API_KEY")


# Agar API key nahi mili to program stop ho jayega
if not my_api_key:
    raise ValueError("API key kaha hai bhai")


# ============================================================
# 3. GROQ CLIENT BANANA
# ============================================================

# Ab Groq client create kar rahe hain.
#
# Isi client ke through hum Groq ke LLM ko request bhejenge.
# ============================================================

client = Groq(api_key=my_api_key)


# Yeh hamara ACTUAL LLM hai.
#
# Dhyaan rakho:
#
# all-MiniLM-L6-v2
#       ↓
# Embedding banata hai
#
# openai/gpt-oss-120b
#       ↓
# Final answer generate karta hai
#
# Dono alag models hain.
# ============================================================

groqmodel = "openai/gpt-oss-120b"


# ============================================================
# 4. HAMARA KNOWLEDGE BASE / DOCUMENTS
# ============================================================

# Yeh hamare company ke documents hain.
#
# Real project mein yeh information PDF, DOCX, website,
# database etc. se aa sakti hai.
#
# Abhi learning ke liye hum directly strings use kar rahe hain.
# ============================================================

documents = [

    "Employees receive 24 days of paid leave per year.",

    "Employees work from the office on Tuesday, Wednesday and Thursday. "
    "Monday and Friday are optional work-from-home days.",

    "Employees receive Rs 3000 per month for gym reimbursement.",

    "Employees can claim Rs 2000 per month for home internet.",

    "Employees have a 90 day notice period."
]


# ============================================================
# 5. DOCUMENTS KO EMBEDDINGS MEIN CONVERT KARNA
# ============================================================

# Ab hum har document ko vector mein convert karenge.
#
# Example:
#
# Document:
# "Employees receive 24 days of paid leave per year."
#
#        ↓
#
# [0.12, -0.21, 0.43, ....]
#
#        ↓
#
# 384 numbers
#
# Yeh sab vectors document_embeddings mein store honge.
# ============================================================

document_embeddings = model.encode(documents)


# Shape check kar rahe hain.
#
# Expected:
#
# (5, 384)
#
# 5  = total documents
# 384 = har document ke vector mein numbers
# ============================================================

print("Embedding shape:", document_embeddings.shape)


# ============================================================
# 6. COSINE SIMILARITY FUNCTION
# ============================================================

# Is function ka kaam hai:
#
# Do vectors kitne similar hain?
#
# Example:
#
# Question vector
#       VS
# Document vector
#
# Agar meaning similar hai:
#
# similarity -> high
#
# Agar meaning different hai:
#
# similarity -> low
# ============================================================

def cosine_similarity(a, b):

    return np.dot(a, b) / (
        np.linalg.norm(a) * np.linalg.norm(b)
    )


# ============================================================
# 7. RETRIEVAL FUNCTION
# ============================================================

# Is function ko user ke question ka embedding milega.
#
# Phir yeh question ko har document ke saath compare karega.
#
# Jis document ka similarity score sabse zyada hoga,
# woh sabse relevant document maana jayega.
# ============================================================

def retrieve(qembedding):

    # Yahan hum similarity scores store karenge.
    #
    # Example:
    #
    # [
    #   (0.82, "Employees receive 24 days..."),
    #   (0.21, "Employees work from office..."),
    #   ...
    # ]
    # ========================================================

    scores = []


    # Har document embedding ke through loop karenge
    for i, document in enumerate(document_embeddings):

        # Question embedding aur document embedding
        # ki similarity calculate kar rahe hain.
        score = cosine_similarity(
            qembedding,
            document
        )


        # Score ke saath original document bhi store karenge.
        #
        # i = document ka index
        # documents[i] = original text
        #
        scores.append(
            (score, documents[i])
        )


    # Scores ko descending order mein sort karenge.
    #
    # Sabse bada similarity score sabse upar aayega.
    #
    # Example:
    #
    # Before:
    # 0.21
    # 0.82
    # 0.18
    #
    # After:
    # 0.82
    # 0.21
    # 0.18
    # ========================================================

    scores.sort(reverse=True)


    # scores[0] = highest similarity wala document
    #
    # Return hoga:
    #
    # (highest_score, best_document)
    # ========================================================

    return scores[0]


# ============================================================
# 8. LLM KO QUESTION + CONTEXT DENA
# ============================================================

# Ab retrieval ho chuka hai.
#
# Is function ka kaam hai:
#
# Question + Relevant Context
#            ↓
#          Groq LLM
#            ↓
#        Final Answer
# ============================================================

def ask_llm(question, context):


    # System prompt mein hum LLM ko rules de rahe hain.
    #
    # Sabse important rule:
    #
    # "Sirf provided context se answer dena."
    #
    # Isse hallucination reduce karne ki koshish hoti hai.
    # ========================================================

    sys_prompt = f"""
Answer in one line only.
Answer only based on this context.
Do not hallucinate.

Context:
{context}
"""


    # System message
    system_message = {
        "role": "system",
        "content": sys_prompt
    }


    # User ka actual question
    message = {
        "role": "user",
        "content": question
    }


    # Dono messages ko list mein rakh rahe hain.
    messages = [
        system_message,
        message
    ]


    # Ab Groq LLM ko request bhej rahe hain.
    response = client.chat.completions.create(

        model=groqmodel,

        messages=messages,

        # Temperature 0 ka matlab:
        # response ko zyada deterministic rakhna.
        temperature=0
    )


    # LLM ke response se actual text nikal rahe hain.
    answer = response.choices[0].message.content


    # Answer return kar rahe hain.
    return answer


# ============================================================
# 9. USER QUERY
# ============================================================

# User ka question.
query = "How much vacation do I get?"


# ============================================================
# 10. QUERY KA EMBEDDING BANANA
# ============================================================

# User ke question ko bhi same embedding model se
# vector mein convert karna zaroori hai.
#
# IMPORTANT:
#
# Documents jis model se embedding mein convert hue,
# question ko bhi SAME model se convert karna chahiye.
#
# Question:
# "How much vacation do I get?"
#
#        ↓
#
# all-MiniLM-L6-v2
#
#        ↓
#
# 384-dimensional vector
# ============================================================

qembedding = model.encode(query)


# ============================================================
# 11. MOST RELEVANT DOCUMENT RETRIEVE KARNA
# ============================================================

# retrieve() question embedding ko documents ke
# embeddings ke saath compare karega.
#
# Return:
#
# score
# context
#
# ============================================================

score, context = retrieve(qembedding)


# Retrieval ka result dekhne ke liye print kar rahe hain.
print("\nSimilarity score:", score)

print("Retrieved context:", context)


# ============================================================
# 12. RETRIEVED CONTEXT + QUESTION → LLM
# ============================================================

# Ab relevant document ko LLM ko de rahe hain.
#
# Question:
# "How much vacation do I get?"
#
# Context:
# "Employees receive 24 days of paid leave per year."
#
# LLM:
#        ↓
# "You receive 24 days of paid leave per year."
# ============================================================

answer = ask_llm(query, context)


# ============================================================
# 13. FINAL ANSWER
# ============================================================

print("\nAnswer:", answer)

