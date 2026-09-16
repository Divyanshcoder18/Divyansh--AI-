import os

from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer
from groq import Groq


# ============================================================
# PART 1 — LOAD ENVIRONMENT VARIABLES
# ============================================================

# .env file se API keys load kar rahe hain.
#
# Tumhari .env file mein ye values honi chahiye:
#
# QDRANT_URL=your_qdrant_url
# QDRANT_API_KEY=your_qdrant_api_key
# GROQ_API_KEY=your_groq_api_key
#
# API keys ko directly code mein mat likhna.
# ============================================================

load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")


# API keys check kar rahe hain
if not QDRANT_URL:
    raise ValueError("QDRANT_URL nahi mila bhai")

if not QDRANT_API_KEY:
    raise ValueError("QDRANT_API_KEY nahi mila bhai")

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY nahi mila bhai")


# ============================================================
# PART 2 — CONNECT TO QDRANT
# ============================================================

# Qdrant Cloud ke saath connection create kar rahe hain.
#
# QDRANT_URL:
#   Qdrant Cloud ka database URL
#
# QDRANT_API_KEY:
#   Qdrant ko authenticate karne ke liye
# ============================================================

client = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY
)

print("Connected to Qdrant Cloud!")


# ============================================================
# PART 3 — CREATE QDRANT COLLECTION
# ============================================================

# Collection ko tum roughly SQL table ki tarah samajh sakte ho.
#
# Is collection ke andar hum:
#
# vector + original text
#
# store karenge.
# ============================================================

COLLECTION_NAME = "knowledge"


# all-MiniLM-L6-v2 384-dimensional embeddings generate karta hai.
#
# Isliye Qdrant ko bhi size=384 batana zaroori hai.
# ============================================================

EMBEDDING_SIZE = 384


# Agar collection pehle se exist karti hai,
# to learning/demo ke liye usko delete kar rahe hain.
#
# IMPORTANT:
# Production application mein normally har baar
# collection delete nahi karte.
# ============================================================

if client.collection_exists(COLLECTION_NAME):

    print(
        f"Deleting existing collection: {COLLECTION_NAME}"
    )

    client.delete_collection(COLLECTION_NAME)


# Fresh collection create kar rahe hain.
#
# Distance.COSINE ka matlab:
# vectors ke beech cosine similarity use hogi.
# ============================================================

client.create_collection(

    collection_name=COLLECTION_NAME,

    vectors_config=VectorParams(

        size=EMBEDDING_SIZE,

        distance=Distance.COSINE,
    ),
)


print(f"Created collection: {COLLECTION_NAME}")
print(f"Vector size: {EMBEDDING_SIZE}")
print("Distance: COSINE")


# ============================================================
# PART 4 — LOAD OUR KNOWLEDGE
# ============================================================

# knowledge.txt file se information read karenge.
#
# Example knowledge.txt:
#
# Employees receive 24 days of paid leave per year.
# Employees work from the office on Tuesday, Wednesday and Thursday.
# Employees receive Rs 3000 per month for gym reimbursement.
# Employees can claim Rs 2000 per month for home internet.
# Employees have a 90 day notice period.
#
# Har non-empty line ko ek document maana jayega.
# ============================================================

with open(
    "knowledge.txt",
    "r",
    encoding="utf-8"
) as f:

    documents = [

        line.strip()

        for line in f

        if line.strip()
    ]


print(f"Loaded {len(documents)} documents")


# ============================================================
# PART 5 — CREATE EMBEDDINGS
# ============================================================

print("Loading embedding model...")


# ------------------------------------------------------------
# EMBEDDING MODEL
# ------------------------------------------------------------
#
# Yeh Groq model nahi hai.
#
# Iska kaam:
#
# Text
#   ↓
# Vector / Embedding
#
# all-MiniLM-L6-v2
#   ↓
# 384 numbers
#
# Hum isi model ka use documents aur user query
# dono ke embeddings ke liye karenge.
# ============================================================

embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)


print("Embedding model ready!")


# Saare documents ko embeddings mein convert kar rahe hain.
#
# Example:
#
# Document:
# "Employees receive 24 days..."
#
#       ↓
#
# [0.12, -0.43, 0.21, ...]
#
# Total 384 values.
# ============================================================

embeddings = embedding_model.encode(documents)


print(f"Generated {len(embeddings)} embeddings")

print(
    f"Embedding size: {len(embeddings[0])}"
)


# ============================================================
# PART 6 — CREATE QDRANT POINTS
# ============================================================

# Qdrant mein ek record ko Point kaha jata hai.
#
# Ek Point ke andar:
#
# ID
# Vector
# Payload
#
# hoga.
# ============================================================

points = []


for i, embedding in enumerate(embeddings):

    point = PointStruct(

        # Unique ID
        #
        # i = 0 → ID = 1
        # i = 1 → ID = 2
        # etc.
        id=i + 1,


        # NumPy array ko Python list mein convert kar rahe hain.
        vector=embedding.tolist(),


        # Payload mein original document store kar rahe hain.
        #
        # Vector search ke baad hume actual text chahiye.
        # ====================================================

        payload={
            "text": documents[i]
        }
    )


    points.append(point)


# ============================================================
# PART 7 — UPLOAD TO QDRANT
# ============================================================

# Ab saare points Qdrant Cloud mein upload kar rahe hain.
#
# upsert =
# insert if new
# update if same ID already exists
# ============================================================

client.upsert(

    collection_name=COLLECTION_NAME,

    points=points
)


print(
    f"Uploaded {len(points)} documents to Qdrant!"
)


# ============================================================
# PART 8 — SEARCH QDRANT
# ============================================================

def search(query, top_k=3):

    # --------------------------------------------------------
    # STEP 1
    # User question ko embedding mein convert karo.
    # --------------------------------------------------------
    #
    # IMPORTANT:
    #
    # Documents jis embedding model se convert hue,
    # query ko bhi SAME model se convert karna chahiye.
    # --------------------------------------------------------

    query_vector = embedding_model.encode(
        query
    ).tolist()


    # --------------------------------------------------------
    # STEP 2
    # Qdrant mein similarity search karo.
    # --------------------------------------------------------
    #
    # Qdrant query vector ko stored vectors ke saath compare
    # karega using COSINE similarity.
    #
    # top_k=3:
    # Sabse relevant 3 documents return karo.
    #
    # with_payload=True:
    # Original text bhi return karo.
    # --------------------------------------------------------

    results = client.query_points(

        collection_name=COLLECTION_NAME,

        query=query_vector,

        limit=top_k,

        with_payload=True,

    ).points


    return results


# ============================================================
# PART 9 — TEST SEARCH
# ============================================================

query = "How many vacation days do I get?"


# Qdrant se top 3 relevant documents retrieve karenge.
results = search(
    query,
    top_k=3
)


print("\nSearch results:")


for result in results:

    print(
        f"Score: {result.score:.3f}"
    )

    print(
        result.payload["text"]
    )

    print()


# ============================================================
# PART 10 — CONNECT TO GROQ
# ============================================================

# Ab Qdrant ka kaam khatam.
#
# Qdrant:
#   Relevant information retrieve karta hai.
#
# Groq LLM:
#   Retrieved information se answer generate karta hai.
# ============================================================

groq_client = Groq(
    api_key=GROQ_API_KEY
)


# ============================================================
# PART 11 — ASK THE LLM
# ============================================================

def ask_llm(question, context):

    # --------------------------------------------------------
    # LLM ko question + retrieved context denge.
    # --------------------------------------------------------
    #
    # Hum LLM ko bol rahe hain:
    #
    # "Sirf context ke basis par answer karo."
    #
    # Agar context mein answer nahi hai,
    # to "I don't know..." bolo.
    # --------------------------------------------------------

    prompt = f"""
Answer the question using only the information provided below.

Context:
{context}

Question:
{question}

If the answer is not present in the context, say:
"I don't know based on the provided information."
"""


    # --------------------------------------------------------
    # Groq LLM ko request bhej rahe hain.
    # --------------------------------------------------------
    #
    # IMPORTANT:
    #
    # Tumhare API key ke saath jo model work karta hai:
    #
    # openai/gpt-oss-120b
    #
    # Isliye yahan old llama model use nahi kar rahe.
    # --------------------------------------------------------

    response = groq_client.chat.completions.create(

        model="openai/gpt-oss-120b",

        messages=[

            {
                "role": "user",
                "content": prompt
            }

        ],

        temperature=0
    )


    # LLM ka actual answer extract kar rahe hain.
    return response.choices[0].message.content


# ============================================================
# PART 12 — COMPLETE RAG PIPELINE
# ============================================================

# User ka final question
question = "How many vacation days do I get?"


# ------------------------------------------------------------
# STEP 1 — RETRIEVAL
# ------------------------------------------------------------
#
# Question → Embedding → Qdrant → Top 3 documents
# ------------------------------------------------------------

results = search(
    question,
    top_k=3
)


# ------------------------------------------------------------
# STEP 2 — EXTRACT CONTEXT
# ------------------------------------------------------------
#
# Qdrant results ke payload se original text nikal rahe hain.
#
# Agar 3 results aaye:
#
# Document 1
# Document 2
# Document 3
#
# to unko ek single context string mein combine karenge.
# ------------------------------------------------------------

context = "\n".join(

    result.payload["text"]

    for result in results
)


# ------------------------------------------------------------
# STEP 3 — GENERATION
# ------------------------------------------------------------
#
# Question + Retrieved Context
#              ↓
#          Groq LLM
#              ↓
#         Final Answer
# ------------------------------------------------------------

answer = ask_llm(
    question,
    context
)


# ============================================================
# FINAL ANSWER
# ============================================================

print("\nFinal Answer:")

print(answer)