           🤖 Assessment 2: Mysoft Heaven AI Chatbot (RAG)
## Project Overview
A Retrieval-Augmented Generation (RAG) based chatbot that answers questions strictly using Mysoft Heaven’s company profile and documents.

## Tech Stack
LLM: Groq (Llama-3.3-70b-versatile)

Vector Database: FAISS (Local)

Embeddings: HuggingFace (all-MiniLM-L6-v2)

Framework: LangChain (LCEL)

UI: Streamlit

📘 Mandatory Explanations
1. Document Chunking Strategy
I used the RecursiveCharacterTextSplitter with the following parameters:

Chunk Size: 700 characters

Chunk Overlap: 50 characters Reason: This size ensures that each chunk contains enough context for the LLM to understand the information without exceeding the model's window. The 50-character overlap prevents information loss at the boundaries of the chunks.

2. Embedding Model Choice
I chose the HuggingFace all-MiniLM-L6-v2 model. Reason:

Cost-Effective: It runs locally on the machine, meaning zero API costs compared to OpenAI.

Speed: It is highly optimized for performance and provides fast vector generation.

Accuracy: It provides a great balance between semantic understanding and resource consumption.

3. Handling Irrelevant or Unsupported Queries
To prevent hallucinations and out-of-scope answers:

Prompt Engineering: The system prompt explicitly instructs the LLM: "Use the following pieces of context to answer the question. If you don't know the answer, say you don't know. Answer only based on the company information."

Constraint: If a user asks something unrelated (e.g., "How to cook pasta?"), the bot identifies that the context is missing and declines to answer.

4. Future Scalability for Multiple Companies
The current architecture can be scaled to support multiple companies by:

Metadata Filtering: Adding a company_id to each chunk in the vector store and filtering searches based on that ID.

Multi-Indexing: Creating a separate FAISS index for each company and loading the specific index based on the user's environment or login.

🚀 How to Run

Install dependencies: pip install -r requirements.txt

## activate:

venv\Scripts\activate

Take GROQ_API_KEY="your key"

Prepare data: Place the PDF in the data/ folder and run python ingest.py.

Launch Chatbot: streamlit run app.py