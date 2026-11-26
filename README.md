**Summarizer**


Summarize poems and prose, and ask questions about them, using open-source models that run locally.

Tech stack : 


UI: Streamlit


LLM: open-source model served locally by Ollama (default: llama3.2)


Embeddings: all-MiniLM-L6-v2 (sentence-transformers), with a keyword fallback if unavailable


RAG: stanza/paragraph-aware chunking -> embeddings -> cosine search -> grounded answers


Files: pypdf for PDF uploads


Setup :

bash


ollama pull llama3.2


pip install -r requirements.txt


python -m streamlit run app.py


Configuration :


Change the model: set LLM_MODEL=mistral (Windows) / export LLM_MODEL=mistral, or edit DEFAULT_MODEL in rag.py. 


TOP_K in rag.py sets how many passages are sent to the LLM when answering questions.
