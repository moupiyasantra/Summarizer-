# 📜 Summarizer

A friendly, privacy-first app that summarizes poems and prose and answers questions about them, using an open-source LLM and Retrieval-Augmented Generation (RAG). Everything runs locally: no API keys, no cloud, no usage fees.

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![Streamlit](https://img.shields.io/badge/UI-Streamlit-FF4B4B)
![Ollama](https://img.shields.io/badge/LLM-Ollama%20%7C%20Llama%203.2-black)
![License](https://img.shields.io/badge/license-MIT-green)

## ✨ Features

- **Flexible input:** paste text or upload `.txt`, `.md` or `.pdf` files.
- **Five summary styles:** short summary, detailed summary, themes and tone, literary devices, and "explain like I'm 12".
- **Chat with your text (RAG):** ask questions and get answers grounded in the actual passages, with the sources shown in a **Passages used** expander.
- **Built for poetry and prose:** poems are chunked by stanza and prose by paragraph, so retrieved passages stay meaningful.
- **Long text support:** texts over ~2,000 words are condensed section by section before the final summary (map-reduce).
- **Private by design:** your text never leaves your machine.
- **Friendly UI:** sample poem button, live word count, streamed responses and downloadable summaries.

## 🧠 How it works

```
Your text ──► Chunk (stanza / paragraph) ──► Embed (MiniLM) ──► In-memory vector store
                                                                        │
Your question ──► Embed ──► Cosine similarity ──► Top 4 passages ───────┘
                                                        │
                                  Prompt + passages ──► Llama 3.2 (Ollama) ──► Answer
```

1. **Chunk:** the text is split on blank lines (stanzas or paragraphs).
2. **Embed:** each chunk is turned into a vector with `all-MiniLM-L6-v2`.
3. **Retrieve:** the most similar chunks to your question are found with cosine similarity.
4. **Generate:** the chunks and the question go to the local LLM, which is instructed to answer only from that context.

## 🛠️ Tech stack

| Layer | Tool | Role |
|---|---|---|
| UI | [Streamlit](https://streamlit.io) | Web interface in pure Python |
| LLM | [Ollama](https://ollama.com) + Llama 3.2 | Local open-source model for summaries and answers |
| Embeddings | [sentence-transformers](https://www.sbert.net) (MiniLM) | Converts text to vectors; falls back to keyword matching if unavailable |
| Vector search | NumPy | In-memory cosine similarity |
| File parsing | pypdf | Reads uploaded PDFs |

## 🚀 Getting started

### Prerequisites
- Python 3.10+
- [Ollama](https://ollama.com/download) installed and running

### Install and run

```bash
# 1. Pull a model
ollama pull llama3.2

# 2. Clone the repo and install dependencies
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt

# 3. Launch
python -m streamlit run app.py
```

The first run downloads the embedding model (about 90 MB), so the first summary is slower.

## ⚙️ Configuration

| Setting | Where | Default |
|---|---|---|
| LLM model | `LLM_MODEL` env var or `DEFAULT_MODEL` in `rag.py` | `llama3.2` |
| Passages sent to the LLM | `TOP_K` in `rag.py` | `4` |
| Ollama address | `OLLAMA_URL` env var | `http://127.0.0.1:11434` |
| Theme | `.streamlit/config.toml` | purple, serif |

Example: `export LLM_MODEL=mistral` (Windows: `set LLM_MODEL=mistral`).

## 📁 Project structure

```
├── app.py                  # Streamlit interface
├── rag.py                  # Chunking, embeddings, vector search, LLM client, prompts
├── requirements.txt
└── .streamlit/config.toml  # Theme
```

## 🩹 Troubleshooting

- **"Can't reach Ollama":** open the Ollama app (or run `ollama serve`).
- **"Model isn't installed":** run `ollama pull llama3.2` (or whichever model you set).
- **`streamlit` not recognized (Windows):** use `python -m streamlit run app.py`.
- **Weaker answers than expected:** if `sentence-transformers` can't load, the app uses a keyword fallback. Reinstall it for better semantic search.

## 🗺️ Roadmap

- Persistent vector store (Chroma or FAISS)
- Multiple documents per session
- OCR for scanned PDFs
- Export summaries to PDF or Word

## 📄 License

MIT. Add a `LICENSE` file to your repo to match.
