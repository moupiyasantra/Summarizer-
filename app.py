"""Prose & Poetry Assistant.  Run with:  python -m streamlit run app.py"""
import streamlit as st

import rag

st.set_page_config(page_title="Prose & Poetry Assistant", page_icon="📜", layout="centered")

st.markdown("""
<style>
.hero {padding: 1.6rem 1.8rem; border-radius: 18px; margin-bottom: 1.2rem; color: #fff;
       background: linear-gradient(135deg, #6C4AB6 0%, #9B7EDE 55%, #F2A6C3 100%);}
.hero h1 {margin: 0; font-size: 2rem; color: #fff; padding: 0;}
.hero p  {margin: .4rem 0 0; opacity: .95; font-size: 1.02rem;}
.stTabs [data-baseweb="tab"] {padding: .55rem 1rem; font-size: 1rem;}
div.stButton > button, div.stDownloadButton > button {border-radius: 12px;}
.footer {text-align: center; opacity: .6; font-size: .85rem; margin-top: 2rem;}
</style>
<div class="hero">
  <h1>📜 Prose &amp; Poetry Assistant</h1>
  <p>Share a poem or story and get a friendly summary,
     then ask questions about it.</p>
</div>
""", unsafe_allow_html=True)

SAMPLE = """The river keeps no calendar,
it only learns the shape of stones,
and carries every yesterday
a little further from its home.

I stood upon its patient bank
and wished that I could flow like that,
unhurried by the clocks of men,
forgiving all the road it had."""


def load_sample():
    st.session_state.input_text = SAMPLE
    st.session_state.kind_choice = "poem"


# ---------- state (survives Streamlit reruns) ----------
ss = st.session_state
if "store" not in ss:
    ss.store = rag.VectorStore()
    ss.chat = []
    ss.summary = ""

model = rag.DEFAULT_MODEL
tab1, tab2 = st.tabs(["✍️ Summarize", "💬 Ask about my text"])

# ---------- Tab 1: submit + summarize ----------
with tab1:
    c1, c2 = st.columns([3, 2])
    kind = c1.radio("What are you sharing?", ["poem", "prose / story / essay"],
                    key="kind_choice", horizontal=True)
    c2.write("")
    c2.button("📄 Try a sample poem", on_click=load_sample, use_container_width=True)

    st.text_area("Paste your text", key="input_text", height=220,
                 placeholder="Paste a poem, story or essay here...")
    up = st.file_uploader("...or upload a file (txt, md, pdf)", type=["txt", "md", "pdf"])

    text = ss.input_text
    if up is not None:
        try:
            if up.name.lower().endswith(".pdf"):
                from pypdf import PdfReader
                text = "\n\n".join((p.extract_text() or "") for p in PdfReader(up).pages)
            else:
                text = up.getvalue().decode("utf-8", errors="ignore")
        except Exception as e:
            st.error(f"Couldn't read that file: {e}")
            text = ""
        if not text.strip():
            st.warning("No text found in the file (scanned PDFs need OCR). Try pasting instead.")
    if text.strip():
        st.caption(f"📏 {len(text.split())} words")

    style = st.selectbox("What would you like?", list(rag.SUMMARY_STYLES))
    clicked = st.button("✨ Summarize", type="primary", use_container_width=True)
    out = st.container()

    if clicked and not text.strip():
        out.warning("Please paste or upload some text first "
                    "(click outside the box after pasting).")
    elif clicked:
        ss.chat, ss.summary = [], ""
        with out.container(border=True):
            try:
                with st.spinner("Reading your text..."):
                    ss.store = rag.VectorStore()
                    ss.store.add(rag.chunk_text(text), source="your submitted text")
                st.markdown(f"**{style}**")
                ss.summary = st.write_stream(rag.summarize(text, style, kind, model))
            except rag.LLMError as e:
                st.error(str(e))
            except Exception as e:
                st.error(f"Something went wrong: {e}")
    elif ss.summary:
        with out.container(border=True):
            st.markdown(f"**{style}**")
            st.markdown(ss.summary)

    if ss.summary:
        st.download_button("⬇️ Download summary", ss.summary, "summary.txt")

# ---------- Tab 2: RAG Q&A over the submitted text ----------
with tab2:
    if len(ss.store) == 0:
        st.info("Share a text in the **Summarize** tab first, then come back to chat about it.")
    else:
        st.caption("Ask anything about your text, e.g. *What does the second stanza mean?*")
        for m in ss.chat:
            st.chat_message(m["role"]).write(m["content"])
        if q := st.chat_input("Ask about your text..."):
            st.chat_message("user").write(q)
            with st.chat_message("assistant"):
                try:
                    hits = ss.store.search(q, rag.TOP_K)
                    ans = st.write_stream(rag.rag_answer(q, hits, model, ss.chat))
                    with st.expander("Passages used"):
                        for h in hits:
                            st.markdown(f"> {h['text']}")
                    ss.chat += [{"role": "user", "content": q},
                                {"role": "assistant", "content": ans}]
                except rag.LLMError as e:
                    st.error(str(e))
                except Exception as e:
                    st.error(f"Something went wrong: {e}")

st.markdown('<div class="footer">Powered by open-source models · '
            'Your text stays on your machine</div>', unsafe_allow_html=True)
