"""Streamlit front end. Run: streamlit run app/streamlit_app.py"""
import os

import streamlit as st

from cfpb_rag.agent.pipeline import build_pipeline
from cfpb_rag.retrieve.retriever import Filters

st.set_page_config(page_title="CFPB complaints assistant", layout="wide")


@st.cache_resource
def pipeline():
    return build_pipeline(llm=os.environ.get("CFPB_RAG_LLM", "auto"))


try:
    pipe = pipeline()
except FileNotFoundError:
    st.error("No index found. Run `make data index` first (see README).")
    st.stop()

st.title("CFPB consumer complaints assistant")
st.caption(
    "Answers come only from the indexed complaint narratives, with citations. Complaints are "
    "unverified consumer statements, not findings about any company."
)
if pipe.llm.name == "stub":
    st.warning("No ANTHROPIC_API_KEY set: using the extractive test stub, not a language model.")

with st.sidebar:
    st.header("Filters")
    pg = st.selectbox("Product", [""] + sorted(pipe.known["product_group"]))
    company = st.selectbox("Company", [""] + sorted(pipe.known["company"]))
    state = st.selectbox("State", [""] + sorted(pipe.known["state"]))
    mode = st.radio("Retrieval", ["hybrid", "dense", "bm25"])
    k = st.slider("Sources (k)", 1, 10, 5)

question = st.text_input("Ask a question", placeholder="Which issue is most common for Mortgage complaints?")
if question:
    f = Filters(product_group=pg or None, company=company or None, state=state or None)
    out = pipe.ask(question, filters=None if f.is_empty() else f, k=k, mode=mode)
    st.subheader("Answer")
    st.markdown(out["answer"])
    cols = st.columns(3)
    cols[0].metric("Route", "SQL aggregate" if out["route"] == "sql" else "Retrieval")
    cols[1].metric("Citations", len(out["citations"]))
    cols[2].metric("Latency", f"{out['latency_ms']} ms")
    if out["abstained"]:
        st.info(f"Abstained: {out['abstain_reason']}")
    if out.get("sql"):
        with st.expander("SQL"):
            st.code(out["sql"], language="sql")
            st.write("params:", out["sql_params"])
    if out["sources"]:
        with st.expander(f"Sources ({len(out['sources'])})", expanded=not out["abstained"]):
            for s in out["sources"]:
                cited = "✅ cited" if s["complaint_id"] in out["citations"] else ""
                m = s["meta"]
                st.markdown(f"**{s['complaint_id']}** · {m['product_group']} · {m['issue']} · "
                            f"{m['company']} · {m['date']} {cited}")
                st.caption(s["snippet"])
