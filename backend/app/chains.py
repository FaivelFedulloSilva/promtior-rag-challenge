# rag_chain.py
from __future__ import annotations

import os
from typing import List

from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate

# Must match ingest.py
PERSIST_DIR = "./chroma_db"
COLLECTION_NAME = "promtior"

OPENAI_EMBED_MODEL = os.getenv("OPENAI_MODEL_EMBEDDINGS", "text-embedding-3-small")
OPENAI_CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")  # buen default costo/latencia

K = 10  # top-k retrieval

def format_docs(docs: List[Document]) -> str:
    # Context compacto y legible (y mantiene URL para cita interna)
    parts = []
    for d in docs:
        url = d.metadata.get("url", "unknown")
        parts.append(f"[SOURCE] {url}\n{d.page_content}")
    return "\n\n---\n\n".join(parts)

def build_vectorstore() -> Chroma:
    embeddings = OpenAIEmbeddings(model=OPENAI_EMBED_MODEL)
    return Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=PERSIST_DIR,
        embedding_function=embeddings,
    )

def build_chain():
    vs = build_vectorstore()
    retriever = vs.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 10, "fetch_k": 40, "lambda_mult": 0.5}
    )

    prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are a helpful assistant for Promtior.\n"
     "Answer ONLY using the provided context.\n"
     "If the answer is not explicitly stated in the context, say:\n"
     "\"I don't know based on the provided context.\"\n\n"
     "When you provide an answer, include a section called 'Sources' "
     "listing the URLs from the context you used.\n\n"
     "Context:\n{context}"
    ),
    ("human", "{question}")
])

    llm = ChatOpenAI(model=OPENAI_CHAT_MODEL, temperature=0)

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain
