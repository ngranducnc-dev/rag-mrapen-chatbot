import streamlit as st
import os
from typing import TypedDict, List, Optional
from sklearn.metrics.pairwise import cosine_similarity

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langgraph.graph import StateGraph, END

# --- 1. SETUP HALAMAN ---
st.set_page_config(page_title="QA Mrapen", page_icon="🔥")
st.title("🔥 Sistem Tanya Jawab Pariwisata Cerdas")
st.subheader("Api Abadi Mrapen - Kabupaten Grobogan")

# --- 2. SETUP API KEY ---
# API Key diambil dari Streamlit Secrets, bukan di-hardcode
os.environ['GROQ_API_KEY'] = st.secrets["GROQ_API_KEY"]

# --- 3. LOAD MODEL & FAISS (Gunakan Cache agar cepat) ---
@st.cache_resource
def load_rag_system():
    # Load Embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name='sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2',
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True},
    )
    # Load FAISS
    vectorstore = FAISS.load_local(
        'faiss_mrapen_index', embeddings, allow_dangerous_deserialization=True
    )
    
    # Load LLM
    llm_main = ChatGroq(
        model='llama-3.3-70b-versatile',
        temperature=0.3,
        max_tokens=1024,
    )
    return embeddings, vectorstore, llm_main

embeddings, vectorstore, llm_main = load_rag_system()

# --- 4. LANGGRAPH PIPELINE (Dari Cell 5) ---
KATA_KUNCI_DOMAIN = [
    'mrapen', 'api abadi', 'grobogan', 'godong', 'manggarmas',
    'tiket', 'harga', 'masuk', 'bayar', 'biaya',
    'fasilitas', 'toilet', 'parkir', 'warung', 'makanan', 'minuman',
    'jam', 'buka', 'tutup', 'operasional', 'waktu', 'kunjungan',
    'sejarah', 'sunan kalijaga', 'walisongo', 'legenda',
    'gas', 'metana', 'fenomena', 'alam', 'geologi', 'pvmbg',
    'wisata', 'pariwisata', 'pengunjung', 'lokasi', 'akses',
    'padam', 'menyala',
]

class RAGState(TypedDict):
    pertanyaan: str
    dokumen: Optional[List]
    konteks: Optional[str]
    jawaban: Optional[str]
    skor_konteks: float
    retry_count: int
    relevan: bool

def cek_relevansi(state: RAGState) -> RAGState:
    relevan = any(kw in state['pertanyaan'].lower() for kw in KATA_KUNCI_DOMAIN)
    return {**state, 'relevan': relevan}

def retrieve(state: RAGState) -> RAGState:
    docs = vectorstore.as_retriever(search_kwargs={'k': 5}).invoke(state['pertanyaan'])
    konteks = '\n\n'.join(f"[{d.metadata.get('sumber','?')}] {d.page_content}" for d in docs)
    return {**state, 'dokumen': docs, 'konteks': konteks}

def validasi_konteks(state: RAGState) -> RAGState:
    if not state.get('konteks'): return {**state, 'skor_konteks': 0.0}
    q_emb = embeddings.embed_query(state['pertanyaan'])
    k_emb = embeddings.embed_query(state['konteks'][:500])
    skor = float(cosine_similarity([q_emb], [k_emb])[0][0])
    return {**state, 'skor_konteks': skor}

def retrieve_ulang(state: RAGState) -> RAGState:
    expanded_query = f"Api Abadi Mrapen wisata {state['pertanyaan']}"
    docs = vectorstore.as_retriever(search_kwargs={'k': 7}).invoke(expanded_query)
    konteks = '\n\n'.join(f"[{d.metadata.get('sumber','?')}] {d.page_content}" for d in docs)
    return {**state, 'dokumen': docs, 'konteks': konteks, 'retry_count': state['retry_count'] + 1}

def generate(state: RAGState) -> RAGState:
    sys_prompt = ('Anda adalah asisten wisata cerdas untuk Api Abadi Mrapen. Jawab informatif, '
                  'ramah. Utamakan sumber resmi. Jika tidak ada di konteks, katakan belum tersedia.')
    user_prompt = f"Konteks:\n{state['konteks']}\n\nPertanyaan: {state['pertanyaan']}"
    res = llm_main.invoke([SystemMessage(content=sys_prompt), HumanMessage(content=user_prompt)])
    return {**state, 'jawaban': res.content}

def fallback(state: RAGState) -> RAGState:
    return {**state, 'jawaban': "Maaf, pertanyaan Anda di luar topik wisata Api Abadi Mrapen. 😊"}

def routing_relevansi(state): return 'retrieve' if state['relevan'] else 'fallback'
def routing_validasi(state): return 'generate' if state['skor_konteks'] >= 0.30 or state['retry_count'] >= 1 else 'retrieve_ulang'

graph = StateGraph(RAGState)
graph.add_node('cek_relevansi', cek_relevansi)
graph.add_node('retrieve', retrieve)
graph.add_node('validasi_konteks', validasi_konteks)
graph.add_node('retrieve_ulang', retrieve_ulang)
graph.add_node('generate', generate)
graph.add_node('fallback', fallback)

graph.set_entry_point('cek_relevansi')
graph.add_conditional_edges('cek_relevansi', routing_relevansi, {'retrieve': 'retrieve', 'fallback': 'fallback'})
graph.add_edge('retrieve', 'validasi_konteks')
graph.add_conditional_edges('validasi_konteks', routing_validasi, {'generate': 'generate', 'retrieve_ulang': 'retrieve_ulang'})
graph.add_edge('retrieve_ulang', 'validasi_konteks')
graph.add_edge('generate', END)
graph.add_edge('fallback', END)
rag_app = graph.compile()

# --- 5. ANTARMUKA STREAMLIT ---
user_input = st.text_input("✏️ Ajukan pertanyaan Anda tentang Mrapen:")
if st.button("🔍 Cari Jawaban"):
    if user_input:
        with st.spinner("Mencari jawaban terbaik dari ulasan pengunjung..."):
            inisial_state = {'pertanyaan': user_input, 'dokumen': None, 'konteks': None, 'jawaban': None, 'skor_konteks': 0.0, 'retry_count': 0, 'relevan': False}
            hasil = rag_app.invoke(inisial_state)
            
            st.success("Selesai!")
            st.write(hasil['jawaban'])
            
            # Tampilkan metrik (berguna untuk UAT)
            with st.expander("Lihat Detail Konteks & Skor"):
                st.info(f"Skor Cosine Similarity: {hasil['skor_konteks']:.3f}")
                if hasil['dokumen']:
                    st.write("Dokumen Referensi:")
                    for i, doc in enumerate(hasil['dokumen'], 1):
                        st.caption(f"[{doc.metadata.get('sumber', 'Unknown')}] {doc.page_content[:150]}...")
    else:
        st.warning("Silakan masukkan pertanyaan terlebih dahulu.")