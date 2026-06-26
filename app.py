import streamlit as st
import os
from typing import TypedDict, List, Optional
from sklearn.metrics.pairwise import cosine_similarity
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langgraph.graph import StateGraph, END

# --- 1. SETUP HALAMAN & TEMA MODERN ---
# Menggunakan layout wide agar tampilan chat lebih lega
st.set_page_config(page_title="QA Mrapen", page_icon="🔥", layout="wide")

# Link gambar latar belakang (Silakan ganti dengan link foto Api Abadi Mrapen yang Anda inginkan)
GAMBAR_BACKGROUND = "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRSHXXZSc-68J7nhUP3dKVszi5TNO1Ni1vFBg&s"

st.markdown(f"""
    <style>
    .hero-container {{
        /* Menggabungkan warna gelap transparan dan gambar latar */
        background-image: linear-gradient(rgba(0, 0, 0, 0.65), rgba(0, 0, 0, 0.65)), url("{GAMBAR_BACKGROUND}");
        background-size: cover;
        background-position: center;
        padding: 60px 20px;
        border-radius: 15px;
        margin-bottom: 30px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.3);
    }}
    .main-header {{
        font-size: 2.8rem;
        color: #ffffff; /* Diubah jadi putih agar kontras */
        text-align: center;
        font-weight: bold;
        margin-bottom: 10px;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.7);
    }}
    .sub-header {{
        font-size: 1.3rem;
        color: #f1c40f; /* Diubah jadi kuning keemasan */
        text-align: center;
        text-shadow: 1px 1px 3px rgba(0,0,0,0.8);
    }}
    </style>
    
  
""", unsafe_allow_html=True)
st.markdown('<p class="main-header">🔥 Sistem Tanya Jawab Pariwisata Cerdas</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Jelajahi Keajaiban Api Abadi Mrapen - Kabupaten Grobogan</p>', unsafe_allow_html=True)

# --- 2. SIDEBAR INFORMASI SISTEM ---
with st.sidebar:
    st.title("ℹ️ Tentang Sistem")
    st.info(
        "Sistem ini menggunakan **Agentic RAG** (LangGraph) untuk merespons pertanyaan "
        "berdasarkan ulasan pengunjung dan data resmi pengelola wisata."
    )
    st.markdown("---")
    st.markdown("**Peneliti:** Yekti Kuncorojati")
    st.markdown("**NIM:** 2207023")
    st.markdown("**Prodi:** S1 Ilmu Komputer")
    st.markdown("**Institusi:** Universitas An Nuur")
    st.markdown("---")
    # Tombol untuk mereset riwayat percakapan
    if st.button("🗑️ Hapus Riwayat Chat"):
        st.session_state.messages = []
        st.rerun()

# --- 3. KONFIGURASI API KEY ---
if "GROQ_API_KEY" in st.secrets:
    os.environ['GROQ_API_KEY'] = st.secrets["GROQ_API_KEY"]
else:
    st.error("⚠️ API Key belum ditemukan! Masukkan di Settings -> Secrets.")
    st.stop()

# --- 4. LOADING ENGINE (Embeddings, Vector Store, LLM) ---
@st.cache_resource(show_spinner=False)
def load_rag_system():
    embeddings = HuggingFaceEmbeddings(
        model_name='sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2',
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True},
    )
    vectorstore = FAISS.load_local(
        'faiss_mrapen_index', 
        embeddings, 
        allow_dangerous_deserialization=True
    )
    llm_main = ChatGroq(
        model='llama-3.3-70b-versatile',
        temperature=0.3,
        max_tokens=1024,
        groq_api_key=os.environ['GROQ_API_KEY']
    )
    return embeddings, vectorstore, llm_main

with st.spinner("⏳ Menyiapkan asisten AI..."):
    embeddings, vectorstore, llm_main = load_rag_system()

# --- 5. LOGIKA LANGGRAPH ---
KATA_KUNCI_DOMAIN = [
    'mrapen', 'api abadi', 'grobogan', 'tiket', 'jam', 'buka', 'fasilitas', 
    'sejarah', 'wisata', 'lokasi', 'harga', 'toilet', 'parkir', 'warung', 
    'makanan', 'minuman', 'tutup', 'operasional', 'waktu', 'kunjungan', 
    'sunan kalijaga', 'walisongo', 'legenda', 'gas', 'metana', 'fenomena', 
    'alam', 'geologi', 'pvmbg', 'pengunjung', 'akses', 'padam', 'menyala'
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
    konteks = '\n\n'.join(f"[{d.metadata.get('sumber','?')}]: {d.page_content}" for d in docs)
    return {**state, 'dokumen': docs, 'konteks': konteks}

def validasi_konteks(state: RAGState) -> RAGState:
    if not state.get('konteks'): return {**state, 'skor_konteks': 0.0}
    q_emb = embeddings.embed_query(state['pertanyaan'])
    k_emb = embeddings.embed_query(state['konteks'][:500])
    skor = float(cosine_similarity([q_emb], [k_emb])[0][0])
    return {**state, 'skor_konteks': skor}

def generate(state: RAGState) -> RAGState:
    sys_prompt = 'Anda adalah asisten wisata cerdas untuk Api Abadi Mrapen. Jawab dengan natural, ramah, dan informatif.'
    user_prompt = f"Konteks:\n{state['konteks']}\n\nPertanyaan: {state['pertanyaan']}"
    res = llm_main.invoke([SystemMessage(content=sys_prompt), HumanMessage(content=user_prompt)])
    return {**state, 'jawaban': res.content}

graph = StateGraph(RAGState)
graph.add_node('cek_relevansi', cek_relevansi)
graph.add_node('retrieve', retrieve)
graph.add_node('validasi_konteks', validasi_konteks)
graph.add_node('generate', generate)
graph.add_node('fallback', lambda s: {**s, 'jawaban': "Maaf, pertanyaan Anda tampaknya di luar topik wisata Mrapen. Ada yang bisa saya bantu terkait fasilitas, tiket, atau lokasinya? 😊"})

graph.set_entry_point('cek_relevansi')
graph.add_conditional_edges('cek_relevansi', lambda s: 'retrieve' if s['relevan'] else 'fallback', {'retrieve': 'retrieve', 'fallback': 'fallback'})
graph.add_edge('retrieve', 'validasi_konteks')
graph.add_edge('validasi_konteks', 'generate')
graph.add_edge('generate', END)
graph.add_edge('fallback', END)
rag_app = graph.compile()


# --- 6. ANTARMUKA CHAT INTERAKTIF ---

# Inisialisasi memori riwayat chat di session_state
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Halo! Saya adalah asisten virtual Api Abadi Mrapen. Ada yang ingin Anda tanyakan seputar tiket, jam buka, fasilitas, atau sejarah lokasi wisata ini?"}
    ]

# Tampilkan seluruh riwayat percakapan sebelumnya
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        # Tampilkan skor konteks jika ada
        if "score" in msg and msg["score"] > 0:
            with st.expander("📊 Metrik (UAT)"):
                st.caption(f"Skor Cosine Similarity: **{msg['score']:.3f}**")

# Input chat di bagian bawah layar (DI SINI PERBAIKANNYA)
if prompt := st.chat_input("Ketik pertanyaan Anda di sini... (misal: Berapa harga tiketnya?)"):
    
    # 1. Tampilkan pertanyaan pengguna ke layar
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. Proses jawaban dari sistem LangGraph
    with st.chat_message("assistant"):
        with st.spinner("Berpikir..."):
            hasil = rag_app.invoke({'pertanyaan': prompt, 'retry_count': 0})
            jawaban = hasil['jawaban']
            skor = hasil.get('skor_konteks', 0.0)
            
            # Tampilkan jawaban
            st.markdown(jawaban)
            
            # Tampilkan metrik di dalam expander
            if skor > 0:
                with st.expander("📊 Metrik (UAT)"):
                    st.caption(f"Skor Cosine Similarity: **{skor:.3f}**")
            
    # 3. Simpan jawaban sistem ke dalam memori
    st.session_state.messages.append({"role": "assistant", "content": jawaban, "score": skor})
