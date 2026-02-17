import os
import sqlite3
from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI

load_dotenv()
app = FastAPI()

# ✅ السماح لجميع الأجهزة بالاتصال عبر Nginx
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- قاعدة البيانات ---
def get_db_connection():
    conn = sqlite3.connect('psych_consultant.db', check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db_connection()
    conn.execute('''CREATE TABLE IF NOT EXISTS messages 
                    (user_id TEXT, role TEXT, content TEXT, timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)''')
    conn.commit()
    conn.close()

init_db()

# --- محرك البحث RAG ---
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

def build_vector_db():
    path = "./knowledge_base/"
    if not os.path.exists(path): 
        os.makedirs(path)
        return None
    loader = PyPDFDirectoryLoader(path)
    documents = loader.load()
    if not documents: return None
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
    splits = text_splitter.split_documents(documents)
    return FAISS.from_documents(splits, embeddings)

db = build_vector_db()

@app.get("/")
async def root():
    return {"status": "online", "message": "Psych Consultant API is running"}

@app.post("/chat")
async def chat_endpoint(request: Request):
    try:
        data = await request.json()
        user_query = data.get("message", "")
        user_id = data.get("user_id", "guest_user") 

        # 1. نظام الحماية
        safety_words = ["انتحار", "إيذاء", "أقتل نفسي", "انتحر", "أنهي حياتي"]
        if any(word in user_query for word in safety_words):
            return {
                "response": "أنا أهتم لأمرك. يرجى التواصل مع المختصين فوراً أو الاتصال بـ 911.", 
                "source": "🚨 نظام الحماية",
                "isEmergency": True
            }

        # 2. حفظ السياق
        conn = get_db_connection()
        conn.execute("INSERT INTO messages (user_id, role, content) VALUES (?, ?, ?)", (user_id, "user", user_query))
        conn.commit()

        # 3. جلب الذاكرة
        past_msgs = conn.execute("SELECT role, content FROM messages WHERE user_id = ? ORDER BY rowid DESC LIMIT 6", (user_id,)).fetchall()
        memory_text = "\n".join([f"{'المستخدم' if m['role']=='user' else 'المستشار'}: {m['content']}" for m in reversed(past_msgs)])

        # 4. البحث في الملفات
        context = ""
        if db:
            docs = db.similarity_search(user_query, k=3)
            context = "\n".join([d.page_content for d in docs])

        system_instruction = f"أنت مستشار نفسي دافئ. سياق المحادثة:\n{memory_text}\n\nمراجع علمية:\n{context}"
        
        # 5. الموديلات (Groq -> OpenAI)
        final_response = ""
        badge = "استشارة ذكية"
        try:
            llm = ChatGroq(temperature=0.4, model_name="llama-3.1-8b-instant", groq_api_key=os.getenv("GROQ_API_KEY"))
            final_response = llm.invoke([{"role": "system", "content": system_instruction}, {"role": "user", "content": user_query}]).content
        except:
            llm_backup = ChatOpenAI(model_name="gpt-4o-mini", openai_api_key=os.getenv("OPENAI_API_KEY"))
            final_response = llm_backup.invoke([{"role": "system", "content": system_instruction}, {"role": "user", "content": user_query}]).content
            badge = "استشارة احتياطية"

        conn.execute("INSERT INTO messages (user_id, role, content) VALUES (?, ?, ?)", (user_id, "assistant", final_response))
        conn.commit()
        conn.close()

        return {"response": final_response, "source": badge, "isEmergency": False}
    except Exception as e:
        return {"response": "حدث خطأ فني.", "error": str(e)}
