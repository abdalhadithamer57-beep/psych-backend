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

# إعداد CORS للسماح بالاتصال من أي جهاز (مهم لتجربة الجوال)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- إعداد قاعدة البيانات ---
def get_db_connection():
    # نضع المحادثات في ملف sqlite محلي
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

# --- إعداد محرك البحث في الملفات (PDF) ---
# نستخدم نموذج يدعم العربية بامتياز
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

def build_vector_db():
    path = "./knowledge_base/"
    if not os.path.exists(path): 
        os.makedirs(path)
        return None
    
    loader = PyPDFDirectoryLoader(path)
    documents = loader.load()
    if not documents: 
        print("⚠️ لم يتم العثور على ملفات PDF في مجلد knowledge_base")
        return None
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
    splits = text_splitter.split_documents(documents)
    return FAISS.from_documents(splits, embeddings)

db = build_vector_db()

GROQ_KEY = os.getenv("GROQ_API_KEY")
OPENAI_KEY = os.getenv("OPENAI_API_KEY")

@app.post("/chat")
async def chat_endpoint(request: Request):
    try:
        data = await request.json()
        user_query = data.get("message", "")
        # استلام user_id المرسل من الواجهة الأمامية
        user_id = data.get("user_id", "guest_user") 

        # 1. فحص كلمات الطوارئ (الأمان)
        safety_words = ["انتحار", "إيذاء", "أقتل نفسي", "انتحر", "أنهي حياتي"]
        if any(word in user_query for word in safety_words):
            emergency_msg = "أنا أهتم لأمرك جداً. يبدو أنك تمر بوقت صعب للغاية. من فضلك، تواصل مع المختصين فوراً أو اتصل بخط المساعدة 911. أنت لست وحدك وهناك دائماً أمل."
            return {
                "response": emergency_msg, 
                "source": "🚨 نظام الحماية الطارئ",
                "isEmergency": True
            }

        # 2. حفظ رسالة المستخدم في القاعدة
        conn = get_db_connection()
        conn.execute("INSERT INTO messages (user_id, role, content) VALUES (?, ?, ?)", 
                     (user_id, "user", user_query))
        conn.commit()

        # 3. جلب آخر 6 رسائل فقط لبناء سياق ذكي (Memory)
        past_msgs = conn.execute("SELECT role, content FROM messages WHERE user_id = ? ORDER BY rowid DESC LIMIT 6", 
                                 (user_id,)).fetchall()
        
        memory_text = ""
        for msg in reversed(past_msgs):
            role_label = "المستخدم" if msg['role'] == 'user' else "المستشار"
            memory_text += f"{role_label}: {msg['content']}\n"

        # 4. البحث في الملفات العلمية (RAG)
        context = ""
        if db:
            try:
                docs = db.similarity_search(user_query, k=3)
                context = "\n".join([d.page_content for d in docs])
            except Exception as e:
                print(f"Search Error: {e}")

        # 5. صياغة التعليمات النهائية للموديل
        system_instruction = f"""
        أنت مستشار نفسي دافئ، متعاطف، ومهني جداً. 
        تستخدم مهارات الاستماع النشط والذكاء العاطفي.
        
        سياق المحادثة السابقة:
        {memory_text}

        مراجع علمية مساعدة (استخدمها بذكاء):
        {context}
        
        أجب باللغة العربية بأسلوب بسيط وهادئ. لا تقدم تشخيصات طبية نهائية، بل قدم دعماً وإرشاداً.
        """
        
        # 6. منطق التبديل التلقائي (Failover)
        final_response = ""
        badge = "استشارة ذكية"
        
        try:
            # الخيار الأول: Groq (سريع جداً)
            llm = ChatGroq(temperature=0.4, model_name="llama-3.1-8b-instant", groq_api_key=GROQ_KEY)
            response = llm.invoke([
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": user_query}
            ])
            final_response = response.content
        except Exception as e:
            print(f"Groq failed: {e}. Switching to OpenAI...")
            try:
                # الخيار الثاني: OpenAI (دقيق جداً)
                llm_backup = ChatOpenAI(model_name="gpt-4o-mini", openai_api_key=OPENAI_KEY)
                response = llm_backup.invoke([
                    {"role": "system", "content": system_instruction},
                    {"role": "user", "content": user_query}
                ])
                final_response = response.content
                badge = "استشارة احتياطية"
            except:
                final_response = "عذراً، أواجه ضغطاً تقنياً حالياً. هل يمكنك المحاولة مرة أخرى بعد لحظات؟"

        # 7. حفظ رد المستشار في القاعدة لتذكره مستقبلاً
        conn.execute("INSERT INTO messages (user_id, role, content) VALUES (?, ?, ?)", 
                     (user_id, "assistant", final_response))
        conn.commit()
        conn.close()

        return {
            "response": final_response, 
            "source": badge, 
            "isEmergency": False
        }

    except Exception as e:
        print(f"Endpoint Error: {e}")
        return {"response": "حدث خطأ تقني غير متوقع.", "error": True}