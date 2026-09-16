"""
ชิ้นที่ 4: สร้าง Web UI ด้วย Streamlit สำหรับระบบค้นหาและประเมินงานวิจัย

วิธีติดตั้งคลังไลบรารีที่จำเป็น:
    pip install streamlit requests python-dotenv

วิธีรันแอปพลิเคชัน:
    streamlit run step4_app.py
"""

import os
import json
import time
import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# ==========================================
# 1. การตั้งค่าหน้า Streamlit (Page Configuration)
# ==========================================
st.set_page_config(
    page_title="ระบบค้นหาและประเมินบทความวิจัย",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

MODEL_NAME = "gemini-3.5-flash-lite"
GEMINI_ENDPOINT = (
    f"https://generativelanguage.googleapis.com/v1beta/models/"
    f"{MODEL_NAME}:generateContent"
)


# ==========================================
# 2. ฟังก์ชันหลัก (Backend Core Functions)
# ==========================================
def search_papers(query: str, limit: int = 5, api_key: str = "", _retry_count: int = 0):
    """
    ดึง Paper จาก Semantic Scholar API
    """
    url = "https://api.semanticscholar.org/graph/v1/paper/search"
    params = {
        "query": query,
        "limit": limit,
        "fields": "title,abstract,url,year,authors,paperId",
    }
    headers = {"x-api-key": api_key} if api_key else {}

    if api_key:
        time.sleep(1.1)

    try:
        response = requests.get(url, params=params, headers=headers, timeout=15)
    except requests.exceptions.RequestException as e:
        st.error(f"เชื่อมต่อ Semantic Scholar API ไม่ได้: {e}")
        return []

    if response.status_code == 429:
        MAX_RETRIES = 5
        if _retry_count >= MAX_RETRIES:
            st.error(f"Semantic Scholar โดน Rate Limit เกิน {MAX_RETRIES} ครั้ง")
            return []
        wait_time = 5 if _retry_count < 3 else 15
        time.sleep(wait_time)
        return search_papers(query, limit, api_key, _retry_count=_retry_count + 1)

    if response.status_code != 200:
        st.error(f"Semantic Scholar ตอบกลับผิดพลาด: HTTP {response.status_code}")
        return []

    data = response.json()
    raw_papers = data.get("data", [])

    papers = []
    for p in raw_papers:
        if p.get("abstract"):
            papers.append({
                "title": p.get("title"),
                "abstract": p.get("abstract"),
                "url": p.get("url") or f"https://www.semanticscholar.org/paper/{p.get('paperId')}",
                "year": p.get("year", "N/A"),
                "authors": [a.get("name") for a in p.get("authors", [])],
            })

    return papers


def evaluate_relevance(user_query: str, abstract: str, gemini_api_key: str, _retry_count: int = 0):
    """
    ประเมินความเกี่ยวข้องเชิงเนื้อหาด้วย Gemini API (gemini-3.5-flash-lite)
    """
    if not gemini_api_key:
        return {"score": None, "reason": "ไม่พบ Gemini API Key"}

    prompt = f"""คุณเป็นผู้ช่วยประเมินความเกี่ยวข้องเชิงเนื้อหา (semantic relevance)
ระหว่างความสนใจของผู้ใช้กับบทคัดย่องานวิจัย

ห้ามตัดสินจากการจับคำตรงตัว (keyword matching) เท่านั้น ให้พิจารณาความหมาย
และความเกี่ยวข้องเชิงเนื้อหาจริงๆ แม้จะใช้คำคนละคำแต่ความหมายใกล้เคียงกัน
ก็ถือว่าเกี่ยวข้องได้

ความสนใจของผู้ใช้: {user_query}

บทคัดย่องานวิจัย: {abstract}

ประเมินและตอบกลับเป็น JSON เท่านั้น ตามโครงสร้างนี้:
- score: คะแนนความเกี่ยวข้อง 0-100 (0 = ไม่เกี่ยวข้องเลย, 100 = ตรงประเด็นมาก)
- reason: เหตุผลสั้นๆ ไม่เกิน 2 ประโยค ระบุทั้งจุดที่ตรงและจุดที่ไม่ตรง (ถ้ามี)
"""

    body = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "responseMimeType": "application/json",
            "responseSchema": {
                "type": "OBJECT",
                "properties": {
                    "score": {"type": "INTEGER"},
                    "reason": {"type": "STRING"},
                },
                "required": ["score", "reason"],
            },
        },
    }

    headers = {
        "Content-Type": "application/json",
        "x-goog-api-key": gemini_api_key,
    }

    try:
        response = requests.post(GEMINI_ENDPOINT, headers=headers, json=body, timeout=30)
    except requests.exceptions.RequestException as e:
        return {"score": None, "reason": f"เชื่อมต่อ Gemini API ไม่ได้: {e}"}

    if response.status_code == 429:
        MAX_RETRIES = 3
        if _retry_count >= MAX_RETRIES:
            return {"score": None, "reason": "Gemini โดน Rate Limit เกินกำหนด"}
        wait_time = (2 ** _retry_count) * 5
        time.sleep(wait_time)
        return evaluate_relevance(user_query, abstract, gemini_api_key, _retry_count=_retry_count + 1)

    if response.status_code != 200:
        return {"score": None, "reason": f"Gemini API Error ({response.status_code})"}

    try:
        data = response.json()
        raw_text = data["candidates"][0]["content"]["parts"][0]["text"]
        result = json.loads(raw_text)

        score = max(0, min(100, int(result["score"])))
        reason = str(result["reason"])
        return {"score": score, "reason": reason}
    except Exception as e:
        return {"score": None, "reason": f"การแปลง JSON ล้มเหลว: {e}"}


# ==========================================
# 3. ส่วนแสดงผล UI ด้วย Streamlit
# ==========================================

# --- Sidebar: ตั้งค่า API Keys ---
with st.sidebar:
    st.header("⚙️ ตั้งค่า API Keys")

    # อ่านค่าเริ่มต้นจาก Environment Variables / .env (ถ้ามี)
    default_gemini_key = os.environ.get("GEMINI_API_KEY", "")
    default_s2_key = os.environ.get("SEMANTIC_SCHOLAR_API_KEY", "")

    input_gemini_key = st.text_input(
        "Gemini API Key (จำเป็น)",
        value=default_gemini_key,
        type="password",
        help="ขอ Key ฟรีได้ที่ https://aistudio.google.com/"
    )

    input_s2_key = st.text_input(
        "Semantic Scholar Key (ไม่บังคับ)",
        value=default_s2_key,
        type="password",
        help="ถ้าใส่จะช่วยเพิ่มอัตราการดึงข้อมูลและไม่ติด Rate Limit"
    )

    st.markdown("---")
    st.markdown("### 📌 คำแนะนำใช้งาน")
    st.markdown("1. กรอก **Gemini API Key** ให้เรียบร้อย")
    st.markdown("2. พิมพ์หัวข้อ/ความสนใจเป็นประโยคภาษาอังกฤษ")
    st.markdown("3. ระบบจะคัดสรรบทความและประเมินคะแนนให้อัตโนมัติ")


# --- Main Page: ส่วนควบคุมและรับอินพุต ---
st.title("🎓 ระบบค้นหาและประเมินบทความวิจัยสำหรับนิสิต")
st.caption("ช่วยค้นหา Paper และประเมินความตรงประเด็นเชิงเนื้อหา (Semantic Relevance) ด้วย Gemini 3.5 Flash Lite")

st.markdown("---")

# เหลือช่องกรอกเดียว (ภาษาอังกฤษ) — ใช้ข้อความเดียวกันทั้งค้นหาและประเมิน
col1, col2 = st.columns([3, 1])

with col1:
    search_keyword = st.text_area(
        "🔍 หัวข้อ/ความสนใจของคุณ (ภาษาอังกฤษ):",
        value="plant disease detection using deep learning",
        height=100,
        help="พิมพ์เป็นประโยคเต็มได้เลย ไม่ต้องเป็นแค่ keyword — "
             "Semantic Scholar และ Gemini รองรับภาษาอังกฤษได้ดีที่สุด"
    )

with col2:
    paper_limit = st.slider(
        "📊 จำนวนบทความที่ต้องการ:",
        min_value=1,
        max_value=20,
        value=5
    )

search_button = st.button("🚀 ค้นหาและประเมินงานวิจัย", type="primary", use_container_width=True)


# --- ส่วนประมวลผลเมื่อกดปุ่มค้นหา ---
if search_button:
    # ตรวจสอบ API Key
    if not input_gemini_key:
        st.warning("⚠️ กรุณากรอก **Gemini API Key** ในแถบ Sidebar ด้านซ้ายก่อนเริ่มค้นหา")
    elif not search_keyword.strip():
        st.warning("⚠️ กรุณากรอกหัวข้อ/ความสนใจของคุณก่อน")
    else:
        # ใช้ search_keyword ตัวเดียวกันทั้งดึง paper และประเมินความเกี่ยวข้อง
        user_query = search_keyword

        # 1. ขั้นตอนดึงข้อมูล Paper
        with st.spinner(f"กำลังค้นหา Paper จาก Semantic Scholar ด้วยคำค้น '{search_keyword}'..."):
            papers = search_papers(query=search_keyword, limit=paper_limit, api_key=input_s2_key)

        if not papers:
            st.error("❌ ไม่พบบทความวิจัยที่ตรงกับคำค้น หรือเกิดข้อผิดพลาดในการดึงข้อมูล")
        else:
            st.success(f"✅ ดึงบทความที่มี Abstract สำเร็จจำนวน {len(papers)} ฉบับ")

            # 2. ขั้นตอนประเมินความเกี่ยวข้องทีละฉบับพร้อม Progress Bar
            progress_bar = st.progress(0)
            status_text = st.empty()

            evaluated_papers = []
            total = len(papers)

            for idx, paper in enumerate(papers, start=1):
                status_text.text(f"🧠 กำลังประเมิน Paper ที่ {idx}/{total}: {paper['title'][:60]}...")

                eval_res = evaluate_relevance(
                    user_query=user_query,
                    abstract=paper["abstract"],
                    gemini_api_key=input_gemini_key
                )

                paper["score"] = eval_res["score"] if eval_res["score"] is not None else -1
                paper["reason"] = eval_res["reason"]
                evaluated_papers.append(paper)

                # อัปเดต Progress Bar
                progress_bar.progress(idx / total)
                time.sleep(0.8)  # หน่วงเวลาเล็กน้อยเพื่อป้องกัน Rate Limit

            status_text.text("✅ ประเมินบทความวิจัยเรียบร้อยแล้ว!")
            time.sleep(0.5)
            status_text.empty()
            progress_bar.empty()

            # 3. จัดเรียงตามคะแนนความเกี่ยวข้อง (Descending)
            evaluated_papers.sort(key=lambda x: x["score"], reverse=True)

            # 4. แสดงผลลัพธ์
            st.markdown("---")
            st.subheader(f"🏆 ผลการจัดอันดับบทความวิจัยตามความเกี่ยวข้อง ({len(evaluated_papers)} รายการ)")

            for rank, paper in enumerate(evaluated_papers, start=1):
                score = paper["score"]

                # จัดสี Badge ตามระดับคะแนน
                if score >= 80:
                    badge_color = "green"
                    score_label = f"🟢 เกี่ยวข้องมาก ({score}/100)"
                elif score >= 50:
                    badge_color = "orange"
                    score_label = f"🟠 เกี่ยวข้องบางส่วน ({score}/100)"
                elif score >= 0:
                    badge_color = "red"
                    score_label = f"🔴 ไม่เกี่ยวข้อง ({score}/100)"
                else:
                    badge_color = "gray"
                    score_label = "⚪ ประเมินไม่ได้"

                # แสดง Card ของ Paper
                with st.container():
                    col_title, col_score = st.columns([4, 1.2])

                    with col_title:
                        st.markdown(f"### อันดับ {rank}. {paper['title']}")
                    with col_score:
                        st.markdown(f"#### :{badge_color}[{score_label}]")

                    authors_str = ", ".join(paper["authors"][:3])
                    if len(paper["authors"]) > 3:
                        authors_str += " et al."

                    st.markdown(f"📅 **ปีที่พิมพ์:** {paper['year']} | 👤 **ผู้แต่ง:** {authors_str}")
                    st.info(f"💡 **เหตุผลจาก Gemini:** {paper['reason']}")

                    st.markdown(f"🔗 [อ่านบทความวิจัยฉบับเต็ม]({paper['url']})")

                    # ปุ่ม Expander ขยายดู Abstract
                    with st.expander("📄 ดูบทคัดย่อ (Abstract)"):
                        st.write(paper["abstract"])

                    st.markdown("---")
