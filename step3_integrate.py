"""
ชิ้นที่ 3: รวมระบบค้นหา Paper และประเมินความเกี่ยวข้องเข้าด้วยกัน

วิธีรัน:
    pip install requests python-dotenv
    python step3_integrate.py


import os
import json
import time
import requests
from dotenv import load_dotenv

# โหลดค่าจาก .env เข้ามาเป็น environment variable (ถ้าไม่มีไฟล์นี้ก็ไม่ error)
load_dotenv()

# ==========================================
# 1. ตั้งค่า API Keys และ Endpoint
# ==========================================
# ค่า default ต้องเป็น "" เท่านั้น — ห้ามใส่ key จริงตรงนี้ (ดูคำเตือนด้านบน)
SEMANTIC_SCHOLAR_API_KEY = os.environ.get("SEMANTIC_SCHOLAR_API_KEY", "")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")

MODEL_NAME = "gemini-3.5-flash-lite"
GEMINI_ENDPOINT = (
    f"https://generativelanguage.googleapis.com/v1beta/models/"
    f"{MODEL_NAME}:generateContent"
)


# ==========================================
# 2. ฟังก์ชันดึง Paper จาก Semantic Scholar (ชิ้นที่ 1)
# ==========================================
def search_papers(query: str, limit: int = 10, _retry_count: int = 0):
    """
    ค้นหา Paper จาก Semantic Scholar API ตาม search_keyword (ภาษาอังกฤษ)
    คืนค่า list ของ dict แต่ละตัวมี title, abstract, url, year, authors
    """
    url = "https://api.semanticscholar.org/graph/v1/paper/search"
    params = {
        "query": query,
        "limit": limit,
        "fields": "title,abstract,url,year,authors",
    }
    headers = {"x-api-key": SEMANTIC_SCHOLAR_API_KEY} if SEMANTIC_SCHOLAR_API_KEY else {}

    if SEMANTIC_SCHOLAR_API_KEY:
        time.sleep(1.1)  # หน่วงเวลา 1.1 วินาทีเพื่อไม่ให้เกินโควตา 1 req/sec

    try:
        response = requests.get(url, params=params, headers=headers, timeout=15)
    except requests.exceptions.RequestException as e:
        print(f"[ERROR] เชื่อมต่อ Semantic Scholar API ไม่ได้: {e}")
        return []

    # จัดการ Rate Limit (429) ของ Semantic Scholar
    if response.status_code == 429:
        MAX_RETRIES = 5
        if _retry_count >= MAX_RETRIES:
            print(f"[ERROR] Semantic Scholar โดน Rate Limit เกิน {MAX_RETRIES} ครั้ง")
            return []

        wait_time = 5 if _retry_count < 3 else 15
        print(f"[WARN] Semantic Scholar โดน Rate Limit (ครั้งที่ {_retry_count + 1}) — รอ {wait_time} วินาที...")
        time.sleep(wait_time)
        return search_papers(query, limit, _retry_count=_retry_count + 1)

    if response.status_code != 200:
        print(f"[ERROR] Semantic Scholar ตอบกลับผิดพลาด: {response.status_code}")
        return []

    data = response.json()
    raw_papers = data.get("data", [])

    papers = []
    for p in raw_papers:
        if p.get("abstract"):  # กรองเอาเฉพาะ Paper ที่มี Abstract เท่านั้น
            papers.append({
                "title": p.get("title"),
                "abstract": p.get("abstract"),
                "url": p.get("url") or f"https://www.semanticscholar.org/paper/{p.get('paperId')}",
                "year": p.get("year", "N/A"),
                "authors": [a.get("name") for a in p.get("authors", [])],
            })

    return papers


# ==========================================
# 3. ฟังก์ชันประเมินความเกี่ยวข้องด้วย Gemini (ชิ้นที่ 2)
# ==========================================
def evaluate_relevance(user_query: str, abstract: str, gemini_api_key: str, _retry_count: int = 0):
    """
    ประเมินความเกี่ยวข้องเชิงเนื้อหาด้วย Gemini API (gemini-3.5-flash-lite)
    คืนค่า dict {"score": int (0-100), "reason": str}
    """
    if not gemini_api_key:
        return {"score": None, "reason": "[ERROR] ไม่พบ GEMINI_API_KEY"}

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
        "contents": [
            {"parts": [{"text": prompt}]}
        ],
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
        response = requests.post(
            GEMINI_ENDPOINT, headers=headers, json=body, timeout=30
        )
    except requests.exceptions.RequestException as e:
        return {"score": None, "reason": f"[ERROR] เชื่อมต่อ Gemini API ไม่ได้: {e}"}

    # จัดการ Rate Limit (429) ของ Gemini API
    if response.status_code == 429:
        MAX_RETRIES = 3
        if _retry_count >= MAX_RETRIES:
            return {"score": None, "reason": f"[ERROR] Gemini โดน Rate Limit เกิน {MAX_RETRIES} ครั้ง"}

        wait_time = (2 ** _retry_count) * 5  # Exponential backoff: 5s, 10s, 20s
        print(f"   [WARN] Gemini โดน Rate Limit (429) — ชะลอรอ {wait_time} วินาที...")
        time.sleep(wait_time)
        return evaluate_relevance(user_query, abstract, gemini_api_key, _retry_count=_retry_count + 1)

    if response.status_code != 200:
        return {"score": None, "reason": f"[ERROR] Gemini API Error ({response.status_code}): {response.text[:150]}"}

    try:
        data = response.json()
        raw_text = data["candidates"][0]["content"]["parts"][0]["text"]
        result = json.loads(raw_text)

        score = max(0, min(100, int(result["score"])))
        reason = str(result["reason"])

        return {"score": score, "reason": reason}

    except (json.JSONDecodeError, KeyError, IndexError, ValueError, TypeError) as e:
        return {"score": None, "reason": f"[ERROR] JSON Parse Error: {e}"}


# ==========================================
# 4. ฟังก์ชันหลักสำหรับรวมระบบ (ชิ้นที่ 3)
# ==========================================
def search_and_evaluate(user_query: str, search_keyword: str, limit: int = 5):
    """
    1. ค้นหา Paper ด้วย search_keyword
    2. ส่งแต่ละ Paper ให้ Gemini ประเมินความเกี่ยวข้องเทียบกับ user_query
    3. เรียงลำดับ Paper ตามคะแนนความเกี่ยวข้อง ( score ) จากมากไปน้อย
    4. คืนค่า list ของ Paper ที่ประเมินเรียบร้อยแล้ว
    """
    print(f"\n🔍 [1/3] กำลังค้นหา Paper จาก Semantic Scholar ด้วยคำค้น: '{search_keyword}'...")
    papers = search_papers(query=search_keyword, limit=limit)

    if not papers:
        print("❌ ไม่พบบทความวิจัยที่ตรงกับคำค้น หรือเกิดข้อผิดพลาดในการดึงข้อมูล")
        return []

    print(f"✅ พบ Paper ที่มี Abstract จำนวน {len(papers)} ฉบับ")
    print(f"\n🧠 [2/3] กำลังประเมินความเกี่ยวข้องด้วย Gemini API (gemini-3.5-flash-lite)...")

    evaluated_papers = []

    for idx, paper in enumerate(papers, start=1):
        print(f"   - ประเมิน Paper ที่ {idx}/{len(papers)}: {paper['title'][:50]}...")

        # เรียก Gemini API เพื่อประเมินความเกี่ยวข้อง
        eval_result = evaluate_relevance(
            user_query=user_query,
            abstract=paper["abstract"],
            gemini_api_key=GEMINI_API_KEY
        )

        # รวมผลประเมินเข้ากับข้อมูล Paper เดิม
        paper["score"] = eval_result["score"] if eval_result["score"] is not None else -1
        paper["reason"] = eval_result["reason"]
        evaluated_papers.append(paper)

        # ใส่ Delay เล็กน้อยระหว่างการยิง Gemini API เพื่อป้องกันโดน Rate Limit 429
        time.sleep(1.0)

    print("\n📊 [3/3] กำลังเรียงลำดับผลลัพธ์ตามคะแนนความเกี่ยวข้อง...")
    # เรียงลำดับจากคะแนนมากไปน้อย (Descending order)
    evaluated_papers.sort(key=lambda x: x["score"], reverse=True)

    return evaluated_papers


# ==========================================
# 5. ฟังก์ชันแสดงผลบน Terminal
# ==========================================
def display_results(user_query: str, evaluated_papers: list):
    """
    แสดงผลลัพธ์ผ่าน Terminal อย่างสวยงาม เรียงตามอันดับคะแนน
    """
    print("\n" + "=" * 80)
    print(f"🎯 ความสนใจของผู้ใช้ (ภาษาไทย): {user_query}")
    print("=" * 80)

    if not evaluated_papers:
        print("ไม่มีข้อมูลที่จะแสดงผล")
        return

    for rank, paper in enumerate(evaluated_papers, start=1):
        score_str = f"{paper['score']}/100" if paper['score'] != -1 else " N/A "
        authors_str = ", ".join(paper["authors"][:3])
        if len(paper["authors"]) > 3:
            authors_str += " et al."

        print(f"\n🏆 อันดับที่ {rank}  |  คะแนนความเกี่ยวข้อง: [{score_str}]")
        print(f"📌 ชื่อเรื่อง : {paper['title']}")
        print(f"📅 ปีที่พิมพ์ : {paper['year']}  |  ผู้แต่ง: {authors_str}")
        print(f"💡 เหตุผลจาก Gemini : {paper['reason']}")
        print(f"🔗 ลิงก์บทความ       : {paper['url']}")
        print("-" * 80)


# ==========================================
# 6. ส่วนทดสอบการรันแบบ Workflow (main)
# ==========================================
if __name__ == "__main__":
    if not GEMINI_API_KEY:
        print("❌ [ERROR] ยังไม่ได้ตั้งค่า GEMINI_API_KEY")
        print("คำแนะนำ: คัดลอก .env.example เป็น .env แล้วใส่ key จริงลงไป\n")
        exit(1)

    # กำหนดค่าทดสอบ
    test_user_query = "อยากได้งานวิจัยเกี่ยวกับการใช้ AI หรือ Deep Learning มาช่วยตรวจจับโรคพืชจากใบพืช"
    test_search_keyword = "plant disease detection deep learning"
    test_limit = 5

    # สั่งรัน Workflow ค้นหาและประเมิน
    results = search_and_evaluate(
        user_query=test_user_query,
        search_keyword=test_search_keyword,
        limit=test_limit
    )

    # แสดงผลลัพธ์
    display_results(user_query=test_user_query, evaluated_papers=results)
