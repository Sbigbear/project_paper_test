"""
ชิ้นที่ 1: ฟังก์ชันดึง paper จาก Semantic Scholar API

วิธีรัน:
    pip install requests python-dotenv
    python step1_semantic_scholar.py
"""

import os
import requests
import time
import urllib.parse
import xml.etree.ElementTree as ET
from dotenv import load_dotenv

# โหลดค่าจากไฟล์ .env (ถ้ามีไฟล์นี้อยู่) เข้ามาเป็น environment variable
# ถ้าไม่มีไฟล์ .env ก็ไม่ error แค่จะไม่มีอะไรถูกโหลดเพิ่ม
load_dotenv()

# ค่า default ต้องเป็น "" เท่านั้น — ห้ามใส่ key จริงตรงนี้ (ดูคำเตือนด้านบน)
API_KEY = os.environ.get("SEMANTIC_SCHOLAR_API_KEY", "")


def search_papers(query: str, limit: int = 10, _retry_count: int = 0):
    """
    ค้นหา paper จาก Semantic Scholar ด้วยคำค้น (query)

    query : ข้อความค้นหา เช่น "plant disease detection"
             (หมายเหตุ: Semantic Scholar API ตัวนี้รองรับภาษาอังกฤษได้ดีที่สุด
              ถ้าจะค้นด้วยประโยคภาษาไทย แนะนำแปลเป็นอังกฤษก่อนส่งเข้า API นี้
              — เดี๋ยวเราจะจัดการเรื่องนี้ในชิ้นถัดๆ ไป)
    limit : จำนวน paper สูงสุดที่ต้องการ (ค่าเริ่มต้น 10)
    _retry_count : ใช้ภายในฟังก์ชันเอง ไม่ต้องส่งค่าตอนเรียกใช้

    คืนค่า: list ของ dict แต่ละตัวมี title, abstract, url, year
    """
    url = "https://api.semanticscholar.org/graph/v1/paper/search"

    params = {
        "query": query,
        "limit": limit,
        # เลือกเฉพาะฟิลด์ที่ต้องใช้ ไม่ดึงข้อมูลเกินจำเป็น
        "fields": "title,abstract,url,year,authors",
    }

    headers = {"x-api-key": API_KEY} if API_KEY else {}

    if _retry_count == 0 and not API_KEY:
        print("[INFO] ไม่ได้ใส่ API key — จะใช้โควตารวม อาจช้ากว่าปกติ "
              "(ขอ key ฟรีได้ที่ semanticscholar.org/product/api)")

    # มี API key แล้ว โควตาคือ 1 request/วินาที
    # หน่วงเวลาเล็กน้อยก่อนยิง request ทุกครั้ง (รวมถึงตอน retry)
    # เพื่อไม่ให้ยิงถี่เกินโควตาแล้วโดน 429 ซ้ำ
    if API_KEY:
        time.sleep(1.1)  # เผื่อ margin เล็กน้อยจาก 1 req/sec พอดิบพอดี

    try:
        response = requests.get(url, params=params, headers=headers, timeout=15)
    except requests.exceptions.RequestException as e:
        print(f"[ERROR] เชื่อมต่อ API ไม่ได้: {e}")
        return []

    # Semantic Scholar free tier มี rate limit (429 = ขอถี่เกินไป)
    if response.status_code == 429:
        MAX_RETRIES = 5
        if _retry_count >= MAX_RETRIES:
            print(f"[ERROR] โดน rate limit เกิน {MAX_RETRIES} ครั้ง — ยอมแพ้ "
                  f"ลองใหม่อีกทีภายหลัง หรือขอ API key เพื่อลดปัญหานี้")
            return []

        # หลัง retry เกิน 3 ครั้ง ให้รอนานขึ้นเป็น 10-15 วินาที
        # (ถี่ๆ ทุก 5 วิเหมือนเดิมมีแต่จะโดน block ซ้ำ)
        wait_time = 5 if _retry_count < 3 else 15
        print(f"[WARN] โดน rate limit (ครั้งที่ {_retry_count + 1}) — "
              f"รอ {wait_time} วินาทีแล้วลองใหม่...")
        time.sleep(wait_time)
        return search_papers(query, limit, _retry_count=_retry_count + 1)

    if response.status_code != 200:
        print(f"[ERROR] API ตอบกลับผิดพลาด: {response.status_code}")
        print(response.text)
        return []

    data = response.json()
    raw_papers = data.get("data", [])

    # กรอง paper ที่ไม่มี abstract ทิ้ง เพราะขั้นตอนถัดไปต้องใช้ abstract
    # ไปประเมินความเกี่ยวข้อง ถ้าไม่มี abstract ก็ประเมินไม่ได้
    papers = []
    for p in raw_papers:
        if p.get("abstract"):
            papers.append({
                "title": p.get("title"),
                "abstract": p.get("abstract"),
                "url": p.get("url"),
                "year": p.get("year"),
                "authors": [a.get("name") for a in p.get("authors", [])],
            })

    return papers


def search_papers_arxiv(query: str, limit: int = 10):
    """
    ฟังก์ชันสำรอง: ค้นหา paper จาก arXiv API แทน Semantic Scholar
    ไม่ต้องใช้ API key เลย ใช้งานได้ทันที

    รูปแบบ input/output เหมือน search_papers() เดิมทุกอย่าง
    เพื่อให้สลับใช้แทนกันได้ในโค้ดส่วนอื่น (เช่นตอนเชื่อม Gemini)

    ข้อจำกัด: arXiv ครอบคลุมสาย physics, math, CS/AI, statistics,
    electrical engineering, quantitative biology/finance, economics
    เป็นหลัก แทบไม่มี paper สายเกษตร/พืชศาสตร์/การแพทย์คลินิก/
    สังคมศาสตร์ ถ้าหัวข้อที่ค้นอยู่นอกสายนี้ ผลลัพธ์จะน้อยหรือไม่ตรง
    """
    base_url = "http://export.arxiv.org/api/query"

    # arXiv ต้องการ query ในรูปแบบ "all:คำค้น" และต้อง URL-encode
    encoded_query = urllib.parse.quote(f"all:{query}")

    params_str = (
        f"search_query={encoded_query}"
        f"&start=0"
        f"&max_results={limit}"
        f"&sortBy=relevance"
        f"&sortOrder=descending"
    )

    url = f"{base_url}?{params_str}"

    try:
        response = requests.get(url, timeout=15)
    except requests.exceptions.RequestException as e:
        print(f"[ERROR] เชื่อมต่อ arXiv API ไม่ได้: {e}")
        return []

    if response.status_code != 200:
        print(f"[ERROR] arXiv API ตอบกลับผิดพลาด: {response.status_code}")
        return []

    # arXiv ตอบกลับเป็น XML (Atom feed) ไม่ใช่ JSON แบบ Semantic Scholar
    # เลยต้อง parse ด้วย xml.etree แทน
    try:
        root = ET.fromstring(response.text)
    except ET.ParseError as e:
        print(f"[ERROR] parse XML จาก arXiv ไม่ได้: {e}")
        return []

    # arXiv ใช้ Atom namespace ต้องระบุตอนหา tag
    ns = {"atom": "http://www.w3.org/2005/Atom"}

    papers = []
    for entry in root.findall("atom:entry", ns):
        title_el = entry.find("atom:title", ns)
        summary_el = entry.find("atom:summary", ns)
        id_el = entry.find("atom:id", ns)
        published_el = entry.find("atom:published", ns)

        title = title_el.text.strip().replace("\n", " ") if title_el is not None else None
        abstract = summary_el.text.strip().replace("\n", " ") if summary_el is not None else None
        url_link = id_el.text.strip() if id_el is not None else None

        # published มีรูปแบบ "2023-05-01T00:00:00Z" เอาแค่ปี (4 ตัวแรก)
        year = int(published_el.text[:4]) if published_el is not None else None

        authors = [
            name_el.text
            for author_el in entry.findall("atom:author", ns)
            for name_el in author_el.findall("atom:name", ns)
        ]

        # ข้าม entry ที่ไม่มี abstract เหมือนกับฝั่ง Semantic Scholar
        if abstract:
            papers.append({
                "title": title,
                "abstract": abstract,
                "url": url_link,
                "year": year,
                "authors": authors,
            })

    return papers


# ---------------- โค้ดทดสอบ ----------------
if __name__ == "__main__":
    test_query = "plant disease detection deep learning"

    # ตั้ง USE_ARXIV = True เพื่อทดสอบ arXiv แทน (ใช้ได้ทันที ไม่ต้องมี key)
    # ตั้ง USE_ARXIV = False เพื่อกลับไปใช้ Semantic Scholar (ต้อง/ควรมี key)
    USE_ARXIV = False  # มี API key แล้ว กลับมาใช้ Semantic Scholar เป็นหลัก

    print(f"กำลังค้นหา: '{test_query}' "
          f"(แหล่งข้อมูล: {'arXiv' if USE_ARXIV else 'Semantic Scholar'}) ...\n")

    if USE_ARXIV:
        results = search_papers_arxiv(test_query, limit=5)
    else:
        results = search_papers(test_query, limit=5)

    print(f"เจอ paper ที่มี abstract ทั้งหมด {len(results)} ฉบับ\n")
    print("=" * 60)

    for i, paper in enumerate(results, start=1):
        print(f"{i}. {paper['title']} ({paper['year']})")
        print(f"   ผู้แต่ง: {', '.join(paper['authors'][:3])}")
        print(f"   abstract: {paper['abstract'][:150]}...")
        print(f"   ลิงก์: {paper['url']}")
        print("-" * 60)
