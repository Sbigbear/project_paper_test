"""
ชิ้นที่ 2: ประเมินความเกี่ยวข้องเชิงเนื้อหาระหว่างความสนใจของผู้ใช้ กับ abstract
โดยใช้ Gemini API (รุ่น gemini-3.5-flash-lite)

*** เรียก Gemini ผ่าน REST API ตรงๆ ด้วย requests ***
(หลีกเลี่ยงปัญหา Dependency บน Python 3.9 + Windows)

วิธีรัน:
    pip install requests
    python step2_gemini_eval.py

การตั้งค่า GEMINI_API_KEY:
    Windows (PowerShell): $env:GEMINI_API_KEY="AIza..."
    macOS/Linux:         export GEMINI_API_KEY="AIza..."
"""

import os
import json
import time
import requests

# ----- ตั้งค่าโมเดลและ Endpoint -----
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")

# อัปเดตโมเดลเป็น gemini-3.5-flash-lite
MODEL_NAME = "gemini-3.5-flash-lite"

GEMINI_ENDPOINT = (
    f"https://generativelanguage.googleapis.com/v1beta/models/"
    f"{MODEL_NAME}:generateContent"
)


def evaluate_relevance(user_query: str, abstract: str, gemini_api_key: str, _retry_count: int = 0):
    """
    ประเมินความเกี่ยวข้องเชิงเนื้อหาระหว่าง user_query กับ abstract โดยใช้ Gemini API
    
    คืนค่า: dict {"score": int (0-100), "reason": str}
            กรณีเกิด error จะคืนค่า {"score": None, "reason": "[ERROR] ..."}
    """
    if not gemini_api_key:
        return {"score": None, "reason": "[ERROR] ไม่พบ GEMINI_API_KEY กรุณาตั้งค่าสภาพแวดล้อมหรือระบุ Key"}

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

    # บังคับโครงสร้าง JSON ตอบกลับผ่าน generationConfig
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

    # จัดการ Error 429: Rate Limit ( exponential backoff )
    if response.status_code == 429:
        MAX_RETRIES = 3
        if _retry_count >= MAX_RETRIES:
            return {
                "score": None,
                "reason": f"[ERROR] โดน Rate Limit เกิน {MAX_RETRIES} ครั้ง — กรุณารอสักครู่แล้วลองใหม่"
            }

        wait_time = (2 ** _retry_count) * 5  # หน่วงเวลาเพิ่มขึ้นเรื่อยๆ: 5s, 10s, 20s
        print(f"[WARN] Gemini โดน Rate Limit (429) - ครั้งที่ {_retry_count + 1} — ชะลอรอ {wait_time} วินาที...")
        time.sleep(wait_time)
        return evaluate_relevance(user_query, abstract, gemini_api_key, _retry_count=_retry_count + 1)

    # จัดการ HTTP status error อื่นๆ
    if response.status_code != 200:
        return {
            "score": None,
            "reason": f"[ERROR] Gemini API ตอบกลับผิดพลาด ({response.status_code}): {response.text[:200]}"
        }

    # จัดการ JSON Parse Error
    try:
        data = response.json()
        raw_text = data["candidates"][0]["content"]["parts"][0]["text"]
        result = json.loads(raw_text)

        score = int(result["score"])
        reason = str(result["reason"])
        
        # คุมขอบเขตคะแนนให้อยู่ในช่วง 0 - 100
        score = max(0, min(100, score))

        return {"score": score, "reason": reason}

    except (json.JSONDecodeError, KeyError, IndexError, ValueError, TypeError) as e:
        return {
            "score": None,
            "reason": f"[ERROR] การแปลงข้อมูล JSON จาก Gemini ล้มเหลว: {e} | Raw: {response.text[:150]}"
        }


# ---------------- โค้ดทดสอบ 3 กรณี ----------------
if __name__ == "__main__":
    if not GEMINI_API_KEY:
        print("❌ [ERROR] ยังไม่ได้ตั้งค่า GEMINI_API_KEY")
        print("คำแนะนำ: กำหนดค่าด้วยคำสั่ง PowerShell ด้านล่างนี้ก่อนรัน:")
        print('  $env:GEMINI_API_KEY="AIza..."\n')
        exit(1)

    test_query = "งานวิจัยเกี่ยวกับการใช้ AI หรือ deep learning ช่วยวินิจฉัยโรคพืช"

    test_cases = [
        {
            "label": "กรณีที่ 1: เกี่ยวข้องมาก (High Relevance)",
            "abstract": (
                "This paper proposes a convolutional neural network model for "
                "detecting pest infestation and plant pathogens from leaf images, "
                "achieving 94% accuracy across five crop species."
            ),
        },
        {
            "label": "กรณีที่ 2: เกี่ยวข้องบางส่วน (Partial Relevance)",
            "abstract": (
                "We survey machine learning techniques applied to agricultural "
                "yield prediction, covering weather data integration and soil "
                "quality analysis, with a brief mention of disease risk factors."
            ),
        },
        {
            "label": "กรณีที่ 3: ไม่เกี่ยวข้อง (Irrelevant)",
            "abstract": (
                "This study analyzes stock market volatility using recurrent "
                "neural networks and evaluates prediction accuracy across "
                "different market conditions over a 10-year period."
            ),
        },
    ]

    print(f"🎯 หัวข้อความสนใจของผู้ใช้: \"{test_query}\"\n")
    print("=" * 70)

    for case in test_cases:
        print(f"\n📌 {case['label']}")
        print(f"📄 Abstract: {case['abstract']}")

        # เรียกใช้งานฟังก์ชันประเมินความเกี่ยวข้อง
        result = evaluate_relevance(test_query, case["abstract"], GEMINI_API_KEY)

        if result["score"] is not None:
            print(f"📊 คะแนนความเกี่ยวข้อง: {result['score']}/100")
            print(f"💡 เหตุผล: {result['reason']}")
        else:
            print(f"❌ สถานะ: {result['reason']}")

        print("-" * 70)