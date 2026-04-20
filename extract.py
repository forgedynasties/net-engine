import fitz
import ollama
import json
import os
import re
import base64
import argparse
from io import BytesIO
from PIL import Image


def image_to_base64(img: Image.Image) -> str:
    buf = BytesIO()
    img.save(buf, format="JPEG")
    return base64.b64encode(buf.getvalue()).decode()


def fix_backslashes(raw: str) -> str:
    return re.sub(r'(?<!\\)\\([a-zA-Z])', lambda m: '\\\\' + m.group(1), raw)


def repair_json(raw: str) -> str:
    raw = re.sub(r',\s*$', '', raw.rstrip())
    raw += ']' * max(0, raw.count('[') - raw.count(']'))
    raw += '}' * max(0, raw.count('{') - raw.count('}'))
    return raw


def clean_question(q: dict) -> dict:
    raw_id = q.get("id", 0)
    if isinstance(raw_id, str):
        q["id"] = int(re.sub(r'\D', '', raw_id) or 0)
    q["question"] = q.get("question", "").strip()
    q["options"]  = [o.strip() for o in q.get("options", [])]
    q["topic"]    = q.get("topic", "").strip()
    return q


PROMPT = r"""
You are an OCR engine for a scanned Pakistani university entrance exam (KIPS Diagnostic Test).

Extract ALL questions on this page exactly as printed.

OUTPUT: valid JSON only, matching this exact structure:
{
  "questions": [
    {
      "id": 10,
      "question": "...",
      "options": ["(a) ...", "(b) ...", "(c) ...", "(d) ..."],
      "topic": "..."
    }
  ]
}

RULES:
- id is an integer (the question number)
- options are in a 2x2 grid: (a) top-left, (b) top-right, (c) bottom-left, (d) bottom-right
- Use LaTeX for math inside $...$. Double-escape backslashes: \\frac \\sqrt \\sin \\theta etc.
- Do not compute or simplify — transcribe exactly
- A plain number like 1 or -3 is a valid option
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf",     required=True)
    parser.add_argument("--subject", required=True)
    parser.add_argument("--start",   type=int, default=0)
    parser.add_argument("--end",     type=int, default=10)
    parser.add_argument("--page",    type=int)
    parser.add_argument("--model",   default="gemma4:latest")
    parser.add_argument("--debug",   action="store_true")
    args = parser.parse_args()

    output_dir = f"data/{args.subject}"
    os.makedirs(output_dir, exist_ok=True)
    if args.debug:
        os.makedirs("debug_images", exist_ok=True)

    doc   = fitz.open(args.pdf)
    pages = [args.page - 1] if args.page else range(args.start, min(args.end, len(doc)))

    for page_num in pages:
        print(f"\n📄 Page {page_num + 1}")
        try:
            page = doc.load_page(page_num)
            pix  = page.get_pixmap(dpi=300)
            img  = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

            if args.debug:
                img.save(f"debug_images/page_{page_num + 1}.jpg")

            print("🧠 Extracting...")
            full = ""
            for chunk in ollama.generate(
                model=args.model,
                prompt=PROMPT,
                images=[image_to_base64(img)],
                format="json",
                stream=True,
                options={"num_predict": 4096, "temperature": 0, "num_ctx": 8192}
            ):
                tok = chunk["response"]
                print(tok, end="", flush=True)
                full += tok

            print()
            full = fix_backslashes(full)
            full = repair_json(full)
            data = json.loads(full)
            questions = data.get("questions", data) if isinstance(data, dict) else data
            questions = [clean_question(q) for q in questions]

            out = os.path.join(output_dir, f"page_{page_num + 1}.json")
            with open(out, "w") as f:
                json.dump({"questions": questions}, f, indent=2, ensure_ascii=False)
            print(f"✅ Saved {len(questions)} questions → {out}")

        except KeyboardInterrupt:
            print("\n⏹️  Interrupted")
            break
        except Exception as e:
            print(f"❌ Page {page_num + 1}: {e}")


if __name__ == "__main__":
    main()
