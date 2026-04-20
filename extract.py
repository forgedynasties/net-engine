import fitz  # PyMuPDF
import ollama
import json
import os
import base64
import argparse
import sys
from io import BytesIO
from PIL import Image

def image_to_base64(img):
    buffered = BytesIO()
    img.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

def main():
    parser = argparse.ArgumentParser(description="Batch OCR for NET MCQs")
    parser.add_argument("--pdf", required=True)
    parser.add_argument("--subject", required=True)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=10)
    parser.add_argument("--page", type=int)
    parser.add_argument("--model", default="gemma4")
    parser.add_argument("--debug", action="store_true")
    
    args = parser.parse_args()

    output_dir = f"data/{args.subject}"
    debug_dir = "debug_images"
    os.makedirs(output_dir, exist_ok=True)
    if args.debug: os.makedirs(debug_dir, exist_ok=True)

    doc = fitz.open(args.pdf)
    pages_to_process = [args.page - 1] if args.page else range(args.start, min(args.end, len(doc)))

    for page_num in pages_to_process:
        print(f"\n🔍 [PyMuPDF] Loading Page {page_num + 1}...")
        page = doc.load_page(page_num)
        pix = page.get_pixmap(dpi=300) 
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        
        if args.debug:
            img.save(f"{debug_dir}/page_{page_num + 1}.jpg")
            print(f"📸 Debug image saved to {debug_dir}")

        prompt = r"""
        You are an expert OCR engine.
        I am looking at a page from a 'KIPS Diagnostic Test'.

        STEP 1: Count how many question numbers are visible on this page (e.g., 10, 11, 12...).
        STEP 2: Extract EVERY SINGLE one of those questions.

        OUTPUT FORMAT: A JSON object with a "questions" array.
        {
          "questions": [
            {"id": 10, "question": "...", "options": ["(a) ...", "(b) ...", "(c) ...", "(d) ..."], "topic": "..."},
            ...
          ]
        }

        RULES:
        1. Use LaTeX for ALL math expressions (inline: $...$, display: $$...$$).
        2. In JSON strings, ALWAYS double-escape LaTeX backslashes: \\frac, \\sqrt, \\theta, etc.
        3. Ensure all 4 options are captured for every question.
        4. Do not stop until the last question on the page is reached.
        """
        
        print(f"🧠 [Ollama] {args.model} is thinking...\n" + "-"*30)
        
        full_response = ""
        try:
            # Enabling streaming for verbose output
            stream = ollama.generate(
                model=args.model,
                prompt=prompt,
                images=[image_to_base64(img)],
                format="json",
                stream=True,
                options={
                    "num_predict": 4096, 
                    "temperature": 0,    # Keep it deterministic
                    "top_p": 0.1,        # Focus the model's attention
                    "num_ctx": 4096      # Ensure context window is large enough for the image + output
                }
            )
            for chunk in stream:
                content = chunk['response']
                print(content, end='', flush=True) # This shows the "thinking" process
                full_response += content

            print("\n" + "-"*30)
            
            # Repair unescaped LaTeX backslashes before parsing
            # Models output \frac but JSON treats \f as formfeed — fix proactively
            import re as _re
            repaired = _re.sub(r'(?<!\\)\\([a-zA-Z])', lambda m: '\\\\' + m.group(1), full_response)

            json_data = json.loads(repaired)
            questions = json_data.get("questions", json_data) if isinstance(json_data, dict) else json_data
            if questions:
                output_path = os.path.join(output_dir, f"page_{page_num + 1}.json")
                with open(output_path, "w") as f:
                    json.dump({"questions": questions}, f, indent=2, ensure_ascii=False)
                print(f"✅ Saved {len(questions)} MCQs to {output_path}")
                
        except Exception as e:
            print(f"\n❌ Error on Page {page_num + 1}: {e}")

if __name__ == "__main__":
    main()
