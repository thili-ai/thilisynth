"""lfsynth de-risking: can a small local model render a freight load tender?

Two questions that decide the drill's default model:
  A. RENDER  — turn a structured LoadTender record into a messy forwarded email thread
               with a planted defect. (lesson 5)
  B. SCHEMA  — honour a JSON schema via Ollama's `format` param. (lesson 4)
"""
import json
import time
import urllib.request

OLLAMA = "http://localhost:11434/api/generate"
MODELS = ["qwen2.5:1.5b", "qwen2.5:7b"]

RECORD = {
    "load_id": "TNDR-48812",
    "shipper": "Ashland Foods LLC",
    "origin": "Fresno, CA",
    "destination": "Reno, NV",
    "pickup_window": "2026-08-11 08:00-12:00 PT",
    "equipment": "53ft reefer",
    "temp_f": 34,
    "weight_lb": 41200,
    "commodity": "Fresh lettuce, palletized",
    "pieces": 22,
    "rate_offered_usd": 1850,
}

# The planted defect: the pickup window must appear ONLY as a vague relative phrase.
RENDER_PROMPT = f"""You write realistic freight logistics emails.

Here is a load tender record:
{json.dumps(RECORD, indent=2)}

Write the email thread a broker would actually receive for this load. Requirements:
- It must be a FORWARDED thread with 2 replies (so 3 messages total, newest at top).
- Include realistic email headers (From/Sent/To/Subject) between messages.
- DEFECT TO PLANT: never state the pickup date or time explicitly. Refer to it only
  vaguely, the way a busy dispatcher would (e.g. "early next week, AM if you can").
- Bury the temperature requirement in a reply, not the first message.
- One message should have a typo or a dropped word. Do not mark it.
- No preamble, no explanation. Output the email thread only.
"""

SCHEMA = {
    "type": "object",
    "properties": {
        "load_id": {"type": "string"},
        "origin_city": {"type": "string"},
        "destination_city": {"type": "string"},
        "equipment": {"type": "string"},
        "weight_lb": {"type": "integer"},
        "pickup_window_stated": {
            "type": "boolean",
            "description": "true only if an explicit date or time appears",
        },
        "temp_required_f": {"type": ["integer", "null"]},
    },
    "required": [
        "load_id", "origin_city", "destination_city", "equipment",
        "weight_lb", "pickup_window_stated", "temp_required_f",
    ],
}


def call(model, prompt, fmt=None, num_predict=900):
    body = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.8, "num_predict": num_predict, "seed": 7},
    }
    if fmt:
        body["format"] = fmt
        body["options"]["temperature"] = 0.0
    req = urllib.request.Request(
        OLLAMA, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}
    )
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=600) as r:
        out = json.loads(r.read())
    elapsed = time.time() - t0
    tok = out.get("eval_count", 0)
    return out["response"], elapsed, tok


for model in MODELS:
    print("=" * 78)
    print(f"MODEL: {model}")
    print("=" * 78)

    text, secs, tok = call(model, RENDER_PROMPT)
    print(f"\n--- A. RENDER  ({secs:.1f}s, {tok} tok, {tok/max(secs,.01):.1f} tok/s) ---\n")
    print(text.strip())

    extract_prompt = (
        "Extract the load tender fields from this email thread. "
        "Set pickup_window_stated to true ONLY if an explicit date or clock time appears.\n\n"
        + text
    )
    js, secs2, tok2 = call(model, extract_prompt, fmt=SCHEMA, num_predict=400)
    print(f"\n--- B. SCHEMA  ({secs2:.1f}s, {tok2} tok) ---")
    try:
        parsed = json.loads(js)
        print("valid JSON:", json.dumps(parsed, indent=2))
        missing = [k for k in SCHEMA["required"] if k not in parsed]
        print("missing required fields:", missing or "none")
    except Exception as e:
        print("INVALID JSON:", e)
        print(js[:500])
    print()
