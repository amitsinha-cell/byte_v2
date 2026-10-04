#!/usr/bin/env python3
"""
Generate Socratic C++ tutor data with Gemma 4 through the Gemini API (Google AI Studio free tier).
No extra packages needed (standard library only), so it works in Termux, Colab or a laptop.

Setup:
  1. Get a key at https://aistudio.google.com/apikey
  2. export GEMINI_API_KEY="your-key"
  3. Keep gemma4_teacher_prompt.md in the same folder as this script.

Usage:
  python generate_data.py --dry-run          # show the batch plan, no API calls
  python generate_data.py --limit 2          # test run: only the first 2 batches
  python generate_data.py                    # full run (safe to stop and re-run: it resumes)
  python generate_data.py --only debug       # only one conversation type
  python generate_data.py --model gemma-4-26b-a4b-it --batch-size 8 --delay 8

Output: teacher_raw.jsonl (append-only). Then run filter_dataset.py on it.
"""
import argparse, json, math, os, re, sys, time, urllib.request, urllib.error

HERE = os.path.dirname(os.path.abspath(__file__))
PROMPT_FILE = os.path.join(HERE, "gemma4_teacher_prompt.md")
RAW = os.path.join(HERE, "teacher_raw.jsonl")
JUNK = os.path.join(HERE, "teacher_junk.txt")
PROGRESS = os.path.join(HERE, "teacher_progress.json")

# type -> how many to generate (about 25% over the number we want to keep)
PLAN = {"explain": 70, "debug": 77, "ladder": 57, "redirect": 60, "review": 35, "quiz": 28, "edge": 16}
TOPICS = ["variables & I/O", "conditionals & operators", "loops", "functions & recursion",
          "arrays & vectors", "strings", "pointers, references & memory", "classes & objects",
          "inheritance & polymorphism",
          "STL containers, algorithms & libraries (<algorithm>, <map>, <cmath>, <random>)"]
VARIETY = {
    "explain": ["use a real-life analogy (kitchen, school, game)", "ask the student to predict output", "mention a relevant header file", "use a very short example"],
    "debug": ["bugs that compile but behave wrongly", "off-by-one and boundary mistakes", "wrong operator or type mistakes", "scope or initialization mistakes", "occasionally a syntax or access error"],
    "ladder": ["student attempt is wrong the first time", "student attempt is right", "student goes straight to /solution after /hint", "student asks a follow-up before /solution"],
    "redirect": ["algorithm requests (search, sort, reverse)", "small program requests (calculator, grade checker)", "game-style requests (guess the number, dice)", "requests using a specific library"],
    "review": ["working code that could be cleaner", "code with a safety issue", "code with a naming or structure issue", "code with an efficiency issue"],
    "quiz": ["output-tracing question", "spot-the-bug question", "student answers wrong and tutor guides", "student answers right and tutor asks a follow-up"],
    "edge": ["correct code under /debug", "frustrated student", "student types /solution immediately", "non-C++ or off-topic request"],
}

def load_system_text():
    md = open(PROMPT_FILE, encoding="utf-8").read()
    m = re.search(r"## Part 1: System instructions\s*~~~~\n(.*?)\n~~~~", md, re.S)
    if not m:
        sys.exit("Could not find Part 1 in gemma4_teacher_prompt.md")
    return m.group(1).strip()

def build_batches(only, batch_size):
    batches = []
    for typ, total in PLAN.items():
        if only and typ != only:
            continue
        for b in range(math.ceil(total / batch_size)):
            n = min(batch_size, total - b * batch_size)
            batches.append({
                "id": f"{typ}|{b}", "type": typ, "n": n,
                "topic": TOPICS[b % len(TOPICS)],
                "variety": VARIETY[typ][b % len(VARIETY[typ])],
                "num": b + 1,
            })
    return batches

def user_message(b):
    return (f"Generate {b['n']} examples.\nTYPE: {b['type']}\nTOPIC: {b['topic']}\n"
            f"EXTRA VARIETY: {b['variety']}. This is batch {b['num']} of this type, so use fresh snippets, "
            f"unusual variable names and different scenarios from earlier batches.\nOutput JSON Lines only.")

def call_api(model, key, system_text, user_text, temperature, use_system=True):
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    if use_system:
        body = {"systemInstruction": {"parts": [{"text": system_text}]},
                "contents": [{"role": "user", "parts": [{"text": user_text}]}]}
    else:
        body = {"contents": [{"role": "user", "parts": [{"text": system_text + "\n\n---\n\n" + user_text}]}]}
    body["generationConfig"] = {"temperature": temperature, "maxOutputTokens": 8192}
    req = urllib.request.Request(url, data=json.dumps(body).encode(), method="POST",
                                 headers={"Content-Type": "application/json", "x-goog-api-key": key})
    with urllib.request.urlopen(req, timeout=300) as r:
        data = json.loads(r.read())
    parts = data["candidates"][0]["content"]["parts"]
    return "".join(p.get("text", "") for p in parts if not p.get("thought"))

def call_with_retries(model, key, system_text, user_text, temperature, state):
    waits = [20, 60, 120, 240]
    for attempt in range(len(waits) + 1):
        try:
            return call_api(model, key, system_text, user_text, temperature, state["use_system"])
        except urllib.error.HTTPError as e:
            msg = e.read().decode(errors="ignore")
            if e.code == 400 and state["use_system"] and "instruction" in msg.lower():
                print("  system instructions not accepted by this model, folding them into the message")
                state["use_system"] = False
                continue
            if e.code in (429, 500, 503) and attempt < len(waits):
                if "PerDay" in msg or "per day" in msg.lower():
                    sys.exit("Daily quota reached. Re-run tomorrow; progress is saved.")
                print(f"  HTTP {e.code}, waiting {waits[attempt]}s and retrying")
                time.sleep(waits[attempt]); continue
            sys.exit(f"API error {e.code}: {msg[:400]}")
        except (urllib.error.URLError, TimeoutError) as e:
            if attempt < len(waits):
                print(f"  network problem ({e}), waiting {waits[attempt]}s"); time.sleep(waits[attempt]); continue
            sys.exit(f"Network error: {e}")

def extract_lines(text):
    good, junk = [], []
    for line in text.splitlines():
        s = line.strip()
        if not s or s.startswith("```"):
            continue
        if s.startswith("{"):
            try:
                obj = json.loads(s)
                if isinstance(obj.get("messages"), list):
                    good.append(json.dumps(obj, ensure_ascii=False)); continue
            except Exception:
                pass
        junk.append(s)
    return good, junk

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gemma-4-31b-it")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--delay", type=float, default=6.0, help="seconds between calls")
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--only", choices=list(PLAN))
    ap.add_argument("--limit", type=int, help="stop after this many batches (for testing)")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    batches = build_batches(a.only, a.batch_size)
    done = set(json.load(open(PROGRESS))) if os.path.exists(PROGRESS) else set()
    todo = [b for b in batches if b["id"] not in done]
    if a.limit:
        todo = todo[:a.limit]
    print(f"{len(batches)} batches in plan, {len(done)} done, running {len(todo)}")
    if a.dry_run:
        for b in todo:
            print(f"  {b['id']:12} n={b['n']}  {b['topic']}  | {b['variety']}")
        return

    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        sys.exit("Set your key first:  export GEMINI_API_KEY='your-key'")
    system_text = load_system_text()
    state = {"use_system": True}
    total = 0
    for i, b in enumerate(todo, 1):
        print(f"[{i}/{len(todo)}] {b['id']} ({b['topic']})")
        text = call_with_retries(a.model, key, system_text, user_message(b), a.temperature, state)
        good, junk = extract_lines(text)
        with open(RAW, "a", encoding="utf-8") as f:
            for g in good: f.write(g + "\n")
        if junk:
            with open(JUNK, "a", encoding="utf-8") as f:
                f.write(f"--- {b['id']}\n" + "\n".join(junk) + "\n")
        done.add(b["id"])
        json.dump(sorted(done), open(PROGRESS, "w"))
        total += len(good)
        print(f"  got {len(good)} examples ({len(junk)} stray lines)")
        time.sleep(a.delay)
    print(f"Finished. {total} new examples appended to {RAW}")

if __name__ == "__main__":
    main()
