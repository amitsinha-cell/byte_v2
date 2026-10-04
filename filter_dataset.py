#!/usr/bin/env python3
"""
Filter and validate teacher-generated Socratic C++ tutor data.

Usage:
  python filter_dataset.py teacher_raw.jsonl            # rule checks only
  python filter_dataset.py teacher_raw.jsonl --compile  # also syntax-check code (needs g++ or clang++)

Outputs (in the same folder):
  cpp_socratic_clean.jsonl   passed everything, system prompt added
  needs_review.jsonl         passed rules but code did not compile (check by hand)
  rejected.jsonl             failed a rule, with a "reason" field
"""
import json, re, sys, shutil, subprocess, tempfile, os

SYS = ("You are Byte, a patient C++ tutor for beginners. Teach by guiding, not by giving answers. "
"When giving feedback on student code, open with something they got right. Ask exactly one guiding question, "
"and keep replies under about 100 words with at most one short code block. "
"With no command or /explain: give an analogy, a tiny example and a question. "
"/debug: hint 1 only. Point to where to look and never state the fix. "
"/hint: a more specific hint, still not the full fix. "
"/solution: always give the fix, explain why it works, and add one practice question. "
"/review: praise one thing, name the single most important issue, and ask the student to fix it. "
"/quiz: ask one question and wait for the answer. "
"If the code is correct, say so and ask a what-if question. Never invent bugs.")

FENCE = re.compile(r"```.*?```", re.S)
HEADERS = ("iostream string vector map unordered_map set algorithm numeric cmath memory "
           "cstdlib ctime random stdexcept cctype iomanip functional limits").split()
PRELUDE = "".join(f"#include <{h}>\n" for h in HEADERS) + "using namespace std;\n"

def prose(text):
    return FENCE.sub(" ", text)

def check(msgs):
    """Return a rejection reason or None."""
    if not msgs or msgs[0]["role"] != "user" or msgs[-1]["role"] != "assistant":
        return "must start with user and end with assistant"
    for i, m in enumerate(msgs):
        want = "user" if i % 2 == 0 else "assistant"
        if m["role"] != want or not isinstance(m.get("content"), str) or not m["content"].strip():
            return "roles must alternate user/assistant with non-empty text"
    for i in range(0, len(msgs) - 1, 2):
        u, a = msgs[i]["content"].strip(), msgs[i + 1]["content"]
        cmd = u.split()[0].lower() if u.startswith("/") else ""
        p = prose(a)
        words = len(p.split())
        limit = 150 if cmd == "/solution" else 110
        if words > limit:
            return f"turn {i//2+1}: reply too long ({words} words)"
        if p.count("?") != 1:
            return f"turn {i//2+1}: needs exactly one question mark outside code (found {p.count('?')})"
        blocks = FENCE.findall(a)
        if len(blocks) > 1:
            return f"turn {i//2+1}: more than one code block"
        if blocks and cmd != "/solution":
            lines = [l for l in blocks[0].splitlines() if l.strip() and not l.strip().startswith("```")]
            if len(lines) > 8:
                return f"turn {i//2+1}: code block longer than 8 lines"
        if cmd in ("/debug", "/hint", "/review") and blocks:
            return f"turn {i//2+1}: {cmd} reply must not show code (would leak the fix)"
    return None

def extract_code(msgs):
    """Student code from the first user message (everything after the command line)."""
    u = msgs[0]["content"].strip()
    if not u.startswith("/"):
        return None
    parts = u.split("\n", 1)
    if parts[0].split()[0].lower() not in ("/debug", "/review") or len(parts) < 2:
        return None
    code = parts[1]
    code = re.sub(r"^\(.*\)$", "", code, flags=re.M)      # drop "(I call ... )" notes
    code = re.sub(r"```(?:cpp|c\+\+)?", "", code)
    return code.strip() or None

def compiles(code):
    cxx = shutil.which("g++") or shutil.which("clang++")
    if not cxx:
        return True
    variants = [PRELUDE + code + "\n",
                PRELUDE + "int main() {\n" + code + "\n}\n"]
    if "int main" in code:
        variants = [code + "\n", PRELUDE + code + "\n"]
    for src in variants:
        with tempfile.NamedTemporaryFile("w", suffix=".cpp", delete=False) as f:
            f.write(src); name = f.name
        try:
            r = subprocess.run([cxx, "-std=c++17", "-fsyntax-only", "-Wno-everything", name],
                               capture_output=True, text=True, timeout=30)
            if r.returncode == 0:
                return True
        finally:
            os.unlink(name)
    return False

def main():
    if len(sys.argv) < 2:
        print(__doc__); return
    src, do_compile = sys.argv[1], "--compile" in sys.argv
    out = os.path.dirname(os.path.abspath(src))
    clean, review, rejected, seen = [], [], [], set()
    for ln, line in enumerate(open(src, encoding="utf-8"), 1):
        line = line.strip()
        if not line or line.startswith("```"):
            continue
        try:
            obj = json.loads(line)
            msgs = obj["messages"]
            if msgs and msgs[0]["role"] == "system":
                msgs = msgs[1:]
        except Exception as e:
            rejected.append({"line": ln, "reason": f"bad JSON: {e}", "raw": line[:200]}); continue
        reason = check(msgs)
        if reason is None:
            key = re.sub(r"\s+", " ", (msgs[0]["content"] + msgs[1]["content"]).lower()).strip()
            if key in seen: reason = "duplicate first message"
            seen.add(key)
        if reason:
            rejected.append({"line": ln, "reason": reason, "messages": msgs}); continue
        final = {"messages": [{"role": "system", "content": SYS}] + msgs}
        code = extract_code(msgs) if do_compile else None
        if code and not compiles(code):
            review.append(final)
        else:
            clean.append(final)
    for name, rows in (("cpp_socratic_clean.jsonl", clean), ("needs_review.jsonl", review), ("rejected.jsonl", rejected)):
        with open(os.path.join(out, name), "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"clean={len(clean)}  needs_review={len(review)}  rejected={len(rejected)}")
    for r in rejected[:10]:
        print("  rejected:", r.get("line"), r["reason"])

if __name__ == "__main__":
    main()
