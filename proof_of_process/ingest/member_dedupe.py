import re
from unidecode import unidecode
from difflib import SequenceMatcher

def _canon(s: str) -> str:
    s = unidecode((s or "").strip().lower())
    s = re.sub(r"[^a-z0-9\s]", "", s)
    s = re.sub(r"\s+", " ", s)
    return s

def dedupe_names(names, threshold=0.92):
    canonical = {}
    clusters = []
    for n in names:
        if not n: continue
        cn = _canon(n)
        matched = False
        for c in clusters:
            if SequenceMatcher(None, cn, c["key"]).ratio() >= threshold:
                c["raw"].add(n); matched=True; break
        if not matched:
            clusters.append({"key": cn, "raw": set([n])})
    for c in clusters:
        rep = sorted(c["raw"], key=lambda x: (-len(x), x))[0]
        for r in c["raw"]:
            canonical[r] = rep
    return canonical

def split_assignees(s: str):
    parts = re.split(r"[;,/|]", (s or ""))
    parts = [p.strip() for p in parts if p.strip()]
    return parts
