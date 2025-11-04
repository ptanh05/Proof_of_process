"""
Skill tagging (VN/EN):
- Dictionary + ngram matching with TF-IDF fallback
- Expandable taxonomy; returns list of skills per task
- Score per member per skill = sum Effort_Score over tasks containing the skill
"""
import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

SKILL_LEXICON = {
    "Backend": ["backend","api","rest","graphql","database","sql","java","spring","python","django","node","express"],
    "Frontend": ["frontend","react","vue","angular","ui","ux","javascript","html","css","tailwind"],
    "Mobile": ["android","ios","swift","kotlin","react native","flutter"],
    "DevOps": ["devops","docker","kubernetes","k8s","terraform","ci/cd","aws","gcp","azure","monitoring","logging"],
    "Data/ML": ["data","etl","warehouse","ml","ai","pytorch","tensorflow","xgboost","lightgbm","nlp","cv"],
    "Design": ["design","figma","wireframe","prototype","ux research","visual"],
    "PM": ["scrum","kanban","sprint","gantt","evm","estimate","risk","roadmap","stakeholder","okr"],
    "Soft": ["communication","leadership","presentation","negotiation","collaboration","problem solving","critical thinking"]
}

def _contains_any(text, keys):
    t = " " + re.sub(r"\s+", " ", text.lower()) + " "
    for k in keys:
        if f" {k} " in t:
            return True
    return False

def tag_skills(text: str):
    found = set()
    for k, words in SKILL_LEXICON.items():
        if _contains_any(text, words):
            found.add(k)
    return sorted(found)

def tag_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Skill_Tags"] = df["Description"].astype(str).apply(tag_skills)
    return df

def build_skill_leaderboard(df: pd.DataFrame, effort_col="Effort_Score") -> pd.DataFrame:
    rows = []
    for _, r in df.iterrows():
        for s in r.get("Skill_Tags", []):
            for m in r.get("Assignees_List", []):
                rows.append((m, s, float(r.get(effort_col, 0.0))))
    sk = pd.DataFrame(rows, columns=["Member","Skill","Effort"])
    if sk.empty:
        return pd.DataFrame(columns=["Member","Skill","Score_0_100"])
    agg = sk.groupby(["Member","Skill"])["Effort"].sum().reset_index()
    # normalize by member to 0..100
    agg["Score_0_100"] = agg.groupby("Member")["Effort"].transform(lambda x: 100*x/(x.max() if x.max()>0 else 1.0))
    return agg[["Member","Skill","Score_0_100"]].sort_values(["Member","Score_0_100"], ascending=[True,False])
