import re

ALIASES = {
    "Task": ["task","title","name","issue","card","subject"],
    "Project": ["project","board","repo","space"],
    "Description": ["description","desc","body","details","content"],
    "Assignees": ["assignees","assignee","owner","members","assigned_to"],
    "Status": ["status","state","stage","column","list"],
    "Week": ["week","date","due","deadline","created","updated"],
    "Planned_Effort": ["planned_effort","estimate","story_points","sp","planned"],
    "Actual_Effort": ["actual_effort","actual","time_spent","hours"],
    "Cost": ["cost","budget","expense","ac"]
}

def guess_column(target, columns):
    t = target.lower()
    for col in columns:
        if col.lower() == t:
            return col
    for alias in ALIASES.get(target, []):
        for col in columns:
            lc = col.lower()
            if re.fullmatch(alias, lc) or alias in lc:
                return col
    return None
