import pandas as pd, numpy as np, random

PROJECTS = ["Apollo","Zephyr"]
MEMBERS = ["Alice Nguyen","A. Nguyen","Bob Tran","Charlie Pham","Duc Le","Minh Vo"]
STATUSES = ["Not Started","In Progress","Completed"]
SKILLS = ["backend api", "database sql", "react ui", "android kotlin", "devops docker", "ml nlp", "ux design", "scrum sprint"]

def make(n_tasks=200, seed=42):
    random.seed(seed); np.random.seed(seed)
    rows = []
    for i in range(n_tasks):
        prj = random.choice(PROJECTS)
        desc = f"Task {i}: {random.choice(SKILLS)}; build/bugfix/feature"
        ass = random.sample(MEMBERS, k=random.choices([1,2,3],[0.7,0.25,0.05])[0])
        status = random.choices(STATUSES,[0.2,0.5,0.3])[0]
        week = pd.Timestamp.today().to_period("W").start_time - pd.Timedelta(weeks=random.randint(0,20))
        plan = max(1.0, np.random.gamma(2.0, 2.0))
        actual = plan*np.random.uniform(0.7,1.3) if status=="Completed" else 0.0
        cost = actual * np.random.uniform(5,15)
        rows.append([f"T{i}", prj, desc, "; ".join(ass), status, week, plan, actual, cost])
    return pd.DataFrame(rows, columns=["Task","Project","Description","Assignees","Status","Week","Planned_Effort","Actual_Effort","Cost"])
