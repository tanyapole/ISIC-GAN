#%%
from xlrd import open_workbook
from xlrd.sheet import Sheet
import pandas as pd


def wrap(a: any) -> Sheet:
    return a


#%%
name = "./_4.xlsx"

df = open_workbook(name)
#%%

works = wrap(df.sheets()[0])
#%%
import datetime
#%%
rows = works.row_values(13)
initial_date = datetime.datetime(2020, 2, 1)
s = rows.index("Февраль")
e = rows.index(44228.0)
index_to_day_map = {}
for d in range(s, e):
    initial_date = initial_date + datetime.timedelta(days=1)
    index_to_day_map[d] = initial_date

#%%

plan_rows = []
fact_rows = []

for r in range(21, 1520):
    sub_row = works.row_values(r, 0, 100)
    if "План" in sub_row:
        plan_rows.append(r)
    if "Факт" in sub_row:
        fact_rows.append(r)

plan_fact_rows = list(zip(plan_rows, fact_rows))
#%%

upper_works = []
for idx,(plan, fact) in enumerate(plan_fact_rows):
    values_plan = works.row_values(plan)
    values_fact = works.row_values(fact)
    core = {
        "work_title": values_plan[1],
        "work_id": idx,
        "measurements": values_plan[2],
        "amount": values_plan[3],
        "work_data": {
            "start_date": {
                "plan": values_plan[4],
                "estimate": values_plan[4],
                "fact": values_plan[4]
            },
            "stop_date": {
                "plan": values_plan[5],
                "estimate": values_plan[5],
                "fact": values_plan[5]
            },
            "complete_state": {
                "plan": "in " + str(values_plan[8]),
                "fact": "in " + str(values_fact[8]),
            },
            "current_remain" : "in " + str((values_plan[8] if values_plan[8] != '' else 0) - (values_fact[8] if values_fact[8] != '' else 0)),
            "while_remain": "in " + str((values_plan[8] if values_plan[8] != '' else 0) - (values_fact[8] if values_fact[8] != '' else 0)),
            "mounth_complite": {
                "plan": "in " + str(values_plan[8]),
                "fact": "in " + str(values_fact[8]),
            },
            "progress": [
                {
                    "day_id": {
                        "plan": "in " + str(values_plan[d] / (values_plan[8] if values_plan[8] != '' else 1)) if values_plan[d] != '' else 0,
                        "fact": "in " + str(values_fact[d] / (values_fact[8] if values_fact[8] != '' else 1)) if values_fact[d] != '' else 0
                    }
                } for d in index_to_day_map.keys()
            ]
        },
        "middle_work": {}
            }

    upper_works.append(core)
#%%
resources = wrap(df.sheets()[1])
#%%
equipments = []
for idx, plan in enumerate(range(4, 106, 1)):
    plan_row = resources.row_values(plan)
    fact = plan + 1
    fact_row = resources.row_values(fact)
    if "план" not in plan_row:
        continue
    core = {
        "equipment_id": idx,
        "equipment_name": plan_row[2],
        "equipment_number": plan_row[0],
        "driver_last_first_name": "",
        "work": {
            "hours": sum(filter(lambda x: x is float, plan_row[8:])),
            "work_type": ""
        },
        "break_hours": 0,
        "repairement" : {
            "hours": sum(filter(lambda x: x is float, fact_row[8:])),
            "work_type": ""
        },
        "progress": [

            {"day_presentations": {
                "plan": plan_row[8 + idx],
                "fact": fact_row[8 + idx]
                }
            } for idx in range(len(plan_row[8:]))

        ]
    }
    equipments.append(core)

# 4, 5
#%%
positions = []
for idx, pos in enumerate(range(1575, 1650)):
    if "План" not in works.row_values(pos):
        continue
    plan_row = works.row_values(pos)
    fact_row = works.row_values(pos + 1)
    next_idx = plan_row.index("План") + 2
    core = {
        "position_id": idx,
        "position_name": plan_row[5],
        "progress": [
            {
                "per_day_count": {
                    "plan": plan_row[d],
                    "fact": fact_row[d]
                }
            } for d in range(next_idx, len(plan_row))
        ]
    }
    positions.append(core)

#%%
import json
result = {
    "file_name": name,
    "upper_works": upper_works,
    "equipments": equipments,
    "positions": positions
}

jsn = json.dumps(result, indent=4)
print(jsn)


if __name__ == "__main__":
    pass