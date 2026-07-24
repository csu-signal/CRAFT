# Chart Scripts
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

import pandas as pd
from sklearn.metrics import cohen_kappa_score
from scipy import stats
import random
df = pd.read_csv('/Users/hannahvanderhoeven/Documents/GitHub/CRAFT/humanValidation/craftValidationOutputs.csv')
plt.figure(figsize=(10, 6))

judge = "PS"
questions = []
if(judge == "SG"):
    questions = ["SG1", "SG2", "SG3", "SG4", "SG5", "SG6", "SG7"]
if(judge == "MM"):
    questions = ["MM1", "MM2", "MM3", "MM4", "MM5", "MM6", "MM7", "MM8"]
if(judge == "PS"):
    questions = ["PS1", "PS2", "PS3", "PS4", "PS5", "PS6"]

df = df.loc[df["Judge"] == judge]
one = df.loc[df["UserType"] == "llm"]
many = df.loc[df["UserType"] == "human"]
manyGrouped = many.groupby(['Question', 'Key'])['ResponseScore'].apply(list).reset_index()

# manyGrouped = many.groupby(['Question', 'Key']).agg({
#     'ResponseScore': 'mean'
# }).reset_index()

print(one)
result = one.groupby(['Question', 'Key']).agg({
    'ResponseScore': 'mean'
}).reset_index()
print(result)

statFormatted = result.merge(manyGrouped[['Question', 'Key', 'ResponseScore']], 
                   on=['Question', 'Key'], 
                   how='left')
statFormatted = statFormatted.rename(columns={'ResponseScore_x': 'oneAverage', 'ResponseScore_y': 'manyScores'})
print(statFormatted)

manyToOneData = []
for question in questions:
    sg = statFormatted.loc[statFormatted["Question"] == question]
    #print(sg)
    for row in sg.iterrows():
        manyScores = row[1]['manyScores']
        average = row[1]['oneAverage']
        if isinstance(manyScores, list):
            for x in manyScores:
                 manyToOneData.append([question, x + random.uniform(-0.1, 0.1), average + random.uniform(-0.1, 0.1)])
        else:
             manyToOneData.append([question, manyScores + random.uniform(-0.1, 0.1), average + random.uniform(-0.1, 0.1)])

manyToOneData_df = pd.DataFrame(manyToOneData, columns=["Question", "many", "one"])
sns.scatterplot(data=manyToOneData_df, x='many', y='one', hue='Question')
#sns.scatterplot(data=df_pivoted, x='human', y='llm', hue='Question')

plt.title(f'{judge} Judge Scores', fontsize=24, fontweight='bold', pad=20)
plt.ylabel('LLM Average Score', fontsize=16)
plt.xlabel('Human Average Score', fontsize=16)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=16)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.xlim(-0.2, 1.2)
plt.ylim(-0.2, 1.2)
plt.tight_layout()
plt.show()


