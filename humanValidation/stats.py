import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

import pandas as pd
from sklearn.metrics import cohen_kappa_score
from scipy import stats
df = pd.read_csv('/Users/hannahvanderhoeven/Documents/GitHub/CRAFT/humanValidation/craftValidationOutputs.csv')

result = df[df['Question'] == 'MM5']
result = result.groupby(['Key', "UserType"]).agg({
    'ResponseScore': 'mean',
})

# 1. Reshape so 'human' and 'llm' are separate columns
# (assuming 'df' is your current multi-indexed DataFrame/Series)
df_unstacked = result.unstack(level=-1)
diff = (df_unstacked[('ResponseScore', 'human')] - df_unstacked[('ResponseScore', 'llm')]).abs()
mismatches = df_unstacked[diff > 0.333334]
print(mismatches)

judge = "PS"
questions = []
if(judge == "SG"):
    questions = ["SG1", "SG2", "SG3", "SG4", "SG5", "SG6", "SG7"]
if(judge == "MM"):
    questions = ["MM1", "MM2", "MM3", "MM4", "MM5", "MM6", "MM7", "MM8"]
if(judge == "PS"):
    questions = ["PS1", "PS2", "PS3", "PS4", "PS5", "PS6"]

df = df.loc[df["Judge"] == judge]
one = df.loc[df["UserType"] == "human"]
many = df.loc[df["UserType"] == "llm"]

# true many
manyGrouped = many.groupby(['Question', 'Key'])['ResponseScore'].apply(list).reset_index()

#average both?
# manyGrouped = many.groupby(['Question', 'Key']).agg({
#     'ResponseScore': 'mean'
# }).reset_index()

# many = many[many['Question'] != 'PS4']
# questionAverageGrader = many.agg({
#     'ResponseScore': 'mean'
# }).reset_index()
# print(questionAverageGrader)

# questionAverage = many.groupby(['Question']).agg({
#     'ResponseScore': 'mean'
# }).reset_index()
# print(questionAverage)

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

manyScoresCrossKeysAllQuestions = []
oneScoresCrossKeysAllQuestions = []
for question in questions:
    sg = statFormatted.loc[statFormatted["Question"] == question]
    #print(sg)
    manyScoresCrossKeys = []
    oneScoresCrossKeys = []
    for row in sg.iterrows():
        manyScores = row[1]['manyScores']
        average = row[1]['oneAverage']
        if isinstance(manyScores, list):
            for x in manyScores:
                manyScoresCrossKeys.append(x)
                manyScoresCrossKeysAllQuestions.append(x)
                oneScoresCrossKeys.append(average)
                oneScoresCrossKeysAllQuestions.append(average)
        else:
            manyScoresCrossKeys.append(manyScores)
            oneScoresCrossKeys.append(average)
            manyScoresCrossKeysAllQuestions.append(manyScores)
            oneScoresCrossKeysAllQuestions.append(average)
    #print(len(manyScoresCrossKeys))
    # print(f'Many: {manyScoresCrossKeys}')
    # print(f'One: {oneScoresCrossKeys}')

    # 1. Pearson Correlation (Linear)
    # Returns: (correlation_coefficient, p_value)
    pearson_r, p_val_pearson = stats.pearsonr(manyScoresCrossKeys, oneScoresCrossKeys)

    # 2. Spearman Correlation (Monotonic/Rank-based)
    # Returns: (correlation_coefficient, p_value)
    spearman_rho, p_val_spearman = stats.spearmanr(manyScoresCrossKeys, oneScoresCrossKeys)

    print(f"{question} Pearson r: {pearson_r:.4f}, p-value: {p_val_pearson:.4g}")
    print(f"{question} Spearman rho: {spearman_rho:.4f}, p-value: {p_val_spearman:.4g}\n")

# 1. Pearson Correlation (Linear)
# Returns: (correlation_coefficient, p_value)
pearson_r, p_val_pearson = stats.pearsonr(manyScoresCrossKeysAllQuestions, oneScoresCrossKeysAllQuestions)

# 2. Spearman Correlation (Monotonic/Rank-based)
# Returns: (correlation_coefficient, p_value)
spearman_rho, p_val_spearman = stats.spearmanr(manyScoresCrossKeysAllQuestions, oneScoresCrossKeysAllQuestions)

print(f"{judge} All Questions Pearson r: {pearson_r:.4f}, p-value: {p_val_pearson:.4g}")
print(f"{judge} All Questions Spearman rho: {spearman_rho:.4f}, p-value: {p_val_spearman:.4g}\n")


