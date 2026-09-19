from enum import Enum
import json
from pathlib import Path
import os
from datasets import load_dataset
from matplotlib import pyplot as plt
import pandas as pd

if __name__ == "__main__":
    validationPath = Path("/home/hannah/CRAFT/CRAFT/gittenExperiments/counterfactuals/divergenceData_policy_seed42_kl0.1_closs0.01_nosamp_hpt0.4")
    validationData = []
    ds = load_dataset("Abhijnan/craft-benchmark-lean")

    for file_path in validationPath.glob("*.json"):
        with open(file_path, "r", encoding="utf-8") as file:
            data = json.load(file)   
            for d in data: 
                d["modelCombo"] =  file_path.name.split("_")[0] 
            validationData.extend(data)

    validationSatP = 0
    validationSatF = 0
    validationTurnWiseP = {}
    validationTurnWiseF = {}
    validationModelWise = {}
    for turn in validationData:   
        validationSatP += turn["satisfaction_partial"]
        validationSatF += turn["satisfaction_full"]

        turnKey = turn["turn"]
        modelCombo = turn["modelCombo"].split("+")[0].strip().lower()
        satP = turn["satisfaction_partial"]
        satF = turn["satisfaction_full"]

        if turnKey in validationTurnWiseP and isinstance(validationTurnWiseP[turnKey], list):
            validationTurnWiseP[turnKey].append(satP)
            validationTurnWiseF[turnKey].append(satF)
        else:
            validationTurnWiseP[turnKey] = [satP]
            validationTurnWiseF[turnKey] = [satF]

        if modelCombo in validationModelWise and isinstance(validationModelWise[modelCombo], list):
            validationModelWise[modelCombo].append(satP)
        else:
            validationModelWise[modelCombo] = [satP]

    print(f"Count Validation: {len(validationData)}")
    if(len(validationData) != 0):
        print(f"Average Validation Partial Satisfaction: {round(validationSatP / len(validationData), 3)}\n")
        print(f"Average Validation Full Satisfaction: {round(validationSatF / len(validationData), 3)}\n")
        validationTurnWiseP = dict(sorted(validationTurnWiseP.items()))
        validationTurnWiseF = dict(sorted(validationTurnWiseF.items()))
        valTurnAveragesP = {
            key: (round(sum(values) / len(values), 3)) if values else 0.0 
            for key, values in validationTurnWiseP.items()
        }

        valTurnAveragesF = {
                    key: (round(sum(values) / len(values), 3)) if values else 0.0 
                    for key, values in validationTurnWiseF.items()
                }

        print(valTurnAveragesP)
        print(valTurnAveragesF)
        print("\n")

        valModelWiseAverages = {
            key: (round(sum(values) / len(values), 3)) if values else 0.0 
            for key, values in validationModelWise.items()
        }

        print(valModelWiseAverages)

    #########################################################

    if(len(validationData) != 0):
        plt.plot(valTurnAveragesP.keys(), valTurnAveragesP.values(), label='Partial IOU', marker='o', color='blue') 
        plt.plot(valTurnAveragesF.keys(), valTurnAveragesF.values(), label='Full IOU', marker='o', color='orange') 

    plt.title('Average Satisfaction over Turns, Counterfactual, Split 42')
    plt.xlabel('Turn')
    plt.ylabel('Average Satisfaction')
    plt.legend()
    plt.ylim(0, 0.6)
    plt.grid(True)
    plt.savefig("counter.png")

    test = 0