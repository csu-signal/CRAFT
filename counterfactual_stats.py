from enum import Enum
import json
from pathlib import Path
import os
from datasets import load_dataset

if __name__ == "__main__":
    validationPath = Path("/home/hannah/CRAFT/CRAFT/divergenceData/validation")
    trainPath = Path("/home/hannah/CRAFT/CRAFT/divergenceData/train")
    validationData = []
    trainData = []
    fullData = []
    ds = load_dataset("Abhijnan/craft-benchmark-lean")

    for file_path in validationPath.glob("*.json"):
        with open(file_path, "r", encoding="utf-8") as file:
            data = json.load(file)   
            validationData.extend(data)
            fullData.extend(data)

    for file_path in trainPath.glob("*.json"):
        with open(file_path, "r", encoding="utf-8") as file:
            data = json.load(file)   
            trainData.extend(data)
            fullData.extend(data)

    validationSat = 0
    for turn in validationData:   
        validationSat += turn[turn["builderSelected"]]["satisfaction"]

    print(f"Count Validation: {len(validationData)}")
    print(f"Average Validation Satisfaction: {round(validationSat / len(validationData), 3)}\n")

    trainSat = 0
    for turn in trainData:   
        trainSat += turn[turn["builderSelected"]]["satisfaction"]

    print(f"Count Train: {len(trainData)}")
    print(f"Average Train Satisfaction: {round(trainSat / len(trainData), 3)}\n")


    turnWise = {}
    modelWise = {}
    for turn in fullData:  
        turnKey = turn["turn"]
        modelCombo = turn[turn["builderSelected"]]["modelCombo"].split("+")[0].strip().lower()
        sat = turn[turn["builderSelected"]]["satisfaction"]

        if turnKey in turnWise and isinstance(turnWise[turnKey], list):
            turnWise[turnKey].append(sat)
        else:
            turnWise[turnKey] = [sat]

        if modelCombo in modelWise and isinstance(modelWise[modelCombo], list):
            modelWise[modelCombo].append(sat)
        else:
            modelWise[modelCombo] = [sat]

    turnWise = dict(sorted(turnWise.items()))
    averages = {
        key: (round(sum(values) / len(values), 3)) if values else 0.0 
        for key, values in turnWise.items()
    }

    print(averages)
    # create plot turn on x, average sat (train and val) on y
    print("\n")

    averages = {
        key: (round(sum(values) / len(values), 3)) if values else 0.0 
        for key, values in modelWise.items()
    }

    print(averages)