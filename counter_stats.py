from enum import Enum
import json
from pathlib import Path
import os
from datasets import load_dataset
from matplotlib import pyplot as plt
import pandas as pd

if __name__ == "__main__":
    validationPath = Path("/home/hannah/CRAFT/CRAFT/gittenExperiments/counterfactuals/validation")
    trainPath = Path("/home/hannah/CRAFT/CRAFT/gittenExperiments/counterfactuals/train")
    validationData = []
    trainData = []
    fullData = []
    ds = load_dataset("Abhijnan/craft-benchmark-lean")

    for file_path in validationPath.glob("*.json"):
        with open(file_path, "r", encoding="utf-8") as file:
            data = json.load(file)   
            for d in data: 
                d["modelCombo"] =  file_path.name.split("_")[0] 
            validationData.extend(data)
            fullData.extend(data)

    for file_path in trainPath.glob("*.json"):
        with open(file_path, "r", encoding="utf-8") as file:
            data = json.load(file) 
            for d in data: 
                d["modelCombo"] =  file_path.name.split("_")[0] 
            trainData.extend(data)
            fullData.extend(data)

    validationSat = 0
    validationTurnWise = {}
    validationModelWise = {}
    for turn in validationData:   
        validationSat += turn["satisfaction"]

        turnKey = turn["turn"]
        modelCombo = turn["modelCombo"].split("+")[0].strip().lower()
        sat = turn["satisfaction"]

        if turnKey in validationTurnWise and isinstance(validationTurnWise[turnKey], list):
            validationTurnWise[turnKey].append(sat)
        else:
            validationTurnWise[turnKey] = [sat]

        if modelCombo in validationModelWise and isinstance(validationModelWise[modelCombo], list):
            validationModelWise[modelCombo].append(sat)
        else:
            validationModelWise[modelCombo] = [sat]

    print(f"Count Validation: {len(validationData)}")
    if(len(validationData) != 0):
        print(f"Average Validation Satisfaction: {round(validationSat / len(validationData), 3)}\n")
        validationTurnWise = dict(sorted(validationTurnWise.items()))
        valTurnAverages = {
            key: (round(sum(values) / len(values), 3)) if values else 0.0 
            for key, values in validationTurnWise.items()
        }

        print(valTurnAverages)
        print("\n")

        valModelWiseAverages = {
            key: (round(sum(values) / len(values), 3)) if values else 0.0 
            for key, values in validationModelWise.items()
        }

        print(valModelWiseAverages)

    #########################################################

    trainSat = 0
    trainTurnWise = {}
    trainModelWise = {}
    for turn in trainData:   
        trainSat += turn["satisfaction"]

        turnKey = turn["turn"]
        modelCombo = turn["modelCombo"].split("+")[0].strip().lower()
        sat = turn["satisfaction"]

        if turnKey in trainTurnWise and isinstance(trainTurnWise[turnKey], list):
            trainTurnWise[turnKey].append(sat)
        else:
            trainTurnWise[turnKey] = [sat]

        if modelCombo in trainModelWise and isinstance(trainModelWise[modelCombo], list):
            trainModelWise[modelCombo].append(sat)
        else:
            trainModelWise[modelCombo] = [sat]

    print(f"\nCount Train: {len(trainData)}")
    if(len(trainData) != 0):
        print(f"Average Train Satisfaction: {round(trainSat / len(trainData), 3)}\n")
        trainTurnWise = dict(sorted(trainTurnWise.items()))
        trainTurnAverages = {
            key: (round(sum(values) / len(values), 3)) if values else 0.0 
            for key, values in trainTurnWise.items()
        }

        print(trainTurnAverages)
        print("\n")

        trainModelaverages = {
            key: (round(sum(values) / len(values), 3)) if values else 0.0 
            for key, values in trainModelWise.items()
        }
        print(trainModelaverages)

    #########################################################

    turnWise = {}
    modelWise = {}
    for turn in fullData:  
        turnKey = turn["turn"]
        modelCombo = turn["modelCombo"].split("+")[0].strip().lower()
        sat = turn["satisfaction"]

        if turnKey in turnWise and isinstance(turnWise[turnKey], list):
            turnWise[turnKey].append(sat)
        else:
            turnWise[turnKey] = [sat]

        if modelCombo in modelWise and isinstance(modelWise[modelCombo], list):
            modelWise[modelCombo].append(sat)
        else:
            modelWise[modelCombo] = [sat]

    print("\nFull Data Turnwise")
    turnWise = dict(sorted(turnWise.items()))
    averages = {
        key: (round(sum(values) / len(values), 3)) if values else 0.0 
        for key, values in turnWise.items()
    }

    print(averages)
    # create plot turn on x, average sat (train and val) on y
    print("\n")

    print("Full Data Modelwise")
    averages = {
        key: (round(sum(values) / len(values), 3)) if values else 0.0 
        for key, values in modelWise.items()
    }

    print(averages)

    if(len(validationData) != 0):
        plt.plot(valTurnAverages.keys(), valTurnAverages.values(), label='Train', marker='o', color='blue')
    if(len(trainData) != 0):
        plt.plot(trainTurnAverages.keys(), trainTurnAverages.values(), label='Validation', marker='s', color='orange')  

    plt.title('Average Satisfaction over Turns, Counterfactual')
    plt.xlabel('Turn')
    plt.ylabel('Average Satisfaction')
    plt.legend()
    plt.ylim(0, 0.40)
    plt.grid(True)
    plt.savefig("counter.png")

    test = 0