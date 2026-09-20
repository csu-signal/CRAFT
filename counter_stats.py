from enum import Enum
import json
from pathlib import Path
import os
from datasets import load_dataset
from matplotlib import pyplot as plt
import pandas as pd
import ast

def calculate_iou_board(current, target):
        """
        Calculate Intersection over Union (IoU) for block positions
        """
        intersection = 0
        union = 0
        
        for coord in current.keys():
            current_blocks = set(current[coord])
            target_blocks = set(target[coord])
            
            intersection += len(current_blocks.intersection(target_blocks))
            #print(f"I: {intersection} {current_blocks} {target_blocks}")
            union += len(current_blocks.union(target_blocks))
            #print(f"{union}")
        
        return intersection / union if union > 0 else 0.0

def normalize_structure(structure):
        """
        Normalize structure format for comparison
        Converts coordinate keys to tuples and handles missing positions
        """
        normalized = {}
        
        for i in range(3):
            for j in range(3):
                # Try both formats: with and without spaces
                coord_key_spaces = f"({i}, {j})"
                coord_key_no_spaces = f"({i},{j})"
                coord_tuple = (i, j)
                
                if coord_key_no_spaces in structure:
                    normalized[coord_tuple] = structure[coord_key_no_spaces]
                elif coord_key_spaces in structure:
                    normalized[coord_tuple] = structure[coord_key_spaces]
                else:
                    normalized[coord_tuple] = []
        
        return normalized

def calculate_iou(list1: list[str], list2: list[str]) -> float:
    """
    Calculates the IoU for two lists of strings.

    Args:
        list1: The first list of strings.
        list2: The second list of strings.

    Returns:
        The IoU as a float between 0 and 1.
    """
    set1 = set(list1)
    set2 = set(list2)

    intersection = len(set1.intersection(set2)) # or len(set1 & set2)
    union = len(set1.union(set2))               # or len(set1 | set2)

    if union == 0:
        return 0.0  # Avoid division by zero if both sets are empty

    return float(intersection) / float(union)

if __name__ == "__main__":
    validationPath = Path("/home/hannah/CRAFT/CRAFT/gittenExperiments/counterfactuals/divergenceData_policy_seed42_kl0.1_closs0.01_nosamp_hpt0.4")
    validationData = []
    factualValidationData = []
    ds = load_dataset("Abhijnan/craft-benchmark-lean")
    with open(f"/home/hannah/CRAFT/CRAFT/data/structures_dataset_20.json", 'r') as file:
            structData = json.load(file)

    for file_path in validationPath.glob("*.json"):
        with open(file_path, "r", encoding="utf-8") as file:
            data = json.load(file)   
            for d in data: 
                d["modelCombo"] =  file_path.name.split("_")[0] 
                factualPath = f"/home/hannah/CRAFT/CRAFT/divergenceData_policy_seed42_kl0.1_closs0.01_nosamp_hpt0.4/validation/{file_path.name}"
                with open(factualPath, "r", encoding="utf-8") as factFile:
                    factualData = json.load(factFile)
                    factData = [item for item in factualData if item["turn"] == d["turn"]]
                    factualValidationData.append(factData) 
            validationData.extend(data)

    validationSatP = 0
    validationSatPFact = 0
    validationSatF = 0
    validationSatFFact = 0

    validationTurnWiseP = {}
    validationTurnWiseF = {}
    validationTurnWisePFact = {}
    validationTurnWiseFFact = {}

    validationModelWise = {}
    for factTurn in factualValidationData:
        fact = factTurn[0][factTurn[0]["builderSelected"] + "_message"]
        turnKey = fact["timestamp"]

        satP = fact["satisfaction"]

        for s in structData:
            if(s['id'] == fact["structure"]):
                targetStruct = s["structure"]
                break

        target_norm = normalize_structure(targetStruct)
        full_structure_norm = normalize_structure(ast.literal_eval(fact["structureAfter"]))
        iou_score_full = calculate_iou_board(full_structure_norm, target_norm)
        satF = iou_score_full

        validationSatFFact += satF
        validationSatPFact += satP

        if turnKey in validationTurnWisePFact and isinstance(validationTurnWisePFact[turnKey], list):
            validationTurnWisePFact[turnKey].append(satP)
            validationTurnWiseFFact[turnKey].append(satF)
        else:
            validationTurnWisePFact[turnKey] = [satP]
            validationTurnWiseFFact[turnKey] = [satF]

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

        print(f"Count Validation (Factuals): {len(factualValidationData)}")
        if(len(factualValidationData) != 0):
            print(f"Average Validation Partial Satisfaction: {round(validationSatPFact / len(factualValidationData), 3)}\n")
            print(f"Average Validation Full Satisfaction: {round(validationSatFFact / len(factualValidationData), 3)}\n")
            validationTurnWisePFact = dict(sorted(validationTurnWisePFact.items()))
            validationTurnWiseFFact = dict(sorted(validationTurnWiseFFact.items()))
            valTurnAveragesPFact = {
                key: (round(sum(values) / len(values), 3)) if values else 0.0 
                for key, values in validationTurnWisePFact.items()
            }
    
            valTurnAveragesFFact = {
                        key: (round(sum(values) / len(values), 3)) if values else 0.0 
                        for key, values in validationTurnWiseFFact.items()
                    }
    
            print(valTurnAveragesPFact)
            print(valTurnAveragesFFact)
            print("\n")

        # valModelWiseAverages = {
        #     key: (round(sum(values) / len(values), 3)) if values else 0.0 
        #     for key, values in validationModelWise.items()
        # }

        # print(valModelWiseAverages)

    #########################################################

    if(len(validationData) != 0):
        plt.plot(valTurnAveragesP.keys(), valTurnAveragesP.values(), label='Partial IOU Counterfactuals', marker='o', color='blue') 
        plt.plot(valTurnAveragesF.keys(), valTurnAveragesF.values(), label='Full IOU Counterfactuals', marker='o', color='orange') 

        plt.plot(valTurnAveragesPFact.keys(), valTurnAveragesPFact.values(), label='Partial IOU Factuals', marker='o', color='purple') 
        plt.plot(valTurnAveragesFFact.keys(), valTurnAveragesFFact.values(), label='Full IOU Factuals', marker='o', color='red') 

    plt.title('Average Satisfaction over Turns, Counterfactual, Split 42')
    plt.xlabel('Turn')
    plt.ylabel('Average Satisfaction')
    plt.legend()
    plt.ylim(0, 0.6)
    plt.grid(True)
    plt.savefig("counter.png")

    test = 0