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
        union += len(current_blocks.union(target_blocks))

    return intersection / union if union > 0 else 0.0


def normalize_structure(structure):
    """
    Normalize structure format for comparison.
    Converts coordinate keys to tuples and handles missing positions.
    """
    normalized = {}

    for i in range(3):
        for j in range(3):
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

    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))

    if union == 0:
        return 0.0  # Avoid division by zero if both sets are empty

    return float(intersection) / float(union)


if __name__ == "__main__":
    validationPath = Path("/home/hannah/CRAFT/CRAFT/gittenExperiments/counterfactuals/divergenceData_policy_seed42_kl0.1_closs0.01_nosamp_hpt0.4")
    validationData = []
    factualValidationData = []

    # FIX #7: Removed unused `ds = load_dataset(...)` call

    with open(f"/home/hannah/CRAFT/CRAFT/data/structures_dataset_20.json", 'r') as file:
        structData = json.load(file)

    for file_path in validationPath.glob("*.json"):
        with open(file_path, "r", encoding="utf-8") as file:
            data = json.load(file)
            for d in data:
                d["modelCombo"] = file_path.name.split("_")[0]
                factualPath = f"/home/hannah/CRAFT/CRAFT/divergenceData_policy_seed42_kl0.1_closs0.01_nosamp_hpt0.4/validation/{file_path.name}"
                with open(factualPath, "r", encoding="utf-8") as factFile:
                    factualData = json.load(factFile)
                    factData = [item for item in factualData if item["turn"] == d["turn"]]

                    # FIX #1: Guard against empty factData before appending
                    if factData:
                        factualValidationData.append(factData)

            validationData.extend(data)

    validationSatP = 0
    validationSatPFact = 0
    validationSatF = 0
    validationSatFFact = 0

    validationTurnWiseP = {}
    validationTurnWiseF = {}
    # FIX #3: Renamed turnKey dicts to use clearer names distinguishing timestamp vs turn number
    validationTimestampWisePFact = {}
    validationTimestampWiseFFact = {}

    validationModelWise = {}

    for factTurn in factualValidationData:
        # FIX #1: Already guarded above, but double-check here for safety
        if not factTurn:
            continue

        fact = factTurn[0][factTurn[0]["builderSelected"] + "_message"]

        # FIX #3: Renamed to timestampKey to distinguish from turn-number keys
        timestampKey = fact["timestamp"]

        satP = fact["satisfaction"]

        # FIX #4: Use next() with a fallback instead of a bare loop with no fallback
        targetStruct = next((s["structure"] for s in structData if s["id"] == fact["structure"]), None)
        if targetStruct is None:
            print(f"Warning: No matching structure found for id '{fact['structure']}', skipping.")
            continue

        target_norm = normalize_structure(targetStruct)
        full_structure_norm = normalize_structure(ast.literal_eval(fact["structureAfter"]))
        iou_score_full = calculate_iou_board(full_structure_norm, target_norm)
        satF = iou_score_full

        validationSatFFact += satF
        validationSatPFact += satP

        if timestampKey in validationTimestampWisePFact and isinstance(validationTimestampWisePFact[timestampKey], list):
            validationTimestampWisePFact[timestampKey].append(satP)
            validationTimestampWiseFFact[timestampKey].append(satF)
        else:
            validationTimestampWisePFact[timestampKey] = [satP]
            validationTimestampWiseFFact[timestampKey] = [satF]

    for turn in validationData:
        validationSatP += turn["satisfaction_partial"]
        validationSatF += turn["satisfaction_full"]

        # FIX #3: Renamed to turnKey (int) to distinguish from timestampKey (above)
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

    # FIX #6: Initialize plot variables to None so we can check before plotting
    valTurnAveragesP = {}
    valTurnAveragesF = {}
    valTurnAveragesPFact = {}
    valTurnAveragesFFact = {}

    print(f"Count Validation: {len(validationData)}")
    if len(validationData) != 0:
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

    # FIX #5: Factual stats block is now independent of validationData being non-empty
    print(f"Count Validation (Factuals): {len(factualValidationData)}")
    if len(factualValidationData) != 0:
        print(f"Average Validation Partial Satisfaction (Factuals): {round(validationSatPFact / len(factualValidationData), 3)}\n")
        print(f"Average Validation Full Satisfaction (Factuals): {round(validationSatFFact / len(factualValidationData), 3)}\n")

        validationTimestampWisePFact = dict(sorted(validationTimestampWisePFact.items()))
        validationTimestampWiseFFact = dict(sorted(validationTimestampWiseFFact.items()))

        valTurnAveragesPFact = {
            key: (round(sum(values) / len(values), 3)) if values else 0.0
            for key, values in validationTimestampWisePFact.items()
        }
        valTurnAveragesFFact = {
            key: (round(sum(values) / len(values), 3)) if values else 0.0
            for key, values in validationTimestampWiseFFact.items()
        }

        print(valTurnAveragesPFact)
        print(valTurnAveragesFFact)
        print("\n")

    #########################################################

    # FIX #6: Only plot series that have data
    if valTurnAveragesP:
        plt.plot(valTurnAveragesP.keys(), valTurnAveragesP.values(), label='Partial IOU Counterfactuals', marker='o', color='blue')
    if valTurnAveragesF:
        plt.plot(valTurnAveragesF.keys(), valTurnAveragesF.values(), label='Full IOU Counterfactuals', marker='o', color='orange')
    if valTurnAveragesPFact:
        plt.plot(valTurnAveragesPFact.keys(), valTurnAveragesPFact.values(), label='Partial IOU Factuals', marker='o', color='purple')
    if valTurnAveragesFFact:
        plt.plot(valTurnAveragesFFact.keys(), valTurnAveragesFFact.values(), label='Full IOU Factuals', marker='o', color='red')

    plt.title('Average Satisfaction over Turns, Counterfactual, Split 42')
    plt.xlabel('Turn')
    plt.ylabel('Average Satisfaction')
    plt.legend()
    plt.ylim(0, 0.6)
    plt.grid(True)
    plt.savefig("counter1.png")