import json

from datasets import load_dataset

modelCombo =  "gemini-3-flash-preview + 4o-mini"
structure = "structure_017"
director = modelCombo.split("+")[0].strip().replace("-", "").lower()
if(director == "deepseeklite"):
    director = "deepseekv2lite"
ds = load_dataset("Abhijnan/craft-benchmark-lean")
filtered_ds = ds.filter(lambda example: example["structure_id"] == structure and example["director_model"].replace("-", "").lower() == director)

if(len(filtered_ds['train']) != 0):
    with open("filteredData.txt", "w", encoding="utf-8") as file:
        for data in filtered_ds['train']:
            file.write("\nTurn: " + str(data["turn_number"]) + "\n")
            # conversation_snapshot = json.loads(data["conversation_snapshot"]) 
            # for utterance in conversation_snapshot[-5:]:
            #     file.write(f"{utterance} \n")
            file.write("D1 Message: " + str(data["D1_message"]) + "\n")
            file.write("D2 Message: " + str(data["D2_message"]) + "\n")
            file.write("D3 Message: " + str(data["D3_message"]) + "\n")
            print(data)