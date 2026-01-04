from datasets import load_dataset
import csv
import os



dataset = load_dataset("OsamaBsher/AITA-Reddit-Dataset")
data = {}
content = dataset['train'][:500]['title']
ah_score = dataset['train'][:500]['verdict']

if os.path.exists("aita.csv"):
    os.remove("aita.csv")
    print("[Log] Existing file deleted successfully")


with open('aita.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile, delimiter=',')
    for i in range(len(ah_score)):
        data[content[i]] = ah_score[i]
        writer.writerow([content[i], ah_score[i]])


print(data)

