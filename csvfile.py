import csv
import random


header = ["ID", "Name", "Age", "Score"]


names = ["Harshita", "Khushi", "Debalina", "Rohan", "Suman", "Ananya"]


with open("data.csv", "w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(header)

  
    for i in range(1, 201):
        row = [
            i,
            random.choice(names),
            random.randint(18, 30),
            random.randint(0, 100)
        ]
        writer.writerow(row)

print("CSV file created successfully!")