import csv

with open("data.csv", newline="") as f:
    reader = csv.reader(f)
    
    for row in reader:
        print(row)      # each row is a list
