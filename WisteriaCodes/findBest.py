import sys

args = sys.argv
boText = args[1]

fileHandle = open(boText, "r")
lineList = fileHandle.readlines()
fileHandle.close()

bestFeatures = None
bestValue = float("-inf")

for row in lineList:

    # delete \n
    if row[-1] == "\n":
        row = row[:-2]

    values = row.split(",")
    features = ",".join(values[:-1])
    value = float(values[-1])

    if value > bestValue:
        bestValue = value
        bestFeatures = features

print(bestFeatures)