import csv
def load_csv(filename):
    lines = csv.reader(open(filename, "rb"))
    dataset = list(lines)
    return dataset;


