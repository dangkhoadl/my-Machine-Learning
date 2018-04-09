csvfilename = "input.csv"
'''
City,Jan,Feb,Mar,Apr,May,Jun,Jul,Aug,Sep,Oct,Nov,Dec
Houston,62,65,72,78,84,90,92,93,88,81,71,63
Baghdad,61,66,75,86,97,108,111,111,104,91,75,64
Moscow,21,25,36,50,64,72,73,70,59,46,34,25
San Francisco,57,60,62,63,64,66,67,68,70,69,63,57
London,43,45,50,55,63,68,72,70,66,57,50,45
Chicago,32,36,46,59,70,81,84,82,75,63,48,36
Sydney,79,79,77,73,68,64,63,64,68,72,75,79
Paris,45,46,54,61,68,73,77,77,70,61,52,46
Tokyo,46,48,54,63,70,75,82,84,79,68,59,52
Shanghai,46,48,55,66,75,81,90,90,81,72,63,52
'''
def printTable(table):
    for row in table:
        # Header column left justified
        print("{:<19}".format(row[0]), end='')
        # Remaining columns right justified
        for col in row[1:]:
            print("{:>4}".format(col), end='')
        print("", end='\n')


################################## Parse csv #####################################
table = []
with open(csvfilename, 'r') as csvfile:
    for line in csvfile:
        line = line.rstrip()
        columns = line.split(',')
        table.append(columns)


############################# CSV module ##########################################
import csv
table = []
with open(csvfilename, 'r') as csvfile:
    csvreader = csv.reader(csvfile, skipinitialspace=True)
    for row in csvreader:
        table.append(row)


############################# CSV dict parse ##########################################
import csv
table = {}
with open(csvfilename, "rt", newline='') as csvfile:
    csvreader = csv.DictReader(csvfile,skipinitialspace=True)
    for row in csvreader:
        table[row['City']] = row