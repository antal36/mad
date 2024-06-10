def getCSVFromArff(fileName: str) -> None:
	"""Converts ARFF file into CSV file"""
	with open(fileName + '.arff', 'r') as file:
		data = file.read().splitlines(True)
	i: int = 0
	cols: list = []
	for line in data:
		line = line.lower()
		if '@data' in line:
			i += 1
			break
		else:
			i += 1
			if (line.startswith('@attribute')):
				if '{' in line:
					cols.append(line[11:line.index('{')-1])
				else:
					cols.append(line[11:line.index(' ', 11)])
	headers = ",".join(cols)
	with open(fileName + '.csv', 'w') as fout:
		fout.write(headers)
		fout.write('\n')
		fout.writelines(data[i:])

getCSVFromArff("./dataset_37_diabetes")

"""This file has also been removed, non-functioning call of the function because of the path."""