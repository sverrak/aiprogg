def init_rows(file_name):
	with open(file_name) as f:
		lines_raw = f.read().splitlines()
		lines = []
		for l in range(len(lines_raw)):
			lines.append(lines_raw[l].split())
		
		for i in range(len(lines)):
			for j in range(len(lines[i])):
				lines[i][j] = int(lines[i][j])
		
		nRows = lines[0][0]
		nColumns = lines[0][1]

		row_indices = []
		col_indices = []

		for i in range(1, nRows + 1):
			row_indices.append(lines[i])

		for i in range(nColumns):
			col_indices.append(lines[nRows + 1 + i])
		
		return row_indices, col_indices, nRows, nColumns

print(init_rows("clover.txt"))

def print_candidates(rows):
	for r in rows:
		print(r)

# Generates the candidate rows for a certain row description
def generate_candidate_lists(row_indices, nColumns):
	candidates = []
	
	### INSERT CODE

	if (len(row_indices)==1):
		for i in range(nColumns - row_indices[0] + 1):
			candidates.append([0 for j in range(0, i)] + [1] + [0 for j in range(i, nColumns-row_indices[0])])
			
	if (len(row_indices)==2):
		for i in range(nColumns - len(row_indices)):
		a = 0
	return candidates

print(generate_candidate_lists([1],4))
