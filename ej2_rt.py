import numpy as np
import matplotlib.pyplot as plt
import copy
import random
import multiprocessing as mp
import keyboard

from plotters import plot_hopfield

all_letters = {
	
	'A': [	[-1,-1,+1,-1,-1],
			[-1,+1,-1,+1,-1],
			[+1,+1,+1,+1,+1],
			[+1,-1,-1,-1,+1],
			[+1,-1,-1,-1,+1]],

	'B': [	[+1,+1,+1,+1,-1],
			[+1,-1,-1,-1,+1],
			[+1,+1,+1,+1,-1],
			[+1,-1,-1,-1,+1],
			[+1,+1,+1,+1,+1]],

	'C': [	[+1,+1,+1,+1,+1],
			[+1,-1,-1,-1,-1],
			[+1,-1,-1,-1,-1],
			[+1,-1,-1,-1,-1],
			[+1,+1,+1,+1,+1]],

	'D':[	[+1,+1,+1,+1,-1],
			[+1,-1,-1,-1,+1],
			[+1,-1,-1,-1,+1],
			[+1,-1,-1,-1,+1],
			[+1,+1,+1,+1,-1]],

	'E': [	[+1,+1,+1,+1,+1],
			[+1,-1,-1,-1,-1],
			[+1,+1,+1,+1,+1],
			[+1,-1,-1,-1,-1],
			[+1,+1,+1,+1,+1]],

	'F': [	[+1,+1,+1,+1,+1],
			[+1,-1,-1,-1,-1],
			[+1,+1,+1,+1,+1],
			[+1,-1,-1,-1,-1],
			[+1,-1,-1,-1,-1]],

	'G': [	[+1,+1,+1,+1,+1],
			[+1,-1,-1,-1,-1],
			[+1,-1,+1,+1,+1],
			[+1,-1,-1,-1,+1],
			[+1,+1,+1,+1,+1]],

	'H': [	[+1,-1,-1,-1,+1],
			[+1,-1,-1,-1,+1],
			[+1,+1,+1,+1,+1],
			[+1,-1,-1,-1,+1],
			[+1,-1,-1,-1,+1]],

	'I': [	[+1,+1,+1,+1,+1], 
			[-1,-1,+1,-1,-1],
			[-1,-1,+1,-1,-1],
			[-1,-1,+1,-1,-1],
			[+1,+1,+1,+1,+1]],

	'J': [	[+1,+1,+1,+1,+1],
			[-1,-1,-1,+1,-1],
			[-1,-1,-1,+1,-1],
			[+1,-1,-1,+1,-1],
			[+1,+1,+1,-1,-1]],

	'K': [	[+1,-1,-1,+1,-1],
			[+1,-1,+1,-1,-1],
			[+1,+1,-1,-1,-1],
			[+1,-1,+1,-1,-1],
			[+1,-1,-1,+1,-1]],

	'L': [	[+1,-1,-1,-1,-1],
			[+1,-1,-1,-1,-1],
			[+1,-1,-1,-1,-1],
			[+1,-1,-1,-1,-1],
			[+1,+1,+1,+1,+1]],

	'M': [	[+1,-1,-1,-1,+1],
			[+1,+1,-1,+1,+1],
			[+1,-1,+1,-1,+1],
			[+1,-1,-1,-1,+1],
			[+1,-1,-1,-1,+1]],

	'N': [	[+1,-1,-1,-1,+1],
			[+1,+1,-1,-1,+1],
			[+1,-1,+1,-1,+1],
			[+1,-1,-1,+1,+1],
			[+1,-1,-1,-1,+1]],

	'O': [	[+1,+1,+1,+1,+1],
			[+1,-1,-1,-1,+1],
			[+1,-1,-1,-1,+1],
			[+1,-1,-1,-1,+1],
			[+1,+1,+1,+1,+1]],

	'P': [	[+1,+1,+1,+1,+1],
			[+1,-1,-1,-1,+1],
			[+1,+1,+1,+1,+1],
			[+1,-1,-1,-1,-1],
			[+1,-1,-1,-1,-1]],

	'Q': [	[+1,+1,+1,+1,+1],
			[+1,-1,-1,-1,+1],
			[+1,-1,+1,-1,+1],
			[+1,-1,-1,+1,+1],
			[+1,+1,+1,+1,+1]],

	'R': [	[+1,+1,+1,+1,-1],
			[+1,-1,-1,-1,+1],
			[+1,+1,+1,+1,-1],
			[+1,-1,+1,-1,-1],
			[+1,-1,-1,+1,-1]],

	'S': [	[+1,+1,+1,+1,+1],
			[+1,-1,-1,-1,-1],
			[-1,+1,+1,+1,-1],
			[-1,-1,-1,-1,+1],
			[+1,+1,+1,+1,+1]],

	'T': [	[+1,+1,+1,+1,+1],
			[-1,-1,+1,-1,-1],
			[-1,-1,+1,-1,-1],
			[-1,-1,+1,-1,-1],
			[-1,-1,+1,-1,-1]],

	'U': [	[+1,-1,-1,-1,+1],
			[+1,-1,-1,-1,+1],
			[+1,-1,-1,-1,+1],
			[+1,-1,-1,-1,+1],
			[-1,+1,+1,+1,-1]],

	'V': [	[+1,-1,-1,-1,+1],
			[+1,-1,-1,-1,+1],
			[-1,+1,-1,+1,-1],
			[-1,+1,-1,+1,-1],
			[-1,-1,+1,-1,-1]],

	'W': [	[+1,-1,-1,-1,+1],
			[+1,-1,-1,-1,+1],
			[+1,-1,+1,-1,+1],
			[+1,+1,-1,+1,+1],
			[+1,-1,-1,-1,+1]],

	'X': [	[+1,-1,-1,-1,+1],
			[-1,+1,-1,+1,-1],
			[-1,-1,+1,-1,-1],
			[-1,+1,-1,+1,-1],
			[+1,-1,-1,-1,+1]],

	'Y': [	[+1,-1,-1,-1,+1],
			[-1,+1,-1,+1,-1],
			[-1,-1,+1,-1,-1],
			[-1,-1,+1,-1,-1],
			[-1,-1,+1,-1,-1]],
			
	'Z':[	[+1,+1,+1,+1,+1],
			[-1,-1,-1,+1,-1],
			[-1,-1,+1,-1,-1],
			[-1,+1,-1,-1,-1],
			[+1,+1,+1,+1,+1]]
}

def create_weights_matrix_alt(patterns):
	K = np.transpose(np.array(patterns))
	N = K.shape[0]
	W_prod = K.dot(np.transpose(K))
	
	for i in range(0, W_prod.shape[0]):
		W_prod[i,i] = 0

	W = (1.0/N) * W_prod

	return W

def create_weights_matrix(patterns):
	pattern_len = len(patterns[0])
	N = len(patterns)
	W = []
	for i in range(0, pattern_len):
		W.append([])
		for j in range(0, pattern_len):
			if(i == j):
				W[i].append(0)
			else:
				pattern_sum = 0
				for n in range(0, N):
					pattern_sum += (patterns[n][i] * patterns[n][j])
				W[i].append((1/N)*(pattern_sum))
	return np.matrix(W)

def convert_pattern_to_list(patterns):
	patterns_as_lists = []
	for n in range(0, len(patterns)):
		pattern = patterns[n]
		patterns_as_lists.append([])
		for i in range(0, len(pattern)):
			for j in range(0, len(pattern[0])):
				patterns_as_lists[n].append(pattern[i][j])
	return patterns_as_lists

def noisify(pattern, pctg):
	pattern = np.array(pattern)
	number_to_noise = pctg * pattern.shape[0]
	noisy_pattern = copy.copy(pattern)
	possible_idxs = np.random.randint(low=0, high=pattern.shape[0], size=pattern.shape[0], dtype=int)
	random.shuffle(possible_idxs)
	idxs = possible_idxs[0:int(number_to_noise)]

	for idx in idxs:
		noisy_pattern[idx] = -noisy_pattern[idx]

	return noisy_pattern

def get_ortogonals(letter, letter_pattern):

	all_letters_with_pattern = copy.copy(all_letters)

	for l in all_letters_with_pattern:
		all_letters_with_pattern[l] = convert_pattern_to_list([ all_letters_with_pattern[l] ])[0]

	all_let = set(all_letters.keys())
	letters_added = set([letter])
	letters_added_list = list([letter])

	letters_not_added = all_let.difference(letters_added)

	while letters_not_added:
		most_ort = None
		most_ort_val = None

		# print(f"Letters added: {letters_added_list}")

		for let in letters_not_added:
			pat = all_letters_with_pattern[let]
			if let not in letters_added:
				inner = sum(map(lambda l: abs(np.inner(all_letters_with_pattern[l], pat)), letters_added))
				if most_ort is None or inner < most_ort_val:
					most_ort = let
					most_ort_val = inner

		# print(f"Added letter '{most_ort}' (inner={most_ort_val:.2f})")

		letters_added.add(most_ort)
		letters_added_list.append(most_ort)
		letters_not_added.remove(most_ort)

	return letters_added_list

def process_input(W, patterns, input):
	initial_p = np.matrix(input)
	stable = False
	spurious_state = True
	found_pttrn = -1
	A = np.transpose(initial_p)
	it = 0
	max_iter = 200

	plotter_q = mp.Queue()
	plotter_q.cancel_join_thread()

	plotter = mp.Process(target=plot_hopfield, args=((plotter_q),))
	plotter.daemon = True
	plotter.start()

	output = [ [ 0 for a in range(0,5) ] for b in range(0,5) ]

	for i in range(0, len(output)):
		for j in range(0, len(output[i])):
			output[i][j] = A[i*5+j, 0]

	plotter_q.put({
		'output': output
	})

	while not stable and it < max_iter:
		# print("aAaaaa")
		B = np.sign(W*A).astype(int)

		for h in range(0, B.shape[0]):
			if B[h, 0] == 0:
				B[h, 0] = A[h, 0]

		B_arr = np.transpose(B)

		output = [ [ 0 for a in range(0,5) ] for b in range(0,5) ]

		for i in range(0, len(output)):
			for j in range(0, len(output[i])):
				output[i][j] = B_arr[0, i*5+j]

		plotter_q.put({
			'output': output
		})

		if np.array_equal(A,B):
			stable = True
			for k in range(0, len(patterns)):
				p = np.transpose(np.matrix(patterns[k]))
				if np.array_equal(B,p):
					# print(f"reached stable config at pattern {k}")
					spurious_state = False
					found_pttrn = k
		A = B
		it+=1
		print(it)

	plotter_q.put("STOP")
	print("Press 'q' to finish plot")
	keyboard.wait("q")
	
	return found_pttrn

def hopfield(patterns, letter, pctg):
	
	np.set_printoptions(linewidth=np.inf)

	W = create_weights_matrix(patterns)
	W_alt = create_weights_matrix_alt(patterns)
	#W = W_alt

	hits = 0
	false_hits = 0
	print(f'pctg: {pctg:.2f}')
	noised_pattern = noisify(patterns[0], pctg)
	res = process_input(W, patterns, noised_pattern)
	if res == 0:
		hits+=1
	elif res >= 0:
		false_hits+=1

	results = {
		'hits': hits,
		'false_hits': false_hits,
		'spureous_hits': 1 - (false_hits + hits)
    }

	return results

if __name__ == '__main__':
    
    # Patterns
	all_patterns_as_matrixes = list(all_letters.values())
	all_patterns_as_lists = convert_pattern_to_list(all_patterns_as_matrixes)

	letter = 'A'
	letter_pattern = all_letters[letter]
	letter_pattern = convert_pattern_to_list([letter_pattern])[0]

	most_ortogonals = get_ortogonals(letter, letter_pattern)

	most_ortogonals = most_ortogonals[0:4]

	print(most_ortogonals)

	patterns_as_matrixes = list(map(lambda l: all_letters[l], most_ortogonals))
	patterns_as_lists = convert_pattern_to_list(patterns_as_matrixes)

    # Hopfield
	pctg = 0.5
	hopfield(patterns_as_lists, letter, pctg)

	exit(0)
