import numpy as np
import matplotlib.pyplot as plt
import copy
import random

ltrJ = [[ 1, 1, 1, 1, 1],
		[-1,-1,-1, 1,-1],
		[-1,-1,-1, 1,-1],
		[ 1,-1,-1, 1,-1],
		[ 1, 1, 1,-1,-1]]

ltrH = [[ 1,-1,-1,-1, 1],
		[ 1,-1,-1,-1, 1],
		[ 1, 1, 1, 1, 1],
		[ 1,-1,-1,-1, 1],
		[ 1,-1,-1,-1, 1]]

ltrO = [[ 1, 1, 1, 1, 1],
		[ 1,-1,-1,-1, 1],
		[ 1,-1,-1,-1, 1],
		[ 1,-1,-1,-1, 1],
		[ 1, 1, 1, 1, 1]]
	
ltrM = [[ 1,-1,-1,-1, 1],
		[ 1, 1,-1, 1, 1],
		[ 1,-1, 1,-1, 1],
		[ 1,-1,-1,-1, 1],
		[ 1,-1,-1,-1, 1]]

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

def process_input(W, patterns, input):
	initial_p = np.matrix(input)
	stable = False
	spurious_state = True
	found_pttrn = -1
	A = np.transpose(initial_p)
	i = 0
	while not stable and i < 200:
		# print("aAaaaa")
		B = np.sign(W*A).astype(int)
		if np.array_equal(A,B):
			stable = True
			for i in range(0, len(patterns)):
				p = np.transpose(np.matrix(patterns[i]))
				if np.array_equal(B,p):
					# print(f"reached stable config at pattern {i}")
					spurious_state = False
					found_pttrn = i
		A = B
		i+=1

	# if spurious_state:
		# print(f"reached stable config at spurious state")
	
	return found_pttrn

def hopfield(patterns):
	
	np.set_printoptions(linewidth=np.inf)

	W = create_weights_matrix(patterns)
	W_alt = create_weights_matrix_alt(patterns)
	#W = W_alt
	
    #[ (para 0)[1pattern 2dopattern 3ro 4to spurius] (para 0.5)[...] ...]
    
	results = []
	
	iterations = 10000
	pattern = 0
	error_step = 0.05
	for pctg in np.arange(0, 1, 0.05):
		hits = 0
		false_hits = 0
		print(f'pctg: {pctg:.2f}')
		for i in range(0, iterations):
			noised_pattern = noisify(patterns[pattern], pctg)
			res = process_input(W, patterns, noised_pattern)
			if res == pattern:
				hits+=1
			elif res >= 0:
				false_hits+=1
		results.append({
			'hits': hits/iterations,
			'false_hits': false_hits/iterations,
			'spureous_hits': 1 - (false_hits + hits)/iterations
		})

	fig, ax = plt.subplots()
	ax.set_title(f'Pattern {pattern} w/ {iterations} iterations and error step {error_step}')
	l1, = ax.plot([pctg for pctg in np.arange(0, 1, error_step)], list(map(lambda a: a['hits'], results)), 'g-')
	l2, = ax.plot([pctg for pctg in np.arange(0, 1, error_step)], list(map(lambda a: a['false_hits'], results)), 'b-')
	l3, = ax.plot([pctg for pctg in np.arange(0, 1, error_step)], list(map(lambda a: a['spureous_hits'], results)), 'r-')
	ax.set_xlabel('Noise pctg')	
	ax.set_ylabel('Hit pctg')
	ax.legend([l1, l2, l3], ['Hits', 'False hits', 'Spureous hits'])
	plt.show()

	return results

def convert_pattern_to_list(patterns):
	patterns_as_lists = []
	for n in range(0, len(patterns)):
		pattern = patterns[n]
		patterns_as_lists.append([])
		for i in range(0, len(pattern)):
			for j in range(0, len(pattern[0])):
				patterns_as_lists[n].append(pattern[i][j])
	return patterns_as_lists

if __name__ == '__main__':
    
    # Patterns
	patterns_as_matrixes = [ltrJ, ltrH, ltrO, ltrM]
	patterns_as_lists = convert_pattern_to_list(patterns_as_matrixes)

    # Hopfield
	hopfield(patterns_as_lists)

	exit(0)







































