#1536242 - Paolo Tamagnini

import csv
import pprint as pp
import networkx as nx
import itertools as it
import math
import scipy.sparse
import random




def pagerank(M, N, nodelist, alpha=0.85, personalization=None, max_iter=100, tol=1.0e-6, dangling=None):
	if N == 0:
		return {}
	S = scipy.array(M.sum(axis=1)).flatten()
	S[S != 0] = 1.0 / S[S != 0]
	Q = scipy.sparse.spdiags(S.T, 0, *M.shape, format='csr')
	M = Q * M
	
	# initial vector
	x = scipy.repeat(1.0 / N, N)
	
	# Personalization vector
	if personalization is None:
		p = scipy.repeat(1.0 / N, N)
	else:
		missing = set(nodelist) - set(personalization)
		if missing:
			raise NetworkXError('Personalization vector dictionary must have a value for every node. Missing nodes %s' % missing)
		p = scipy.array([personalization[n] for n in nodelist], dtype=float)
		p = p / p.sum()
	
	# Dangling nodes
	if dangling is None:
		dangling_weights = p
	else:
		missing = set(nodelist) - set(dangling)
		if missing:
			raise NetworkXError('Dangling node dictionary must have a value for every node. Missing nodes %s' % missing)
		# Convert the dangling dictionary into an array in nodelist order
		dangling_weights = scipy.array([dangling[n] for n in nodelist], dtype=float)
		dangling_weights /= dangling_weights.sum()
	is_dangling = scipy.where(S == 0)[0]

	# power iteration: make up to max_iter iterations
	for _ in range(max_iter):
		xlast = x
		x = alpha * (x * M + sum(x[is_dangling]) * dangling_weights) + (1 - alpha) * p
		# check convergence, l1 norm
		err = scipy.absolute(x - xlast).sum()
		if err < N * tol:
			return dict(zip(nodelist, map(float, x)))
	raise NetworkXError('power iteration failed to converge in %d iterations.' % max_iter)




def create_graph_set_of_users_set_of_items(user_item_ranking_file):
	graph_users_items = {}
	all_users_id = set()
	all_items_id = set()
	g = nx.DiGraph()
	input_file = open(user_item_ranking_file, 'r')
	input_file_csv_reader = csv.reader(input_file, delimiter='\t', quotechar='"', quoting=csv.QUOTE_NONE)
	for line in input_file_csv_reader:
		user_id = int(line[0])
		item_id = int(line[1])
		rating = int(line[2])
		g.add_edge(user_id, item_id, weight=rating)
		all_users_id.add(user_id)
		all_items_id.add(item_id)
	input_file.close()
	graph_users_items['graph'] = g
	graph_users_items['users'] = all_users_id
	graph_users_items['items'] = all_items_id
	return graph_users_items
	












def create_item_item_graph(graph_users_items):
	g = nx.Graph()
	#####BEGIN_MY_CODE#####
	import itertools
	g_old = graph_users_items['graph']

	d = {}
	all_items_id = graph_users_items['items']
	allEdgesOld = g_old.edges()
	for x in all_items_id:
		d[x] = []
	for y in allEdgesOld:
		d[y[1]].append(y[0])

	listOfPairs = list(itertools.combinations(all_items_id, 2))
	for x in listOfPairs:
		edgesOfInterest0 = d[x[0]]
		edgesOfInterest1 = d[x[1]]
		count = len(set(edgesOfInterest0) & set(edgesOfInterest1))
		if count != 0:
			g.add_edge(x[0], x[1], weight=count)
	#####END_MY_CODE#####		
	return g



def create_preference_vector_for_teleporting(user_id, graph_users_items):
	preference_vector = {}
	#####BEGIN_MY_CODE#####
	for x in graph_users_items['items']:
		preference_vector[x] = 0.0
	dUser = graph_users_items['graph'][user_id]
	for x in dUser:
		preference_vector[x] = dUser[x]['weight']
	#####END_MY_CODE#####
	return preference_vector
	



def create_ranked_list_of_recommended_items(page_rank_vector_of_items, user_id, training_graph_users_items):
	# This is a list of 'item_id' sorted in descending order of score.
	sorted_list_of_recommended_items = []
	# You can produce this list from a list of [item, score] couples sorted in descending order of score.
	#####BEGIN_MY_CODE#####
	userToErase = training_graph_users_items['graph'][user_id].keys()
	for x in userToErase:
		del page_rank_vector_of_items[x]
	zipofDict = zip(page_rank_vector_of_items.keys(),page_rank_vector_of_items.values())
	zipofDict.sort(key = lambda x: x[1], reverse = True)
	sorted_list_of_recommended_items = list(zip(*zipofDict)[0])
	#####END_MY_CODE#####
	return sorted_list_of_recommended_items




def discounted_cumulative_gain(user_id, sorted_list_of_recommended_items, test_graph_users_items):
	dcg = 0.
	#####BEGIN_MY_CODE#####
	import numpy as np
	moviesList = []
	ratingsList = []
	dictz = test_graph_users_items['graph'][user_id]
	for x in dictz:
		moviesList.append(x)
		ratingsList.append(dictz[x]['weight'])
	tupleUserRatings = zip(moviesList,ratingsList)

	sortedtupleUserRatings = []
	for x in sorted_list_of_recommended_items:
		for y in tupleUserRatings:
			if y[0] == x:
				sortedtupleUserRatings.append(y)
				break

	dcg = sortedtupleUserRatings[0][1]
	for i in range(1,len(sortedtupleUserRatings)):
		l = np.log2(i+1)
		dcg = dcg + sortedtupleUserRatings[i][1]/l
	#####END_MY_CODE#####
	return dcg
	



def minimum_discounted_cumulative_gain(user_id, test_graph_users_items):
	dcg = 0.
	#####BEGIN_MY_CODE#####
	import numpy as np
	moviesList = []
	ratingsList = []
	dictz = test_graph_users_items['graph'][user_id]
	for x in dictz:
		moviesList.append(x)
		ratingsList.append(dictz[x]['weight'])
	worstSorting = zip(moviesList,ratingsList)
	worstSorting.sort(key = lambda x: x[1], reverse = False)

	dcg = worstSorting[0][1]
	for i in range(1,len(worstSorting)):
		l = np.log2(i+1)
		dcg = dcg + worstSorting[i][1]/l
	#####END_MY_CODE#####
	return dcg





def maximum_discounted_cumulative_gain(user_id, test_graph_users_items):
	dcg = 0.
	#####BEGIN_MY_CODE#####
	import numpy as np
	moviesList = []
	ratingsList = []
	dictz = test_graph_users_items['graph'][user_id]
	for x in dictz:
		moviesList.append(x)
		ratingsList.append(dictz[x]['weight'])
	bestSorting = zip(moviesList,ratingsList)
	bestSorting.sort(key = lambda x: x[1], reverse = True)

	dcg = bestSorting[0][1]
	for i in range(1,len(bestSorting)):
		l = np.log2(i+1)
		dcg = dcg + bestSorting[i][1]/l
	#####END_MY_CODE#####
	return dcg













