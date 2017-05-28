#1536242 - Paolo Tamagnini

import csv
import time
import pprint as pp
import networkx as nx

import Network_Based_Recommendation_System_FUNCTIONS as homework_3



print
print "Current time: " + str(time.asctime(time.localtime()))
print
print


all_groups = [
	{1701: 1, 1703: 1, 1705: 1, 1707: 1, 1709: 1}, ### Movie night with friends.
	{1701: 1, 1702: 4}, ### First appointment scenario: the preferences of the girl are 4 times more important than those of the man.
	{1701: 1, 1702: 2, 1703: 1, 1704: 2}, ### Two couples scenario: the preferences of the girls are still more important than those of the man.
	{1701: 1, 1702: 1, 1703: 1, 1704: 1, 1705: 1, 1720:10}, ### Movie night with a special guest.
	{1701: 1, 1702: 1, 1703: 1, 1704: 1, 1705: 1, 1720:10, 1721:10, 1722:10}, ### Movie night with 3 special guests.
]
print
pp.pprint(all_groups)
print


graph_file = "./input_data/u_data_homework_format.txt"

pp.pprint("Load Graph.")
print "Current time: " + str(time.asctime(time.localtime()))
graph_users_items = homework_3.create_graph_set_of_users_set_of_items(graph_file)
print " #Users in Graph= " + str(len(graph_users_items['users']))
print " #Items in Graph= " + str(len(graph_users_items['items']))
print " #Nodes in Graph= " + str(len(graph_users_items['graph']))
print " #Edges in Graph= " + str(graph_users_items['graph'].number_of_edges())
print "Current time: " + str(time.asctime(time.localtime()))
print
print


pp.pprint("Create Item-Item-Weighted Graph.")
print "Current time: " + str(time.asctime(time.localtime()))
item_item_graph = homework_3.create_item_item_graph(graph_users_items)
print " #Nodes in Item-Item Graph= " + str(len(item_item_graph))
print " #Edges in Item-Item Graph= " + str(item_item_graph.number_of_edges())
print "Current time: " + str(time.asctime(time.localtime()))
print
print


### Conversion of the 'Item-Item-Graph' to a scipy sparse matrix representation.
### This reduces a lot the PageRank running time ;)
print
print " Conversion of the 'Item-Item-Graph' to a scipy sparse matrix representation."
N = len(item_item_graph)
nodelist = item_item_graph.nodes()
M = nx.to_scipy_sparse_matrix(item_item_graph, nodelist=nodelist, weight='weight', dtype=float)
print " Done."
print
#################################################################################################


output_file = open("./Output_Recommendation_for_Group.tsv", 'w')
output_file_csv_writer = csv.writer(output_file, delimiter='\t', quotechar='"', quoting=csv.QUOTE_NONE)
print
for current_group in all_groups:
	print "Current group: "
	pp.pprint(current_group)
	print "Current time: " + str(time.asctime(time.localtime()))
	
	sorted_list_of_recommended_items_for_current_group = []
	#####BEGIN_MY_CODE#####
	import numpy as np
	#setting total number of items
	nI = len(graph_users_items['items'])
	#building keys for dictionary of overall preference vector
	overallPrefVectorkeys = list(graph_users_items['items'])
	#initializing values for dictionary of overall preference vector
	overallPrefVectorvalues = np.zeros(nI)
	#for each member of the current group..
	for ui in current_group:
		#compute his own pref. vect.
		preference_vector = homework_3.create_preference_vector_for_teleporting(ui, graph_users_items)
		#then convert its values in an array
		pVValues = np.array(preference_vector.values())
		#and multiply each component of this vector of ratings for weight of the member within the group
		pVValues = pVValues*float(current_group[ui])
		
		#then add such values to the values vector of the overall pref vector
		
		#I made sure we are not mixing ratings of different movies because the values of a dictionary 
		#are given with the same ascending order of keys which are the same in prefVector for the single 
		#member of the group and in the overall pref vect.
		
		overallPrefVectorvalues = overallPrefVectorvalues + pVValues
	#once we have visited each member of the group we can put together our overall preference vector for the current group
	overallPrefVector = dict(zip(list(overallPrefVectorkeys),list(overallPrefVectorvalues)))
	
	#with such vector is possible to compute the personalized page rank, personalized on the overall result of the group
	page_rank_vector_of_items = homework_3.pagerank(M, N, nodelist, alpha=0.85, personalization=overallPrefVector)
	
	#it is time now to compute a result from the page rank
	sorted_list_of_recommended_items = []
	
	#let's erase any movies watched by any member of the group
	#(we don't want to force anyone to watch a movie he/she has seen already, otherwise he will spoil it to everyone else)
	
	#we make a list of movies seen/rated by the members of the current group
	itemsToErase = []
	for u in current_group:
			itemsToErase =  itemsToErase + graph_users_items['graph'][u].keys()
	itemsToErase = list(set(itemsToErase))
	
	#then we delete each of those entry from the page rank result
	for i in itemsToErase:
		del page_rank_vector_of_items[i]
	
	#once we have done that we just need to sort it
	
	#get zip from dictionary
	zipofDict = zip(page_rank_vector_of_items.keys(),page_rank_vector_of_items.values())
	#sort the zip
	zipofDict.sort(key = lambda g: g[1], reverse = True)
	#just get the list of sorted movies from the zip
	sorted_list_of_recommended_items_for_current_group = list(zip(*zipofDict)[0])
	#####END_MY_CODE#####
	
	
	
	print "Recommended Sorted List of Items:"
	print(str(sorted_list_of_recommended_items_for_current_group[:30]))
	print
	output_file_csv_writer.writerow(sorted_list_of_recommended_items_for_current_group)
	
output_file.close()	
	
	




print
print
print "Current time: " + str(time.asctime(time.localtime()))
print "Done ;)"
print
