# -*- coding: utf-8 -*-
# ad click prediction : a view from the trenches
# __credits__ : tinrtgu

from math import sqrt, exp, log
from csv import DictReader, reader
#import pandas as pd
#import numpy as np
import random
import cPickle as pickle

class ftrl(object):
	def __init__(self, alpha, beta, l1, l2, bits):
		self.z = [0.] * bits
		self.n = [0.] * bits
		self.alpha = alpha
		self.beta = beta
		self.l1 = l1
		self.l2 = l2
		self.w = {}
		self.X = []
		self.y = 0.
		self.bits = bits
		self.Prediction = 0.
	
	def sgn(self, x):
		if x < 0:
			return -1  
		else:
			return 1

	def fit(self,line):
		try:
			self.ID = line['ID']
			del line['ID']
		except:
			pass

		try:
			#if self.y == 1.:
			#	print line['ObjectType'], self.y, line['IsClick']
			self.y = float(line['IsClick'])
			del line['IsClick']
		except:
			pass

		#del line['HistCTR']
		#title = line['Title']
		del line['SearchID']
		title = line['Title']
		del line['Title']

		#print len(line)	
		self.X = [0.] * len(line)
		for i, key in enumerate(line):
			val = line[key]
			self.X[i] = (abs(hash(key + '_' + val)) % self.bits)
		self.X = [0] + self.X

		for word in title.split(" "):
			self.X.append(abs(hash('Title'+'_'+word)) % self.bits)

	def logloss(self):
		act = self.y
		pred = self.Prediction
		predicted = max(min(pred, 1. - 10e-15), 10e-15)
		return -log(predicted) if act == 1. else -log(1. - predicted)

	def predict(self):
		W_dot_x = 0.
		w = {}
		for i in self.X:
			if abs(self.z[i]) <= self.l1:
				w[i] = 0.
			else:
				w[i] = (self.sgn(self.z[i]) * self.l1 - self.z[i]) / (((self.beta + sqrt(self.n[i]))/self.alpha) + self.l2)
			W_dot_x += w[i]
		self.w = w
		self.Prediction = 1. / (1. + exp(-max(min(W_dot_x, 35.), -35.)))
		return self.Prediction

	def update(self, prediction): 
		for i in self.X:
			g = (prediction - self.y) #* i
			sigma = (1./self.alpha) * (sqrt(self.n[i] + g*g) - sqrt(self.n[i]))
			self.z[i] += g - sigma*self.w[i]
			self.n[i] += g*g

if __name__ == '__main__':

	"""
	SearchID	AdID	Position	ObjectType	HistCTR	IsClick
	"""
	train = '../Data/trainSearchStream.tsv'

	# craeting a dict based on ads info #
	ads_dict= {}
	ads_reader = DictReader(open("../Data/AdsPreProcessed.tsv"))
	for row in ads_reader:
		ads_dict[row['AdID']] = {'Price':row['Price'], 'CategoryID':row['CategoryID'], 'NumParams':row['NumParams'], 'CatLevel':row['CatLevel'], 'ParentCategoryID':row['ParentCategoryID'], 'SubCategoryID':row['SubCategoryID'], 'Title':row['Title']  }

	# reading the search info stored as pkl files #
	pkl_index = 1
	pkl_file = open("./PklFiles/search_dict_pkl"+str(pkl_index)+".p", "rb")
	search_dict = pickle.load( pkl_file)
	pkl_file.close()

	# calling the ftrl function #
	clf = ftrl(alpha = 0.05, 
			   beta = 1., 
			   l1 = 0.0,
			   l2 = 1.0, 
			   bits = 2**25)

	print "Training.."
	random.seed(1234)
	loss = 0.
	total_count = 0
	count = 0
	except_count = 0
	for t, line in enumerate(DictReader(open(train), delimiter='\t')):

		# use only those rows where click info is present #
		try:
			total_count += 1
			float(line['IsClick'])
		except:
			continue

		#if random.choice([1,2,3]) == 1:
		#	continue

		line['Price'] = ads_dict[line['AdID']]['Price']
		line['CategoryID'] = ads_dict[line['AdID']]['CategoryID']
		line['NumParams'] = ads_dict[line['AdID']]['NumParams']
		line['CatLevel'] = ads_dict[line['AdID']]['CatLevel']
		line['ParentCategoryID'] = ads_dict[line['AdID']]['ParentCategoryID']
		line['SubCategoryID'] = ads_dict[line['AdID']]['SubCategoryID']
		line['Title'] = ads_dict[line['AdID']]['Title']
		line['AdPos'] = line['AdID'] + "_" + line['Position']

		if int(line['SearchID']) > pkl_index*2500000 and not search_dict.has_key(int(line['SearchID'])):
			print "Reading next pickle file : SearchID, pkl index : ", int(line['SearchID']), pkl_index
                        pkl_index += 1
                        search_dict = {}
                        pkl_file = open("./PklFiles/search_dict_pkl"+str(pkl_index)+".p", "rb")
                        search_dict = pickle.load( pkl_file)
                        pkl_file.close()
                        print "Pkl loaded.."
		
		try:
			search_list = search_dict[int(line['SearchID'])]
		except:
			except_count += 1
			search_list = ['-999', '-999', '-999', '-999', '-999', '-999', '-999', '-999', '-999', '-999', '-999', '-999', '-999']

		line['IPID'] = str(search_list[0])
		line['UserID'] = str(search_list[1])
		line['UserLoggedOn'] = str(search_list[2])
		line['LocationID'] = str(search_list[3])
		line['WeekDay'] = str(search_list[4])
		line['Hour'] = str(search_list[5])
		line['UserAgentID'] = str(search_list[6])
		line['UserAgentOSID'] = str(search_list[7])
		line['UserAgentDeviceID'] = str(search_list[8])
		line['UserAgentFamilyID'] = str(search_list[9])
		line['LocLevel'] = str(search_list[10])
		line['RegionID'] = str(search_list[11])
		line['CityID'] = str(search_list[12])

		clf.fit(line)
		pred = clf.predict()
		loss += clf.logloss()
		clf.update(pred)
		count += 1
		if count%100000 == 0: 
			print ("(seen, loss) : ", (count, loss * 1./count))
			print "Total count, Except count is : ", total_count, except_count



	print "Final Train values are : "
	print ("(seen, loss) : ", (count, loss * 1./count))
	print "Total count is : ", total_count

	print "Testing..."

	# reading the search info stored as pkl files #
	search_dict = {}
        pkl_index = 1
        pkl_file = open("./PklFiles/search_dict_pkl"+str(pkl_index)+".p", "rb")
        search_dict = pickle.load(pkl_file)
        pkl_file.close()

	test = '../Data/testSearchStream.tsv'
	except_count = 0 
	with open('temp.csv', 'w') as output:
		for t, line in enumerate(DictReader(open(test), delimiter='\t')):

			try:
				line['Price'] = ads_dict[line['AdID']]['Price']
                		line['CategoryID'] = ads_dict[line['AdID']]['CategoryID']
                		line['NumParams'] = ads_dict[line['AdID']]['NumParams']
				line['CatLevel'] = ads_dict[line['AdID']]['CatLevel']
                		line['ParentCategoryID'] = ads_dict[line['AdID']]['ParentCategoryID']
                		line['SubCategoryID'] = ads_dict[line['AdID']]['SubCategoryID']
				line['Title'] = ads_dict[line['AdID']]['Title']
				line['AdPos'] = line['AdID'] + "_" + line['Position']
			except:
				line['Price'] = '0'
				line['CategoryID'] = '0'
				line['NumParams'] = '0'
				line['CatLevel'] = '0'
				line['ParentCategoryID'] = '0'
				line['SubCategoryID'] = '0'
				line['Title'] = '0'
				line['AdPos'] = '0_0'

			if int(line['SearchID']) > pkl_index*2500000 and not search_dict.has_key(int(line['SearchID'])):
				print "Except count is : ", except_count 
				print "Reading next pickle file : SearchID, pkl index : ", int(line['SearchID']), pkl_index
				pkl_index += 1
				search_dict = {}
				pkl_file = open("./PklFiles/search_dict_pkl"+str(pkl_index)+".p", "rb")
				search_dict = pickle.load(pkl_file)
				pkl_file.close()
				print "Pkl loaded.."

			try:
	                        search_list = search_dict[int(line['SearchID'])]
        	        except:
				except_count += 1
                	        search_list = ['-999', '-999', '-999', '-999', '-999', '-999', '-999', '-999', '-999', '-999', '-999', '-999', '-999']

			line['IPID'] = str(search_list[0])
			line['UserID'] = str(search_list[1])
			line['UserLoggedOn'] = str(search_list[2])
			line['LocationID'] = str(search_list[3])
			line['WeekDay'] = str(search_list[4])
			line['Hour'] = str(search_list[5])
			line['UserAgentID'] = str(search_list[6])
			line['UserAgentOSID'] = str(search_list[7])
			line['UserAgentDeviceID'] = str(search_list[8])
			line['UserAgentFamilyID'] = str(search_list[9])
			line['LocLevel'] = str(search_list[10])
			line['RegionID'] = str(search_list[11])
			line['CityID'] = str(search_list[12])



			clf.fit(line)
			output.write('%s\n' % str(clf.predict()))
	print "Done.. Writing.."

