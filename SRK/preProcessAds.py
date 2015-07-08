# -*- coding: utf-8 -*-
from math import sqrt, exp, log
from csv import DictReader,writer
import pandas as pd
import numpy as np

if __name__ == "__main__":
	adsfile = "../Data/AdsInfo.tsv"
	outfile = writer(open("../Data/AdsPreProcessed.tsv","w"))
	outfile.writerow(['AdID','Price','CategoryID','NumParams','Title', 'CatLevel', 'ParentCategoryID', 'SubCategoryID'])

	# Reading Categories file and store in a dict #
	catfile = "../Data/Category.tsv"
	tsv_reader = DictReader(open(catfile), delimiter='\t')
	cat_dict = {}
	for row in tsv_reader:
		cat_dict[row['CategoryID']] = [row['Level'], row['ParentCategoryID'], row['SubcategoryID']]

	count = 0
	total_count = 0
	for t, line in enumerate(DictReader(open(adsfile), delimiter='\t')):
		total_count += 1
		if line['IsContext'] == '1':
			count += 1
			try:
				num_params = len(eval(line['Params']).keys())
			except:
				num_params = 1
			out_row = [line['AdID'], line['Price'], line['CategoryID'], num_params, line['Title'].replace(",","") ] 

			try:
				out_row.extend( cat_dict[line['CategoryID']] )
			except KeyError:
				out_row.extend( [-999, -999, -999] )
			outfile.writerow( out_row )
			#break
			if count % 100000 == 0:
				print count, total_count
