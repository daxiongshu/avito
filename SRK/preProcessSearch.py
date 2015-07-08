# -*- coding: utf-8 -*-
import csv
import cPickle as pickle
from time import sleep
from datetime import datetime 

if __name__ == "__main__":
	print "Creating user dict.."
	user_dict = {}
	user_file = open("../Data/UserInfo.tsv")
	for row in csv.DictReader(user_file, delimiter='\t'):
		user_id = int(row['UserID'])
		user_dict[user_id] = [int(row['UserAgentID']), int(row['UserAgentOSID']), int(row['UserDeviceID']), int(row['UserAgentFamilyID'])]
	user_file.close()

	print "Creating location dict.."
	location_dict = {}
	location_file = open("../Data/Location.tsv")
	for row in csv.DictReader(location_file, delimiter='\t'):
		location_id = int(row['LocationID'])
		try:
			location_dict[location_id] = [int(row['Level']), int(row['RegionID']), int(row['CityID'])]
		except:
			location_dict[location_id] = [int(row['Level']), row['RegionID'], row['CityID']]
	location_file.close()

	print "Creating Pickle files.."
	search_dict = {}
	search_file = open("../Data/SearchInfo.tsv")
	pkl_index = 1
	count = 0 
	for row in csv.DictReader(search_file, delimiter='\t'):
		search_id = int(row['SearchID'])
		out_list = []

		try:
			out_list.append( int(row['IPID']) )
		except:
			out_list.append( -999 )

		try:
                        out_list.append( int(row['UserID']) )
                except:
                        out_list.append( -999 )

		try:
                        out_list.append( int(row['IsUserLoggedOn']) )
                except:
                        out_list.append( -999 )

		try:
                        out_list.append( int(row['LocationID']) )
                except:
                        out_list.append( -999 )

		try:
                        out_list.append( datetime.strptime(row['SearchDate'][:10], "%Y-%m-%d").weekday() )
                except:
                        out_list.append( -999 )

		try:
                        out_list.append( int(row['SearchDate'][11:13]) )
                except:
                        out_list.append( -999 )

		try:
			out_list.extend(user_dict[ int(row['UserID']) ])
		except:
			out_list.extend( [-999,-999,-999,-999] )

		try:
			out_list.extend(location_dict[ int(row['LocationID']) ])
		except:
			out_list.extend( [-999,-999,-999] )

		#print out_list
		search_dict[search_id] = out_list[:]
		#break

		count += 1
		if count % 500000 == 0:
			print "Processed : ", count

		if search_id >= pkl_index*2500000 :
			print "Writing pickle file with index : ", pkl_index
			pkl_file = open("./PklFiles/search_dict_pkl"+str(pkl_index)+".p", "wb")
			pickle.dump( search_dict, pkl_file)
			pkl_file.close()
			search_dict = {}
			pkl_index += 1
			print "Search dict is : ", search_dict
			print "Pickle index is : ", pkl_index

	pkl_file = open("./PklFiles/search_dict_pkl"+str(pkl_index)+".p", "wb")
	pickle.dump( search_dict, pkl_file)
	pkl_file.close()	

	print "Total Processed : ", count
	
