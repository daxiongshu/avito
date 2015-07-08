Order of files to run

1. preProcessAds.py - This will create a new file "AdsPreProcessed.tsv" by removing the non-contextual ads from AdsInfo file

2. Create a directory "PklFiles" in the working folder to store the pickle files created in next step

3. preProcessSearch.py - This will create dictionaries from search info, user info,  location and then store it as pickle files which could be used by FTRL model building process.

4. ftrl.py - FTRL code from Tinrtgu with modifications specific to this data set. This will create a temp.csv file.

5. tmpToSub.py - to create submission file from temp.csv
