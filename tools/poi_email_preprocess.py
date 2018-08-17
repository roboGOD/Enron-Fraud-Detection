
import sys
from poi_email_addresses import poiEmails
sys.path.append("../tools")
from parse_out_email_text import parseOutText
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectPercentile, f_classif


### dumpEmailData: Operations Performed:
###		Get the email_address of POIs and Non POIs
###		Go to emails_by_address directory
### 	Open files "to" and "from" that email_address(if they exist)
###		Go to the enron mail directory and the specified paths in the file.
###		Fetch the content of the emails
### 	Store it in a dictionary(word_data) with the person name as key
###		After all the emails are done, dump the word_data to pickle file. 
def dump_email_data(data_dict):

	directory = "emails_by_address/"
	counter = 0
	word_data = {}


	ls = poiEmails()
	for key in ls:
		email = ls[key]
		path1 = directory + "from_" + email + ".txt"
		path2 = directory + "to_" + email + ".txt"
		words = ""
		
		try:
			f1 = open(path1, "r")
			ls1 = f1.readlines()
			f1.close()
			for path in ls1:
				path = "../" + path[:-1]
				f2 = open(path, "r")
				words = words + " " + (parseOutText(f2))
				f2.close()

		except Exception:
			pass

		try:
			f1 = open(path2, "r")
			ls1 = f1.readlines()
			f1.close()
			for path in ls1:
				path = "../" + path[:-1]
				f2 = open(path, "r")
				words = words + " " + (parseOutText(f2))
				f2.close()

		except Exception:
			pass

		if words != "":
			if key in word_data:
				word_data[key] = word_data[key] + " " + words
			else:
				word_data[key] = words
			del words

	for key in data_dict:
		email = data_dict[key]['email_address']
		path1 = directory + "from_" + email + ".txt"
		path2 = directory + "to_" + email + ".txt"
		words = ""

		try:
			f1 = open(path1, "r")
			ls = f1.readlines()
			f1.close()
			for path in ls:
				path = "../" + path[:-1]
				f2 = open(path, "r")
				words = words + " " + (parseOutText(f2))
				f2.close()
		except Exception:
			pass

		try:
			f1 = open(path2, "r")
			ls = f1.readlines()
			f1.close()
			for path in l1:
				path = "../" + path[:-1]
				f2 = open(path, "r")
				words = words + " " + (parseOutText(f2))
				f2.close()
		except Exception:
			pass

		if words != "":
			if key in word_data:
				word_data[key] = word_data[key] + " " + words
			else:
				word_data[key] = words
			del words
		
		counter += 1
		print counter

	pickle_out = open("email_data.pkl", "wb")
	pickle.dump(word_data, pickle_out)
	pickle_out.close()

	print "\nSuccess!\nEmail Data Fetched."



### Preprocess the email data
def email_preprocessor(data_dict, word_data_path="email_data.pkl"):

	pickle_in = open(word_data_path, "rb")
	word_data = pickle.load(pickle_in)
	pickle_in.close()

	authors = []
	words = []
	pois = []

	for key in word_data:
		authors.append(key)
		words.append(word_data[key])
		pois.append(data_dict[key]['poi'])


	vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5, stop_words='english')
	words_transformed = vectorizer.fit_transform(words)
	
	########################################################################
	### Perform Some Cleaning here!!!!

	selector = SelectPercentile(f_classif, percentile=10)
	selector.fit(words_transformed, pois)
	words_transformed = selector.transform(words_transformed).toarray()

	### Word Finder
	'''
	from sklearn.tree import DecisionTreeClassifier
	
	clf = DecisionTreeClassifier(min_samples_split = 2)
	clf.fit(words_transformed, pois)

	print "No. of Training Points: ", len(words_transformed)
	print "Accuracy: ", clf.score(words_transformed, pois)

	importances = clf.feature_importances_
	listOfWords = vectorizer.get_feature_names()
	for j,i in enumerate(importances):
		if i > 0.2:
			print "Importance:", round(i,4),
			print "Index:", j,
			print "Item:", listOfWords[j]

	'''

	######################################################################

	out_dict = {}
	for i,j in enumerate(words_transformed):
		out_dict[authors[i]] = j


	pickle_out = open("emails_preprocessed.pkl", "wb")
	pickle.dump(out_dict, pickle_out)
	pickle_out.close()

	print "\nSuccess!\nEmails Preprocessed."

	#print word_data["LAY KENNETH L"]

