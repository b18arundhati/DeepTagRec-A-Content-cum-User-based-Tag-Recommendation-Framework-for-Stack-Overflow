import json
import pickle
import re, string
from gensim.models.keyedvectors import KeyedVectors
import numpy as np

# utility functions


# word_model = KeyedVectors.load_word2vec_format('vec_skipgram_ver3_iter30.bin', binary=True, unicode_errors='ignore') 

# clean the question
def clean_question(q):
  
	#nq = re.sub('<code>(.|\n)*?<\/code>', '<code><\/code>', q)
	#code = q - nq
	#print 'given Qs: ',q
	#print 'Code: ',code

	#nq = nq.translate(None, string.punctuation)

	#nq = re.sub('<pre><code><\/code><\/pre>',code,nq)
	html_tags = ['<strong>','<h>','<p>','<ul>','<li>','<ol>','<br>','<h1>','<h2>','<h3>','<h4>','<br />','<br/>','<em>','<blockquote>']
	
	x = re.split('<pre>|</pre>',q)
	#print 'x: '
	#print x
	for i, part in enumerate(x):
		#print i, part
		x[i] = re.sub('<a (.|\n)*?>','',x[i])
		x[i] = re.sub('<\/a>','',x[i])
		for h in html_tags:
			x[i] = re.sub(h, '', x[i])
			rh = '</' + h[1:]
			x[i] = re.sub(rh, '', x[i])
		if not part.startswith('<code>'):
			if '<code>' in part or '</code>' in part:
				x[i] = re.sub('<code>','',x[i])
				x[i] = re.sub('<\/code>','',x[i])
			x[i] = x[i].translate(None,string.punctuation)
	  
	nq = ''.join(x)
	#print 'nq: '
	#print nq
	nq = re.sub('<code>', '', nq)
	nq = re.sub('<\/code>', '', nq)
	#nq = re.sub('\-?\d+','',nq)
	#nq = re.sub('<code>(.|\n)*?<\/code>', '', nq)
	nq = re.sub( '\n', ' ', nq )  
	nq = re.sub( '\s+', ' ', nq ).strip()
	#nq = re.sub( ' \-?\d+ ', ' NUMBER ', nq)



	#for h in html_tags:
	#  nq = re.sub(h, '', nq)
	#  rh = '</' + h[1:]
	#  nq = re.sub(rh, '', nq)

	nq = re.sub( '\s+(\-?\d+)\s+', ' NUMBER ', nq)

	return nq



# clean the question
def clean_hashtags(q):
  
	#nq = re.sub('<code>(.|\n)*?<\/code>', '<code><\/code>', q)
	#code = q - nq
	#print 'given Qs: ',q
	#print 'Code: ',code

	#nq = nq.translate(None, string.punctuation)

	#nq = re.sub('<pre><code><\/code><\/pre>',code,nq)
	html_tags = ['<strong>','<h>','<p>','<ul>','<li>','<ol>','<br>','<h1>','<h2>','<h3>','<h4>','<br />','<br/>','<em>','<blockquote>', '<pre>']
	
	x = re.split('<pre>|</pre>',q)
	#print 'x: '
	#print x
	for i, part in enumerate(x):
		#print i, part
		x[i] = re.sub('<a (.|\n)*?>','',x[i])
		x[i] = re.sub('<\/a>','',x[i])
		for h in html_tags:
			x[i] = re.sub(h, '', x[i])
			rh = '</' + h[1:]
			x[i] = re.sub(rh, '', x[i])
		if not part.startswith('<code>'):
			if '<code>' in part or '</code>' in part:
				x[i] = re.sub('<code>','',x[i])
				x[i] = re.sub('<\/code>','',x[i])
			#x[i] = x[i].translate(None,string.punctuation)
	  
	nq = ''.join(x)
	
	#print 'nq: '
	#print nq
	nq = re.sub('<code>', '', nq)
	nq = re.sub('<\/code>', '', nq)
	#nq = re.sub('\-?\d+','',nq)
	#nq = re.sub('<code>(.|\n)*?<\/code>', '', nq)
	nq = re.sub( '\n', ' ', nq )  
	nq = re.sub( '\s+', ' ', nq ).strip()
	#nq = re.sub( ' \-?\d+ ', ' NUMBER ', nq)



	#for h in html_tags:
	#  nq = re.sub(h, '', nq)
	#  rh = '</' + h[1:]
	#  nq = re.sub(rh, '', nq)

	nq = re.sub( '\s+(\-?\d+)\s+', ' NUMBER ', nq)

	return nq

def traverse_dataset():
	openfile = open("TrainingSetQuestions_5Lakh.pickle", "rb")
	x =  pickle.load(openfile)
	total = 0
	c = 0
	nc = 0
	s1 = set()
	s2 = set()
	for y in x:
		# print y['tags']
		tags = y['tags'].encode('utf-8')
		tags = tags.split('<')
		tags = tags[1:]
		for i in range(len(tags)) :
			tags[i] =  tags[i].rstrip('>')
			
		for t in tags:
			# total = total + 1
			s1.add(t)
			try : 
				vec =  word_model[t]
				
			except:
				
				s2.add(t)
		

		# break
	total = len(s1)
	nc = len(s2)	
	print total
	print nc
	print float(nc)/total

def get_tag_list(tag_string):
	tags = tag_string.split('<')
	tags = tags[1:]
	for i in range(len(tags)) :
		tags[i] =  tags[i].rstrip('>')

	return tags

def all_tags(filename):
	openfile = open(filename, "rb")
	x =  pickle.load(openfile)
	s1 = set()
	for y in x:
		try:
			tags = y['Tags'].encode('utf-8')
		except:
			tags = y['tags'].encode('utf-8')		
		tags = tags.split('<')
		tags = tags[1:]
		for i in range(len(tags)) :
			tags[i] =  tags[i].rstrip('>')
			
		for t in tags:
			s1.add(t)

	l = np.array(list(s1))
	return l


def create_tag_dict(filename):
	# openfile = open("TrainingSetQuestions_5Lakh.pickle", "rb")
	openfile = open(filename, "rb")
	x =  pickle.load(openfile)
	
	s1 = set()
	for y in x:
		try:
			tags = y['Tags'].encode('utf-8')
		except:
			tags = y['tags'].encode('utf-8')		
		tags = tags.split('<')
		tags = tags[1:]
		for i in range(len(tags)) :
			tags[i] =  tags[i].rstrip('>')
			
		for t in tags:
			s1.add(t)
		
	n = len(s1)
	l = list(s1)

	d = {}
	order = {}
	number = {}
	for i in range(len(l)):
		d[l[i]] = np.zeros(n)
		d[l[i]][i] = 1
		order[l[i]] = i
		number[i] = l[i]

	tag_vec = pickle.load(open("Tag_vectors.p", 'rb'))
	tag_vecs = []
	for i in range(len(l)):
	    try:	
                tag_vecs.append(tag_vec[l[i]])	
	    except:
                tag_vecs.append(np.zeros(300))
        tag_vecs = np.array(tag_vecs)	

	# print d[l[n-1]]
	return d, order, number
	

def create_tag_dict_2(filename):
	# openfile = open("TrainingSetQuestions_5Lakh.pickle", "rb")
	openfile = open(filename, "rb")
	x =  pickle.load(openfile)
	
	s1 = set()
	for y in x:
		try:
			tags = y['Tags'].encode('utf-8')
		except:
			tags = y['tags'].encode('utf-8')		
		tags = tags.split('<')
		tags = tags[1:]
		for i in range(len(tags)) :
			tags[i] =  tags[i].rstrip('>')
			
		for t in tags:
			s1.add(t)
		
	n = len(s1)
	l = list(s1)

	d = {}
	order = {}
	for i in range(len(l)):
		d[l[i]] = np.zeros(n)
		d[l[i]][i] = 1
		order[l[i]] = i


	tag_vec = pickle.load(open("Tag_vectors.p", 'rb'))
	tag_vecs = []
	for i in range(len(l)):
	    try:	
                tag_vecs.append(tag_vec[l[i]])	
	    except:
                tag_vecs.append(np.zeros(300))
	tag_vecs = np.array(tag_vecs)	

	# print d[l[n-1]]
	# return d, order, l
	return d,tag_vecs
	

def generate_dataset_1(pickle_filename,json_filename):
	data = []

	pickle_file = pickle.load(open(pickle_filename, "rb"))
	cnt = 0
	for y in pickle_file:
		cnt += 1
		e = {}
		question = y['Body'].encode('utf-8')
		question = clean_question(question)

		t = None
		try:
			t = y['tags'].encode('utf-8')
		except:
			t = y['Tags'].encode('utf-8')

		tags = get_tag_list(t)

		e['question'] = question
		e['tags'] = tags

		data.append(e)

	with open(json_filename, 'w') as fout:
	    json.dump(data, fout)


	print "count = ", cnt

if __name__ == "__main__":
	# d = create_tag_dict('TrainingSetQuestions_5Lakh.pickle')
	# with open('tag_dict.json', 'w') as f:
	#     json.dump(d, f)
	# generate_dataset_1('TrainingQuestionsProcessed.pickle','training_old_dataset.json')
	generate_dataset_1('ValidationPosts_15000.pickle','validation_old_dataset.json')
	generate_dataset_1('Test.pickle','test_old_dataset.json')

	
