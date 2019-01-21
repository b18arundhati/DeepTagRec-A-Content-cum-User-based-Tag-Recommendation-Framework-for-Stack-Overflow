import numpy as np
import utils
import h5py
from keras.models import Model
from keras.layers import Bidirectional, GRU, Reshape, RepeatVector, dot, Dot, Input, LSTM, merge, Dense, TimeDistributed, Dropout, Embedding, Activation, Reshape, Lambda, Flatten, Add, Multiply, Concatenate
import cPickle as pickle
from keras import optimizers
from gensim.models.keyedvectors import KeyedVectors
from keras.models import load_model
from keras.activations import softmax
#import gensim

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.7
set_session(tf.Session(config=config))

# basic model for tag recommendation
# Embedding layer -> BiLSTM -> Dense with softmax

# word model
#embeddings_file_bin = '../glove/vectors.bin'
#word_model = KeyedVectors.load_word2vec_format('../glove/vectors.txt', binary=False, unicode_errors='ignore')
word_model = KeyedVectors.load_word2vec_format('word2vec/vec_Body_Title.bin', binary=True, unicode_errors='ignore') 
#meta_model = KeyedVectors.load_word2vec_format('metapath2vec/code_metapath2vec/stack_new_1000', binary=True, unicode_errors='ignore')  
user_id = pickle.load(open("user.p", 'rb'))
user_tag = pickle.load(open("user_tags.p", 'rb'))
user_num = pickle.load(open("user_num.p", 'rb'))



count = len(user_tag)

meta_model = {}
openfile = open("graph_train.emd", 'r')
for line in openfile:
	arr = line.split()
	meta_model[arr[0]] = arr[1:]

#print meta_model['0']
openfile.close()
#word_model = {}
#file = open("word2vec/vec_Body_Title.txt",'r')
#for line in file:
#	arr = line.split(" ")
#	if len(arr) > 300:
#		word_model[arr[0]] = map(float, arr[1:len(arr)-1])

#file.close()



# maximum sentence length is taken to be 300

MAX_LENGTH = 300
blank = np.zeros(300)
best = 5
beta = 1.0


#filename = "Training_Body_Title_same.p"#"TrainingSetQuestions_5Lakh.pickle"
tag_dict, order, set_tag, _ = utils.create_tag_dict("Body_Title_same.p")


'''
tag_vecs = []

length = len(tag_dict)

#two = 0 
for i in xrange(length):
	try:
		tag_vecs.append(np.asarray(meta_model[str(i+count)], dtype=float))
	except:
		#print set_tag[i]
		#two += 1
		tag_vecs.append(np.zeros(128))	
#print two
tag_vecs = np.asarray(tag_vecs)
'''


num_tags = vec_size = len(tag_dict)
print num_tags
# overall batch size for memory purposes
batch_size = 10000

# dropout rate
dropout_rate = 0.5


# get question embedding
def get_question_embedding(text):
	l = text.split()
	encoding = []
	
	if len(l) < 300 :
		for i in range(300 - len(l)):
			encoding.append(blank)

	n = min(300,len(l))
	for i in range(n) : 
		t = l[i]
		try : 
			enc = word_model[t]
		except :
			enc = blank


		encoding.append(enc)

	encoding = np.asarray(encoding)
	return encoding


def get_tag_encoding(tag):
	enc = np.zeros(num_tags)
	for t in tag:
		enc = np.add(enc,tag_dict[t])

	return enc


# model
'''
question_input = Input(shape=(MAX_LENGTH,300), name='question_input')
user_input = Input(shape=(128,), name='user_input')
gru_forward = GRU(units = 1000, name="gru")(question_input)

lstm_concat = gru_forward#Flatten()(gru_forward)

user_layer1 = Dense(units=1000, activation='relu', name='user_layer1')(user_input)
Adder = Add()([user_layer1, lstm_concat])
# Merged_layer = Concatenate()([user_layer1, lstm_concat])
tag_output = Dense(units=num_tags, activation="softmax", name="tag_output")(Adder)
# tag_output = Dense(units=num_tags, activation="softmax", name="tag_output")(Merged_layer)

model = Model(inputs=[question_input, user_input], outputs=[tag_output])
rmsprop = optimizers.RMSprop(lr=0.001, decay=1e-6)#, momentum=0.9, nesterov=True)
model.compile(optimizer=rmsprop, loss="binary_crossentropy", metrics=["accuracy"])
'''

#model_rel = load_model('model4_User_tag.h5')
#model_rel.load_weights('model4_User_tag_weights.h5')

'''
tag_input = Input(shape=(1, 300), name='tag_input')
tag_vec = Input(shape=(vec_size, 300), name='tag_vec')

score = Dot( axes=(2,2), normalize = True)([tag_input, tag_vec])#merge([ques_vec, tag_vec], mode='dot',dot_axes=(1,1))#, normalize = True)
#print score.shape
#score = Reshape((vec_size,))(score)
#score = Activation("softmax")(score)
#print score.shape
model_rel = Model(inputs=[tag_input,tag_vec], outputs=[score])
'''

'''
tag_input = Input(shape=(5, 300), name='tag_input')
tag_vec = Input(shape=(vec_size, 300), name='tag_vec')

score = Dot( axes=(2,2), normalize = False)([tag_input, tag_vec])#merge([ques_vec, tag_vec], mode='dot',dot_axes=(1,1))#, normalize = True)
print score.shape
#score = Reshape((vec_size,))(score)
score = Activation("softmax")(score)
print score.shape
model_rel = Model(inputs=[tag_input,tag_vec], outputs=[score])
'''


def train():
	openfile = open("Training_Body_Title_user.p", "rb")
	x =  pickle.load(openfile)


	for loop in xrange(5):
		x_train = []
		user = []
		y_train = []
		cnt = 0
		trace = 0
		for y in x :
			try:
				question = y['Title'].encode('utf-8') + ' '		
			except:
				question = ''
			question = question + y['Body'].encode('utf-8')
			question = utils.clean_question(question)
			
			tag_string = y['tags'].encode('utf-8')
			#print tag_string
			tag_list = utils.get_tag_list(tag_string)
			
			question_enc = get_question_embedding(question)
			tag_enc = get_tag_encoding(tag_list)
			
			cnt = cnt + 1
			x_train.append(question_enc)
			y_train.append(tag_enc)
			try:
				user.append(meta_model[str(user_num[y['OwnerUserId']])])
			except:
				#trace += 1
				#print trace
				user.append(np.zeros(128))	

			if cnt == batch_size:
				x_train = np.asarray(x_train)
				y_train = np.asarray(y_train)
				user = np.asarray(user)

				# print (x_train.shape)
				model.fit([x_train, user],y_train, epochs=1)
				model.save('model4_train_add_Body_Title_gru_epochs10.h5')
				model.save_weights('model4_train_add_Body_Title_weights_gru_epochs10.h5')

				x_train = []
				user = []
				y_train = []

				cnt = 0

	#x_train = np.asarray(x_train)
	#y_train = np.asarray(y_train)

	#model.fit(x_train,y_train, epochs=1)
	#model.save('model4_Body_Title_epochs1.h5')
	#model.save_weights('model4_Body_Title_weights_epochs1.h5')

	#x_train = []
	#y_train = []

	#cnt = 0


def get_key(val):
	print val
	print type(val)
	for k,v in tag_dict.iteritems():
		if (v == val).all():
			return k

def run_user(predicted, user):
	total = len(predicted)
	new_predicted = []
	best_tags = 50

	for i in range(total):
		#print i 
		global tag_vecs
		#arr = predicted[i].argsort()[-best:][::-1]
		#arr = predicted[i].argsort()[-best:][::-1]

		#tag_rep = []
		#for pred in arr:
		#	tag_rep.append(tag_vecs[pred])
		#tag_rep = np.asarray(tag_rep)

		scores = model_rel.predict(np.asarray([user[i]]))[0]
		#arr = scores.argsort()[-best_tags:][::-1]
			
		#sum_ = np.zeros(vec_size)
		#for pred in arr:
		#	sum_[pred] = 1
		#sum_ = sum_ * predicted[i]

		
		#print scores.shape
		sum_ = beta * predicted[i]
		sum_ = sum_ + (1.0 - beta) * scores
		#for score in scores:
		#sum = np.zeros(vec_size)
		#sum += actual
		#ind = 0
		#for pred in arr:
			#print ind
		#	sum_ = sum_ + predicted[i][pred] * (1.0 - beta) * scores[ind]
		#	ind += 1

		new_predicted.append(sum_)

	new_predicted = np.asarray(new_predicted)
	dict1 = {}
	for i in xrange(len(new_predicted)):
		dict1[i] = new_predicted[i]
	pickle.dump(dict1, open("predicted.p",'wb'))
	
	return new_predicted		

def user_rep():
	openfile = open("Training_Body_Title_user.p", "rb")
	x =  pickle.load(openfile)

	repre = {}
	count = 0

	use = 0

	for y in x :
		tag_string = y['tags'].encode('utf-8')
		#print tag_string
		tag_list = utils.get_tag_list(tag_string)
		tag_enc = get_tag_encoding(tag_list)

		count += 1
		print count

		try:
			repre[user_id[y['OwnerUserId']]] += tag_enc
		except:
			try:
				repre[user_id[y['OwnerUserId']]] = np.zeros(len(tag_dict))
				repre[user_id[y['OwnerUserId']]] += tag_enc
			except:
			#	use += 1
			#	print use
				continue	

	for key in repre:
		repre[key] = softmax(repre[key])
		print repre[key].shape

	return repre


# correct solution:
def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0) # only difference


def run(predicted):
	total = len(predicted)
	new_predicted = []
	#best_tags = 50

	for i in range(total):
		#print i 
		global tag_vecs
		arr = predicted[i].argsort()[-best:][::-1]

		tag_rep = []
		for pred in arr:
			tag_rep.append(tag_vecs[pred])
		tag_rep = np.asarray(tag_rep)

		scores = model_rel.predict([np.asarray([tag_rep]), np.asarray([tag_vecs])])[0]
		
		
		print scores.shape
		sum_ = beta * predicted[i]
		#for score in scores:
		#sum = np.zeros(vec_size)
		#sum += actual
		ind = 0
		for pred in arr:
			#print ind
			sum_ = sum_ + predicted[i][pred] * (1.0 - beta) * scores[ind]
			ind += 1

		new_predicted.append(sum_)
	new_predicted = np.asarray(new_predicted)
	'''
	dict1 = {}
	for i in xrange(len(new_predicted)):
		dict1[i] = new_predicted[i]
	pickle.dump(dict1, open("predicted.p",'wb'))
	'''
	return new_predicted	

def evaluate(actual, predicted, k):
	temp = range(num_tags)
	
	
	total = len(predicted)
	precision = 0.0
	recall = 0.0


	for i in range(total):
		#print i
		num = 0
		den_p = 0
		den_r = 0
		# print actual[i]
		# print i," checked"
		#k = 10
		r = sorted(zip(predicted[i], temp), reverse=True)[:k]
		p = np.zeros(num_tags)

		for loop in xrange(k):
			p[r[loop][1]] = 1
		

		#print get_key(p),"\t",get_key(actual[i])
		#if i == 10:
		#	exit(-1)
		#	break
		
		for j in range(num_tags):
			if p[j] == 1:
				den_p += 1
			if actual[i][j] == 1:
				den_r += 1
			if p[j] == 1 and actual[i][j]==1:
				num += 1
		precision += (float)(num)/den_p
		recall += (float)(num)/den_r
		# break
		
	'''	
	#return (num, den_p, den_r, total)
	'''
	precision = precision/total
	recall = recall/total
	
	return precision, recall, total
	#return 0,0,1

def calc_precision_new(actual, predicted):
	temp = range(num_tags)
	cnt = 0
	total = len(predicted)
	for i in range(total):
		# print i," checked"
		r = sorted(zip(predicted[i], temp), reverse=True)[:5]
		p = np.zeros(num_tags)
		p[r[0][1]] = 1
		p[r[1][1]] = 1
		p[r[2][1]] = 1
		p[r[3][1]] = 1
		p[r[4][1]] = 1

		for j in range(num_tags):
			if p[j] == 1 and actual[i][j] == 1:
				cnt = cnt + 1
				break

	return (cnt,total)

def calc_precision(actual, predicted):
	temp = range(num_tags)
	cnt = 0
	total = len(predicted)
	for i in range(total):
		# print i," checked"
		r = sorted(zip(predicted[i], temp), reverse=True)[:5]

		p1 = np.zeros(num_tags)
		p2 = np.zeros(num_tags)
		p3 = np.zeros(num_tags)
		p4 = np.zeros(num_tags)
		p5 = np.zeros(num_tags)

		p1[r[0][1]] = 1
		p2[r[1][1]] = 1
		p3[r[2][1]] = 1
		p4[r[3][1]] = 1
		p5[r[4][1]] = 1

		p1 = get_key(p1)
		p2 = get_key(p2)
		p3 = get_key(p3)
		p4 = get_key(p4)
		p5 = get_key(p5)

		for a in actual[i]:
			if a==p1 or a==p2 or a==p3 or a==p4 or a==p5:
				cnt += 1

	return (cnt, total)


def find_accs(actual, predicted):
	temp = range(num_tags)
	
	
	total = len(predicted)
	top3 = 0
	top5 = 0
	top10 = 0
	ex1 = 0
	ex2 = 0
	ex3 = 0
	ex4 = 0
	ex5 = 0
	index_locs = []
	for i in range(total):
		#print i
		index_locs = []
		num = 0
		den_p = 0
		den_r = 0
		# print actual[i]
		# print i," checked"
		#k = 10
		r = sorted(zip(predicted[i], temp), reverse=True)[:10]
		p = np.zeros(num_tags)

		for loop in xrange(10):
			p[r[loop][1]] = 1
			index_locs.append(r[loop][1])
	
		for _i in range(3):
			if actual[i][index_locs[_i]] == 1:
				top3+=1
				break
		for _i in range(5):
			if actual[i][index_locs[_i]] == 1:
				top5+=1
				break
		for _i in range(10):
			if actual[i][index_locs[_i]] == 1:
				top10+=1
				break
		
		if actual[i][index_locs[0]] == 1:
			ex1 += 1
		if actual[i][index_locs[1]] == 1:
			ex2 += 1
		if actual[i][index_locs[2]] == 1:
			ex3 += 1
		if actual[i][index_locs[3]] == 1:
			ex4 += 1
		if actual[i][index_locs[4]] == 1:
			ex5 += 1


	print("top k acc  k =3 : " + str(top3))
	print("top k acc  k =5 : " + str(top5))
	print("top k acc  k =10 : " + str(top10))
	print("ex k acc  k =1 : " + str(ex1))
	print("ex k acc  k =2 : " + str(ex2))
	print("ex k acc  k =3 : " + str(ex3))
	print("ex k acc  k =4 : " + str(ex4))
	print("ex k acc  k =5 : " + str(ex5))
	



def test():
	# openfile = open("ValidationPosts_15000.pickle", "rb")
	openfile = open("cmnt_high.p", "rb")
	x =  pickle.load(openfile)

	x_test = []
	user = []
	actual = []
	cnt = 0
	correct = 0
	precision = 0.0
	recall = 0.0
	total = 0
	count = 0
	for y in x :
		#if total > 5:
		#	break
		#question = y['Body'].encode('utf-8')
		
		question = y['Title'].encode('utf-8')
		question = question + ' ' + y['Body'].encode('utf-8')
		question = utils.clean_question(question)
		
		tag_string = y['Tags'].encode('utf-8')
		tag_list = utils.get_tag_list(tag_string)
		
		question_enc = get_question_embedding(question)
		tag_enc = get_tag_encoding(tag_list)
		try:
			user.append(meta_model[str(user_num[y['OwnerUserId']])])
		except:
			#trace += 1
			#print trace
			user.append(np.zeros(128))

		cnt = cnt + 1
		#print cnt
		x_test.append(question_enc)
		actual.append(np.asarray(tag_enc))
		#print(cnt)
		
			
	user = np.asarray(user)
	x_test = np.asarray(x_test)
	#s = model.predict([x_test, user])
	s = model.predict(x_test)
	actual = np.asarray(actual)
	predicted = s#run_user(s, user)
			#print predicted
			#break
			# (t_correct, t_total) = calc_precision_new(actual,s)
			#(t_num, t_den_p, t_den_r, t_total) = evaluate(actual,s)
			
			#num += t_num
			#den_p += t_den_p
			#den_r += t_den_r
			# correct += t_correct
			#total += t_total
			# print "correct = ", correct
			#print "total = ", total
			#print "============="
			#predicted = []
			#dict1 = pickle.load(open("predicted.p", 'rb'))
			#for i in dict1.keys():
			#	predicted.append(dict1[i])
			#predicted = np.asarray(predicted)
			#actual = np.asarray(actual)
			#'''
			#print 'I m done'
			#break
			#break
	
	for i in [3, 5, 10]:		
		#print actual
		precision = 0.0
		recall = 0.0
		total = 0
		
		p, r, t = evaluate(actual, predicted, i)
		precision += p*t
		recall += r*t
		total += t		

		

		precision = precision/total
		recall = recall/total
		#precision = (float(num)/den_p)/ float(total)
		#recall = (float(num)/den_r)/ float(total)

		print "Precision @" + str(i) +": ", precision
		print "Recall @" + str(i) +": ", recall
	
	#print("Predicted")

	#find_accs(actual, predicted)

if __name__=="__main__":
	#train()
	# model1_Body_Title_gru_epochs10.h5
	model = load_model('model1_Body_Title_gru_epochs10.h5')
	model.load_weights('model1_Body_Title_weights_gru_epochs10.h5')
	#repre = user_rep()
	# train()
	
	#train()
	test()
	# print(get_question_embedding("Hello"))
