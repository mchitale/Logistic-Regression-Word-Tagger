import numpy as np
import math
import sys
import time

start_time = time.time()

def main():

	IPfile = str(sys.argv[1])
	ValidFile = str(sys.argv[2])
	TestFile = str(sys.argv[3])
	TrainOut = str(sys.argv[4])
	TestOut = str(sys.argv[5])
	MetricsOut = str(sys.argv[6])
	numepochs = int(sys.argv[7])
	feat_flag = int(sys.argv[8])


	data_train = np.genfromtxt(open(IPfile,'r'), delimiter = '\t',dtype = object)
	data_test = np.genfromtxt(open(TestFile,'r'), delimiter = '\t',dtype = object)
	data_valid = np.genfromtxt(open(ValidFile,'r'), delimiter = '\t', dtype = object)


	#numepochs = 2

	train_lines = open(IPfile, 'r').readlines()

	nl_indexes_train = find_nl(IPfile)
	nl_indexes_test = find_nl(TestFile)
	nl_indexes_valid = find_nl(ValidFile)

	x_dict = create_vocabulary(data_train)
	labels = create_labels(data_train)

	if feat_flag == 1:
		
		model_1(data_train,data_valid,data_test,numepochs,x_dict,labels,nl_indexes_train,nl_indexes_test,nl_indexes_valid,TrainOut,TestOut,MetricsOut)
	
	elif feat_flag == 2:
		
		model_2(data_train,data_valid,data_test,numepochs,x_dict,labels,nl_indexes_train,nl_indexes_test,nl_indexes_valid,TrainOut,TestOut,MetricsOut)


	print "time taken = ", time.time()-start_time

def model_2(data_train,data_valid,data_test,numepochs,x_dict,labels,nl_indexes_train,nl_indexes_test,nl_indexes_valid,TrainOut,TestOut,MetricsOut):

	y_r = np.shape(labels)
	y_r = y_r[0]
	#Dimensions of theta are 3M + 3, len(labels)
	theta = np.zeros([(3*len(x_dict)),y_r]) 

	m = open(MetricsOut, 'w')
	
	for i in range(0,numepochs):
		n = i+1
		m.write('epoch=')
		m.write(str(n))
		
		theta = SGD_2(data_train, x_dict,labels,theta,nl_indexes_train)
		#print "\n<<<<<<<<LIKELIHOOD(TRAIN):>>>>>>>>\n"
		m.write(' likelihood(train): ')
		likelihood_t = log_likelihood_2(data_train,theta,x_dict,labels,nl_indexes_train)
		m.write(str(round(likelihood_t,6)))
		m.write('\n')

		#print "\n<<<<<<<<LIKELIHOOD(VALIDATION)>>>>>>>>\n"
		m.write('epoch=')
		m.write(str(n))
		m.write(' likelihood(validation): ')
		likelihood_v = log_likelihood_2(data_valid,theta,x_dict,labels,nl_indexes_valid)
		m.write(str(round(likelihood_v,6)))
		m.write('\n')


	pred_train = predict_2(nl_indexes_train,data_train, x_dict, theta, labels,TrainOut)
	error_train = error_rate(data_train,pred_train)
	m.write('error(train): ')
	m.write(str(round(error_train,6)))
	m.write('\n')

	pred_test = predict_2(nl_indexes_test,data_test, x_dict, theta, labels,TestOut)
	error_test = error_rate(data_test,pred_test)
	m.write('error(test): ')
	m.write(str(round(error_test,6)))
	m.write('\n')

def model_1(data_train,data_valid,data_test,numepochs,x_dict,labels,nl_indexes_train,nl_indexes_test,nl_indexes_valid,TrainOut,TestOut,MetricsOut):
	y_r = np.shape(labels)
	y_r = y_r[0]
	theta = np.zeros([len(x_dict),y_r])

	m = open(MetricsOut, 'w')

	for i in range(0,numepochs):
		n = i+1
		m.write('epoch=')
		m.write(str(n))
		theta = SGD_1(data_train, x_dict,labels,theta)
		#print "\n<<<<<<<<LIKELIHOOD(TRAIN):>>>>>>>>\n"
		m.write(' likelihood(train): ')
		likelihood_t = log_likelihood_1(data_train,theta,x_dict,labels)
		m.write(str(round(likelihood_t,6)))
		m.write('\n')

		#print "\n<<<<<<<<LIKELIHOOD(VALIDATION)>>>>>>>>\n"
		m.write('epoch=')
		m.write(str(n))
		m.write(' likelihood(validation): ')
		likelihood_v = log_likelihood_1(data_valid,theta,x_dict,labels)
		m.write(str(round(likelihood_v,6)))
		m.write('\n')


	pred_train = predict_1(nl_indexes_train,data_train, x_dict, theta, labels,TrainOut)
	error_train = error_rate(data_train,pred_train)
	m.write('error(train): ')
	m.write(str(round(error_train,6)))
	m.write('\n')

	pred_test = predict_1(nl_indexes_test,data_test, x_dict, theta, labels,TestOut)
	error_test = error_rate(data_test,pred_test)
	m.write('error(test): ')
	m.write(str(round(error_test,6)))
	m.write('\n')

def SGD_2(data, x_dict, labels, theta, nl_indexes):

	print "<<<Computing SGD>>>"

	data_rows = np.shape(data)
	data_rows = data_rows[0]

	len_labels = np.shape(labels)
	len_labels = len_labels[0]
	gradient = []

	for i in range(0,data_rows):

		#the np array data_rows has a value for every 'i' thus current will never be EOS or BOS
		current = x_dict[data[i][0]]
		
		#to check for Beginning, check if previous index is a new line index
		if i==0 or i in nl_indexes:
			prev = 0
		else:
			prev = x_dict[data[i-1][0]]

		#to check for end of line, check if next index is a new line index
		if i == data_rows-1 or (i+1) in nl_indexes:
			nxt = 0
		else:
			nxt = x_dict[data[i+1][0]]


		denom = 0

		
		for k in range(0,len_labels):
			denom += math.exp(theta[0][k]+theta[current][k]+theta[prev+(len(x_dict))][k]+theta[nxt+(2*(len(x_dict)))][k])


		for j in range(0,len_labels):
			g = -(indicator(data,labels,i,j) - (math.exp(theta[0][j]+theta[current][j]+theta[prev+(len(x_dict))][j]+theta[nxt+(2*(len(x_dict)))][j]))/float(denom))
			gradient.append(g)

		update = 0.5 * np.array(gradient)
		gradient = []

		for k in range(0,len_labels):
			theta[0][k] -= update[k]
			theta[current][k] -= update[k]
			theta[prev+(len(x_dict))][k] -= update[k]
			theta[nxt+(2*(len(x_dict)))][k] -= update[k]


	return theta

def indicator(data,labels,i,j):

	flag = 0

	if data[i][1]==labels[j]:
		flag = 1
	else:
		flag = 0

	return flag

def predict_2(nl_indexes_test,test_data, x_dict, theta, labels,outfile):


	outfile = open(outfile,'w')
	rows_test, cols_test = np.shape(test_data)
	len_labels = np.shape(labels)
	len_labels = len_labels[0]
	predict = []

	#print rows_test, '\n'
	#n=1
	for i in range(0,rows_test):
		
		if i in nl_indexes_test:
			outfile.write('\n')
		
		#n+=1

		likelihoods = []
		#the np array data_rows has a value for every 'i' thus current will never be EOS or BOS
		current = x_dict[test_data[i][0]]
		
		#to check for Beginning, check if previous index is a new line index
		if i==0 or i in nl_indexes_test:
			prev = 0
		else:
			prev = x_dict[test_data[i-1][0]]

		#to check for end of line, check if next index is a new line index
		if i == rows_test-1 or (i+1) in nl_indexes_test:
			nxt = 0
		else:
			nxt = x_dict[test_data[i+1][0]]
		
		for j in range(0,len_labels):
			
			l = (math.exp(theta[0][j]+theta[current][j]+theta[prev+(len(x_dict))][j]+theta[nxt+(2*(len(x_dict)))][j]))

			likelihoods.append(l)

		largest = 0
		op = 0
		for m in range(0,len(likelihoods)):
			if likelihoods[m] > largest:
				largest = likelihoods[m]
				op = m	


		output = labels[op]
		predict.append(output)


		#print output
		outfile.write(output)
		outfile.write('\n')

		

	return predict

def error_rate(test_data,predict):

	rows = len(predict)
	count = 0

	for i in range(0,rows):
		if test_data[i][1] != predict[i]:
			count+=1

	error_rate = count/float(rows)

	#print "error rate = \t", error_rate
	return error_rate

def log_likelihood_2(data,theta,x_dict,labels,nl_indexes):

	data_rows, data_cols = np.shape(data)
	len_labels = np.shape(labels)
	len_labels = len_labels[0]

	loglik = 0

	for i in range(0,data_rows):
		#the np array data_rows has a value for every 'i' thus current will never be EOS or BOS
		current = x_dict[data[i][0]]
		
		#to check for Beginning, check if previous index is a new line index
		if i==0 or i in nl_indexes:
			prev = 0
		else:
			prev = x_dict[data[i-1][0]]

		#to check for end of line, check if next index is a new line index
		if i == data_rows-1 or (i+1) in nl_indexes:
			nxt = 0
		else:
			nxt = x_dict[data[i+1][0]]
		
		denom = 0

		
		for k in range(0,len_labels):
			denom += math.exp(theta[0][k]+theta[current][k]+theta[prev+(len(x_dict))][k]+theta[nxt+(2*(len(x_dict)))][k])


		loglik += sum((math.log((math.exp(theta[0][j]+theta[current][j]+theta[prev+(len(x_dict))][j]+theta[nxt+(2*(len(x_dict)))][j]))/float(denom))*indicator(data,labels,i,j)) for j in range(0,len_labels))
		

	#print 'likelihood : ',-1*loglik/data_rows

	return -1*loglik/data_rows

def create_labels(data_train):
	data_rows, data_cols = np.shape(data_train)
	y = np.empty([data_rows, 1], dtype = object)
	for i in range(0,data_rows):
		y[i] = data_train[i][1]

	#LABELS VECTOR CONTAINING ALL UNIQUE LABELS IN DATA SET
	labels = np.unique(y)
	y_r = np.shape(labels)
	y_r = y_r[0]

	return labels

def create_vocabulary(data_train):
	#FINDING ALL THE UNIQUE FEATURES FROM THE DATA SET
	data_rows, data_cols = np.shape(data_train)

	X = []
	for i in range(0,data_rows):
		X.append(data_train[i][0])

	UX = np.unique(np.array(X))
	u_r = np.shape(UX)
	u_r = u_r[0]

	x_dict = {}

	x_dict['bias'] = 0

	for i in range(0,u_r):

		x_dict[UX[i]] = i+1

	return x_dict

def find_nl(filename_):

	lines = open(filename_,'r').readlines()
	
	nlc = 0
	nl_indexes = []
	for line in lines:
		if line == '\n':
			nl_indexes.append(nlc)
		nlc+=1

	
	new_nl = []
	for i in range(0,len(nl_indexes)):
		nll = nl_indexes[i]-i
		new_nl.append(nll)

	nl_indexes = new_nl

	return nl_indexes

def SGD_1(data, x_dict, labels, theta):

	print "<<<Entered SGD>>>"

	data_rows = np.shape(data)
	data_rows = data_rows[0]

	len_labels = np.shape(labels)
	len_labels = len_labels[0]
	gradient = []

	for i in range(0,data_rows):
		denom = 0
		#index = -1

		if x_dict[data[i][0]]:
			index = x_dict[data[i][0]]

		for k in range(0,len_labels):
			denom += math.exp(theta[0][k]+theta[index][k])

		for j in range(0,len_labels):
			g = -(indicator(data,labels,i,j) - (math.exp(theta[0][j]+theta[index][j]))/float(denom))
			gradient.append(g)

		update = 0.5 * np.array(gradient)
		gradient = []

		for k in range(0,len_labels):
			theta[0][k] -= update[k]
			theta[index][k] -= update[k]


	return theta

def predict_1(nl_indexes,test_data, x_dict, new_theta, labels,outfile):

	
	rows_test, cols_test = np.shape(test_data)
	len_labels = np.shape(labels)
	len_labels = len_labels[0]
	predict = []
	outfile = open(outfile,'w')

	#ipfl = open(filename_,'r').read()

	for i in range(0,rows_test):

		if i in nl_indexes:
			outfile.write('\n')
			
		likelihoods = []
		if x_dict[test_data[i][0]]:
			index = x_dict[test_data[i][0]]

		for j in range(0,len_labels):
			l = math.exp(new_theta[0][j]+new_theta[index][j])
			likelihoods.append(l)

		largest = 0
		op = 0
		for m in range(0,len(likelihoods)):
			if likelihoods[m] > largest:
				largest = likelihoods[m]
				op = m

		output = labels[op]
		predict.append(output)

		#print output
		outfile.write(output)
		outfile.write('\n')
	return predict

def log_likelihood_1(data,new_theta,x_dict,labels):

	data_rows, data_cols = np.shape(data)
	len_labels = np.shape(labels)
	len_labels = len_labels[0]

	loglik = 0

	for i in range(0,data_rows):
		denom = 0
		#index = -1
		if x_dict[data[i][0]]:
			index = x_dict[data[i][0]]

		for k in range(0,len_labels):
			denom += math.exp(new_theta[0][k]+new_theta[index][k])


		loglik += sum((math.log((math.exp(new_theta[0][j]+new_theta[index][j]))/float(denom))*indicator(data,labels,i,j)) for j in range(0,len_labels))
		

	#print -1*loglik/data_rows

	return -1*loglik/data_rows


main()





