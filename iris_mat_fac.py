import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import time as ti

from sklearn import datasets

def task(N,D,T):
	a=np.zeros((len(N),len(N[0])))
	if (T == "d"): #len(N)==len(D) and len(N[0])==len(D[0]) and 
		for i in range(len(N)):
			for j in range(len(N[0])):
				a[i][j]=N[i][j]/D[i][j]
	elif (T == "m"):
		for i in range(len(N)):
			for j in range(len(N[0])):
				a[i][j]=N[i][j]*D[i][j]			
	return a #else return zero		


def mat_fac(X,K,S=7000):
	st = ti.time()
	W = np.random.rand(len(X),K)
	H = np.random.rand(K,len(X[0]))
	(s,e,ne)=(0,0,0)
	print "W",W, "\n"
	print "H", H, "\n"

	while (s<S):	
		
		h1 = np.dot(np.transpose(W),task(X,np.dot(W,H),"d"))
		h2 = np.dot(np.transpose(W),np.ones((len(W),len(h1[0]))))
		h3 = task(h1,h2,"d")
		H = task(H,h3,"m")

		w1 = np.dot(task(X,np.dot(W,H),"d"),np.transpose(H))
		w2 = np.dot(np.ones((len(w1),len(H[0]))),np.transpose(H))
		w3 = task(w1,w2,"d")
		W = task(W,w3,"m")

		nX = np.dot(W,H)
		for i in range(len(X)):
			for j in range(len(X[i])):
				#ne = ne + pow((X[i][j]-nX[i][j]),2)
				ne = ne + (X[i][j]*np.log((X[i][j]+0.0)/(nX[i][j]+0.0))-X[i][j]+nX[i][j])
		print ne
		if np.abs(e-ne)>0.00001:
		#if np.abs(ne)>0:
			e=ne
			ne=0
		else:
			break
		s+=1
	print "step =", s
	print "time taken =", ti.time()-st
	return (W,H,nX)

iris = np.transpose(np.array(datasets.load_iris()['data'],dtype='f'))
i1 = iris
print iris, "\n"

(nW,nH,nX) = mat_fac(iris,2)
print "new W :", "\n", nW, "\n"
print "new H :", "\n", nH, "\n"
print "new Matrix :", "\n", nX, "\n"

for i in range(len(i1)):
	for j in range(len(i1[0])):
		i1[i][j] = i1[i][j]-nX[i][j]

print "Difference Matrix :", "\n", i1
