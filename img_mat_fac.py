import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import time as ti

from sklearn.datasets import load_digits

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


def mat_fac(X,K,S=10000):
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
				ne = ne + (X[i][j]*np.log((X[i][j]+0.0)/(nX[i][j]+0.0))-X[i][j]+nX[i][j])
		print ne
		if np.abs(ne-e)>0:
			e=ne
			ne=0
		else:
			print e
			print s
			break
		s+=1
	# print X, "\n"
	# print W, "\n"
	# print H, "\n"
	# print nX #np.dot(W,H)

	# diff=np.zeros((len(X),len(X[0])))
	# for i in range(len(X)):
	# 	for j in range(len(X[i])):
	# 		diff[i][j]=X[i][j]-nX[i][j]
	# print diff
	print "step =", s
	print "time taken =", ti.time()-st
	return nX

# img = mpimg.imread('lena1.jpg')

# img_ar = np.dot(img,1)
# img_ar[100][10] = 0
# plt.imshow(img_ar, cmap = plt.get_cmap('gray'))
# plt.show()

#hand-written digits
digits = load_digits()

fig = plt.figure()
ax1 = fig.add_subplot(1,2,1)
#img = mpimg.imread('lena1.jpg')
img = digits.images[4]
img_ar = np.array(img,dtype='f') #taking a new with floating characters
print "img array", img_ar, "\n"
ax1.imshow(img_ar, cmap = plt.get_cmap('gray'))

ax2 = fig.add_subplot(1,2,2)
for i in range(len(img_ar)):
	for j in range(len(img_ar[0])):
		img_ar[i][j] = img_ar[i][j]*0.001+pow(.1,20)
print "new img array", img_ar
new_img_ar = mat_fac(img_ar,4)
for i in range(len(img_ar)):
	for j in range(len(img_ar[0])):
		new_img_ar[i][j] = new_img_ar[i][j]*1000-pow(.1,15)
print "regenerated array", new_img_ar		
ax2.imshow(new_img_ar, cmap = plt.get_cmap('gray'))

fig.savefig(str(ti.time())+".jpg")
plt.show()
