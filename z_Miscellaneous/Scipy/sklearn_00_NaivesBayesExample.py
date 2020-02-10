
import numpy as np

# Dataset
X = np.array([
	[-1, -1],
	[-2, -1],
	[-3, -2],
	[1, 1],
	[2, 1],
	[3, 2]])

# Labels
Y = np.array([1, 1, 1, 2, 2, 2])

from sklearn.naive_bayes import GaussianNB

# Create a classifier
clf = GaussianNB()

# Train
clf.fit(X, Y)

# Predict
print(clf.predict([[-0.8, -1]]))
