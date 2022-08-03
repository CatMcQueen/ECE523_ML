import pandas as pd
import numpy as np
from sklearn.ensemble import AdaBoostClassifier


class Adaboost:
	def __init__(self):
		self.features = []
		self.classifiers = []
		self.max_iterations = 10
		self.target_error = .1
		
		self.data = None
		self.weights = None
			
		

	def fit(self, data, labels):
		if not isinstance(data, np.ndarray):
			data = np.array(data)
		if not isinstance(labels, np.ndarray):
			labels = np.array(labels)
		
		
		

	def predict(self, test_data):
		






if __name__ == "__main__":

	(X,y) = importData
	clf = AdaBoostClassifier(n_estimators=100, random_state=0)
	clf.fit(X,y)
	clf.predict(testX)
	clf.score(X,y)
