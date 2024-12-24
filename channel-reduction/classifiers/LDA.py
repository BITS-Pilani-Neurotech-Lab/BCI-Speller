from Classifier import Classifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

class LDA(Classifier):
	def __init__(self):
		self.lda = None

	def train(self, train_data, train_labels):
		train_data_copy = train_data.reshape(train_data.shape[0], -1)

		self.lda = LinearDiscriminantAnalysis()
		self.lda.fit(train_data_copy, train_labels)
	
	def predict(self, data):
		data_copy = data.reshape(data.shape[0], -1)
		return self.lda.predict_proba(data_copy)
