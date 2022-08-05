from pandas import read_csv
from pandas.plotting import scatter_matrix
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

# Load CSV file as a Pandas DataFrame and set column names.
path = '/Users/stefanvuleta/Projects/iris_machine_learning/iris.csv'
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = read_csv(path, names = names)

# Find number of DataFrame rows and columns.
print(dataset.shape)

# Access first 20 rows in DataFrame.
print(dataset.head(20))

# Calculate some statistical data like percentile, mean and std of the numerical values of the DataFrame.
print(dataset.describe())

# Get int representing the number of elements in class column by group.
print(dataset.groupby('class').size())

# Create box and whisker plots.
dataset.plot(kind = 'box', subplots = True, layout = (2,2), sharex = False, sharey = False)
pyplot.show()

# Create four histograms.
dataset.hist()
pyplot.show()

# Create a scatter plot matrix to visualize bivariate relationships.
scatter_matrix(dataset)
pyplot.show()

# Split-out validation dataset. Dataset split into 2 parts. 80% will be used to train, evaluate and select among models.
# 20% will be held back as a validation dataset.
array = dataset.values
X = array[:,0:4]
y = array[:,4]
X_train, X_validation, Y_train, Y_validation = train_test_split(X, y, test_size = 0.20, random_state=1)

# Spot check algorithms.
models = []
models.append(('LR', LogisticRegression(solver = 'liblinear', multi_class = 'ovr'))) # Logistic Regression. Simple linear algorithm.
models.append(('LDA', LinearDiscriminantAnalysis())) # Linear Discriminant Analysis. Simple linear algorithm.
models.append(('CART', DecisionTreeClassifier())) # Classification and Regression Trees. Nonlinear algorithm.
models.append(('NB', GaussianNB())) # Gaussian Naive Bayes. Nonlinear algorithm.
models.append(('SVM', SVC(gamma = 'auto'))) # Support Vector Machines. Nonlinear algorithm.

# Evaluate each model. Results may vary given the stochastic nature of the algorithm or evaluation procedure or differences in numerical precision.
results = []
names = []
for name, model in models:
	kfold = StratifiedKFold(n_splits = 10, random_state = 1, shuffle = True)
	cv_results = cross_val_score(model, X_train, Y_train, cv = kfold, scoring = 'accuracy')
	results.append(cv_results)
	names.append(name)
	print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))

# Compare algorithms by creating a box and whisker plot for each distribution.
pyplot.boxplot(results, labels = names)
pyplot.title('Algorithm Comparison')
pyplot.show()

# Fit the model on the entire training dataset and make predictions on the validation dataset.
model = SVC(gamma = 'auto')
model.fit(X_train, Y_train)
predictions = model.predict(X_validation)

# Evaluate predictions by comparing them to the expected results in the validation set. Then calculate classification accuracy,
# as well as a confusion matrix and a classification report.
print(accuracy_score(Y_validation, predictions)) # Accuracy is .966 or about 96% on the dataset.
print(confusion_matrix(Y_validation, predictions)) # Provides an indication of the errors made.
print(classification_report(Y_validation, predictions)) # Provides breakdown of each class by precision, recall, f1-score, and support.