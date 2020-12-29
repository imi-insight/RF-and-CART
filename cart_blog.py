# %matplotlib inline
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn import tree
import matplotlib.pyplot as plt

X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2 ,random_state = 42)
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)

predictions = clf.predict(X_test)
print("Accuracy :",float(100*accuracy_score(predictions, y_test)))

plt.figure(figsize=(20,20))
tree.plot_tree(clf, filled=True)

"""#Cost Complexity Pruning"""

path = clf.cost_complexity_pruning_path(X_train,y_train)
ccp_alphas, impurities = path.ccp_alphas, path.impurities

ccp_alphas, impurities

clfs = []
for i in ccp_alphas:
  clf = DecisionTreeClassifier(random_state=42, ccp_alpha = i)
  clf.fit(X_train,y_train)
  clfs.append(clf)
print("Number of nodes in last tree : {} with cc_aplha : {} ".format(clfs[-1].tree_.node_count, ccp_alphas[-1]))

train_scores = [clf.score(X_train, y_train) for clf in clfs]
test_scores = [clf.score(X_test, y_test) for clf in clfs]

train_scores.pop(),test_scores.pop()
len(train_scores), len(test_scores)

ccp_alphas
alphax = ccp_alphas[:-1]
print(alphax)
alphax.shape
print(train_scores)
print(test_scores)

fig, ax = plt.subplots(figsize=(10,10))
ax.set_xlabel('Alpha')
ax.set_ylabel('Accuracy')
ax.set_title('Accuracy over time for training & testing')
ax.plot(alphax, train_scores, marker = 'o', label='train', drawstyle='steps-post')
ax.plot(alphax, test_scores, marker = 'o', label='test', drawstyle='steps-post')
ax.legend()
plt.show()

train_scores = [clf.score(X_train, y_train) for clf in clfs]
test_scores = [clf.score(X_test, y_test) for clf in clfs]

fig, ax = plt.subplots(figsize=(20,10))
ax.set_xlabel('ccp_aplha')
ax.set_ylabel('Accuracy')
ax.set_title('Accuracy over time for training & testing')
ax.plot(ccp_alphas, train_scores, marker = 'o', label='train', drawstyle='steps-post')
ax.plot(ccp_alphas, test_scores, marker = 'o', label='test', drawstyle='steps-post')
ax.legend()
plt.show()

ccp_alphas

clf = DecisionTreeClassifier(random_state=42, ccp_alpha=0.012)
clf.fit(X_train, y_train)

predictions = clf.predict(X_test)
print('Accuracy after pruning : ',float(100*accuracy_score(predictions,y_test)))

plt.figure(figsize=(20,20))
tree.plot_tree(clf, filled=True)

for i in ccp_alphas:
  clf = DecisionTreeClassifier(random_state=42, ccp_alpha =i)
  clf.fit(X_train, y_train)
  predictions = clf.predict(X_test)
  print('Accuracy after pruning : ',float(100*accuracy_score(predictions,y_test)),'for alpha : ', i)

