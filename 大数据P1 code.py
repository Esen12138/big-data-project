
import pandas as pd 
import time
from sklearn.metrics import accuracy_score 
from sklearn.tree import DecisionTreeClassifier  
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

col_names = ['winner','firstBlood','firstTower','firstInhibitor','firstBaron','firstDragon','firstRiftHerald','t1_towerKills','t1_inhibitorKills','t1_baronKills','t1_dragonKills','t1_riftHeraldKills','t2_towerKills','t2_inhibitorKills','t2_baronKills','t2_dragonKills','t2_riftHeraldKills'] 

pima1 = pd.read_csv("C:\\Users\\Administrator\\Desktop\\test_set.csv", header=None, names=col_names)  
pima1 = pima1.iloc[1:]
pima1.head() 

pima2 = pd.read_csv("C:\\Users\\Administrator\\Desktop\\new_data.csv", header=None, names=col_names) 
pima2 = pima2.iloc[1:]  
pima2.head() 


feature_cols = ['firstBlood','firstTower','firstInhibitor','firstBaron','firstDragon','firstRiftHerald','t1_towerKills','t1_inhibitorKills','t1_baronKills','t1_dragonKills','t1_riftHeraldKills','t2_towerKills','t2_inhibitorKills','t2_baronKills','t2_dragonKills','t2_riftHeraldKills'] 
X_test = pima1[feature_cols] 
y_test = pima1.winner 

X_train = pima2[feature_cols] 
y_train = pima2.winner 

start=time.time()

clf_DT = DecisionTreeClassifier() 
 
clf_DT = clf_DT.fit(X_train,y_train) 
end=time.time()
 
y_pred_DT = clf_DT.predict(X_test)
 
print("Accuracy_DT:",accuracy_score(y_test, y_pred_DT),"running time: %s seconds"%(end-start)) 

start2=time.time()
clf_knn = KNeighborsClassifier() 
 
clf_knn = clf_knn.fit(X_train,y_train) 
 
end2=time.time() 
y_pred_knn = clf_knn.predict(X_test)


print("Accuracy_KNN:",accuracy_score(y_test, y_pred_knn),"running time: %s seconds"%(end2-start2)) 

start3=time.time()
clf_MLP = MLPClassifier(max_iter=20000) 
 
clf_MLP = clf_MLP.fit(X_train,y_train) 
 
end3=time.time()
y_pred_MLP = clf_MLP.predict(X_test)


print("Accuracy_MLP:",accuracy_score(y_test, y_pred_MLP),"running time: %s seconds"%(end3-start3)) 

start4=time.time()
clf_svm = SVC() 
 
clf_svm = clf_svm.fit(X_train,y_train) 
end4=time.time()

y_pred_svm = clf_svm.predict(X_test)


print("Accuracy_svm:",accuracy_score(y_test, y_pred_svm),"running time: %s seconds"%(end4-start4)) 