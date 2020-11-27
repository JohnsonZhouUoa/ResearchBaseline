from skmultiflow.data.data_stream import DataStream
from skmultiflow.trees import HoeffdingTreeClassifier,HoeffdingAdaptiveTreeClassifier, ExtremelyFastDecisionTreeClassifier
import arff
import pandas
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier


drift_point = 250000

data = arff.load("SEA.arff")
df = pandas.DataFrame(data)

print(df.iloc[:, df.shape[1]-1].value_counts())

df_first = df.iloc[0:drift_point, ]
df_second = df.iloc[drift_point:, ]

print(df_first.iloc[:, df_first.shape[1]-1].value_counts())

print(df_second.iloc[:, df_second.shape[1]-1].value_counts())

x_first = df_first.iloc[:, 0:df_first.shape[1]-1]
y_first = df_first.iloc[:, df_first.shape[1]-1]
x_second = df_second.iloc[:, 0:df_second.shape[1]-1]
y_second = df_second.iloc[:, df_second.shape[1]-1]

df_first_first = df_first.iloc[0:int(df_first.shape[0] / 2), ]
df_first_second= df_first.iloc[int(df_first.shape[0] / 2):, ]
df_second_first = df_second.iloc[0:int(df_second.shape[0] / 2), ]
df_second_second= df_second.iloc[int(df_second.shape[0] / 2):, ]

x_first_first = df_first_first.iloc[:, 0:df_first_first.shape[1]-1]
y_first_first  = df_first_first.iloc[:, df_first_first.shape[1]-1]
x_first_second = df_first_second.iloc[:, 0:df_first_second.shape[1]-1]
y_first_second = df_first_second.iloc[:, df_first_second.shape[1]-1]
x_second_first  = df_second_first.iloc[:, 0:df_second_first.shape[1]-1]
y_second_first  = df_second_first.iloc[:, df_second_first.shape[1]-1]
x_second_second = df_second_second.iloc[:, 0:df_second_second.shape[1]-1]
y_second_second = df_second_second.iloc[:, df_second_second.shape[1]-1]

ht_first = DecisionTreeClassifier()
ht_first.fit(x_first, y_first)

ht_second = DecisionTreeClassifier()
ht_second.fit(x_second, y_second)

pred_first = ht_first.predict(x_second)
print("First tree maximum depth: " + str(ht_first.tree_.max_depth))
pred_second = ht_second.predict(x_first)
print("Second tree maximum depth: " + str(ht_second.tree_.max_depth))

pred_first_first = ht_first.predict(x_first_first)
pred_first_second = ht_second.predict(x_first_second)
pred_second_first = ht_first.predict(x_second_first)
pred_second_second = ht_second.predict(x_second_second)


print("Accuracy from ht_1 on first first half: ", metrics.accuracy_score(y_first_first, pred_first_first))
print("Accuracy from ht_2 on first second half: ", metrics.accuracy_score(y_first_second, pred_first_second))
print("Accuracy from ht_1 on second first half: ", metrics.accuracy_score(y_second_first, pred_second_first))
print("Accuracy from ht_2 on second second half: ", metrics.accuracy_score(y_second_second, pred_second_second))
