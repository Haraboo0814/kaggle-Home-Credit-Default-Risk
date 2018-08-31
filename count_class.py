import pandas as pd
import matplotlib.pyplot as plt

train_df = pd.read_csv("C:/Users/Kyohei Harada/Desktop/HomeCreditDefaultRisk/all/application_train.csv")

count_classes = pd.value_counts(train_df['TARGET'], sort = True)
count_classes.plot(kind = 'bar', rot=0, figsize = (10,6))
plt.xticks(range(2), ["0","1"])
plt.savefig("result/target.png")
