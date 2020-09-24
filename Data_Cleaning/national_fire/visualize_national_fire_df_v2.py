import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


df =  pd.read_csv('/Users/lingfengcao/Leyao/ANLY501/Portfolio_Code_for_cleanData/national_fire_dropped_v2.csv')

df['SIZECLASS'].value_counts().plot(kind='bar')
plt.xlabel('Fire Size (A: smallest)')
plt.ylabel('Counts')

plt.show()
