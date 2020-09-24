import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


df =  pd.read_csv('/Users/lingfengcao/Leyao/ANLY501/Portfolio_Code_for_cleanData/FPA_FOD/FPA_FOD_CA_clean.csv')

df['STAT_CAUSE_DESCR'].value_counts().plot(kind='bar')
plt.xlabel('Cause of Fire')
plt.ylabel('Counts')

plt.show()
