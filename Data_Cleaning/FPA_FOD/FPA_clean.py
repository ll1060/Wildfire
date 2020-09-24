
#### SQL query to select out columns needed and
#### reform the julian dates in DISCOVERY_DATE and CONT_DATE
#### to readable date format

# SELECT  FPA_ID,
# 		SOURCE_REPORTING_UNIT,
# 		SOURCE_REPORTING_UNIT_NAME,
# 		FIRE_CODE,
# 		STAT_CAUSE_CODE,
# 		STAT_CAUSE_DESCR,
# 		FIRE_SIZE,
# 		FIRE_SIZE_CLASS,
# 		LATITUDE,
# 		LONGITUDE
# 		FIRE_YEAR,
# 	    DATETIME(DISCOVERY_DATE) AS DISCOVERY_DATE,
# 	    DATETIME(CONT_DATE) AS CONT_DATE
# FROM   Fires
# WHERE STATE = 'CA' AND FIRE_YEAR >= 2010


#### ALSO check whether STAT_CAUSE_CODE, STAT_CAUSE_DESCR,
#### FIRE_SIZE_CLASS, FIRE_SIZE, LATITUDE, LOGITUDE has null values

# select *
# From Fires
# WHERE FIRE_YEAR >= 2010 AND STATE = 'CA' AND FIRE_SIZE_CLASS IS NULL
#### since the query gives 0 result, these columns do not need to be further cleaned with respect to NULL values
import pandas as pd

df = pd.read_csv('/Users/lingfengcao/Leyao/ANLY501/Portfolio_Code_for_cleanData/FPA_FOD/FPA_FOD_CA.csv')


## drop one more column
df.drop(['FIRE_CODE'],axis=1, inplace=True)

## drop rows that have null value in CONT_DATE
df = df[pd.notnull(df['CONT_DATE'])]

## save df
df.to_csv('/Users/lingfengcao/Leyao/ANLY501/Portfolio_Code_for_cleanData/FPA_FOD/FPA_FOD_CA_clean.csv')
