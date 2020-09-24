import pandas as pd

## drop columns with irrevalent information
# df = pd.read_csv('/Users/lingfengcao/Leyao/ANLY501/Portfolio_Code_for_cleanData/National_USFS_Final_Fire_Perimeter__Feature_Layer_.csv')
# df.drop(['GLOBALID','FIREOCCURID','CN','REVDATE','COMPLEXNAME','UNIQFIREID','DISCOVERYDATETIME','COMMENTS','DATASOURCE','PERIMETERDATETIME','OWNERAGENCY', 'UNITIDOWNER',
# 'PROTECTIONAGENCY', 'UNITIDPROTECT','FIRESTATQC','DBSOURCEID','DBSOURCEDATE'],axis=1, inplace=True)
df =  pd.read_csv('/Users/lingfengcao/Leyao/ANLY501/Portfolio_Code_for_cleanData/national_fire_dropped.csv')

## only investigate the fires that occurred in 2019 and 2020, and remove fires with wrong year(there were a few that had FIREYEAR of '9999')
df = df[df.FIREYEAR >=2015]
df= df[df.FIREYEAR<= 2020]


## drop more redundant columns
df.drop(['SOFIRENUM','LOCALFIRENUM','REPORTINGUNIT','FEATURECAT','GISACRES','SHAPEAREA'],axis=1, inplace=True)

## check rows that contain NA value and delete these rows
df = df.dropna()

## reset index after dropping rows
df.reset_index(drop=True)

## the SHAPELEN col has too many digits
## limit the digits to up to 2 in the whole df
df = df.round(decimals=2)

## the STATCAUSE col has various formats of the causes
## Adjust the format so it is in pure string
#### 1 - Lightning
#### 5 - debris burning
#### 7 - Arson
#### 9 - miscellaneous

## Unmathced number and causes:
## 6, 8, 2, 3, 4
## Equipment Use, Railroad, campfire, smoking, Incendiary

df.loc[df.STATCAUSE == '1', 'STATCAUSE'] = "Lightning"
df.loc[df.STATCAUSE == '1 -  Lightning', 'STATCAUSE'] = "Lightning"
df.loc[df.STATCAUSE == '1 - Lightning', 'STATCAUSE'] = "Lightning"

df.loc[df.STATCAUSE == '5', 'STATCAUSE'] = "Debris Burning"
df.loc[df.STATCAUSE == '5 -  Debris Burning', 'STATCAUSE'] = "Debris Burning"

df.loc[df.STATCAUSE == '7', 'STATCAUSE'] = "Miscellaneous"
df.loc[df.STATCAUSE == '7 -  Arson', 'STATCAUSE'] = "Arson"

df.loc[df.STATCAUSE == '9', 'STATCAUSE'] = "Miscellaneous"
df.loc[df.STATCAUSE == '9 -  Miscellaneous', 'STATCAUSE'] = "Miscellaneous"

## because we do not know the map between other numbers and the causes
## we will categorize all other causes as 'Other'
df.loc[df.STATCAUSE == '2', 'STATCAUSE'] = "Other"
df.loc[df.STATCAUSE == '3', 'STATCAUSE'] = "Other"
df.loc[df.STATCAUSE == '4', 'STATCAUSE'] = "Other"
df.loc[df.STATCAUSE == '6', 'STATCAUSE'] = "Other"
df.loc[df.STATCAUSE == '8', 'STATCAUSE'] = "Other"
df.loc[df.STATCAUSE == 'Smoking', 'STATCAUSE'] = "Other"
df.loc[df.STATCAUSE == 'Campfire', 'STATCAUSE'] = "Other"
df.loc[df.STATCAUSE == 'Equipment Use', 'STATCAUSE'] = "Other"

## drop rows that has 0 as value
df = df[df.STATCAUSE != 0]
df = df[df.TOTALACRES != 0]

df.to_csv('/Users/lingfengcao/Leyao/ANLY501/Portfolio_Code_for_cleanData/national_fire_dropped_v2.csv')
