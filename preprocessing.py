import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv('Life Expectancy Data.csv')


## Category "Status" to continuous ##
for index, row in df.iterrows():
    if row['Status'] == "Developing":
        df.at[index, 'Status'] = 0
    else:
        df.at[index, 'Status'] = 1



## Min-Max normalisation ##

df_normalised = df.copy()

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
df_normalised[['Status', 'Life Expectancy', 'Adult Mortality',
       'Infant Deaths', 'Alcohol', 'Percentage Expenditure', 'Hepatitis B',
       'Measles', 'BMI', 'Under-five Deaths', 'Polio', 'Total Expenditure',
       'Diphtheria', 'HIV/AIDS', 'GDP', 'Population', 'Thinness 1-19 Years',
       'Thinness 5-9 Years', 'Income Composition of Resources', 'Schooling']] = scaler.fit_transform(df_normalised[['Status', 'Life Expectancy', 'Adult Mortality',
       'Infant Deaths', 'Alcohol', 'Percentage Expenditure', 'Hepatitis B',
       'Measles', 'BMI', 'Under-five Deaths', 'Polio', 'Total Expenditure',
       'Diphtheria', 'HIV/AIDS', 'GDP', 'Population', 'Thinness 1-19 Years',
       'Thinness 5-9 Years', 'Income Composition of Resources', 'Schooling']])

# df_normalised = df_normalised.dropna() ## if NaN are causing too much trouble
df_normalised



## Date data split ##

D_train = df_normalised[df_normalised['Year'] <= 2011]
D_train_X = D_train.drop(columns = ['Life Expectancy']).copy()
D_train_y = D_train['Life Expectancy']

D_val = df_normalised[(df_normalised['Year'] > 2011) & (df_normalised['Year'] <= 2013)]
D_val_X = D_val.drop(columns = ['Life Expectancy']).copy()
D_val_y = D_val['Life Expectancy']

D_test = df_normalised[df_normalised['Year'] > 2013]
D_test_X = D_test.drop(columns = ['Life Expectancy']).copy()
D_test_y = D_test['Life Expectancy']


## Output ##
D_train_X.to_csv('D_train_X.csv', sep=',', encoding='utf-8', index=False)
D_train_y.to_csv('D_train_y.csv', sep=',', encoding='utf-8', index=False)

D_val_X.to_csv('D_val_X.csv', sep=',', encoding='utf-8', index=False)
D_val_y.to_csv('D_val_y.csv', sep=',', encoding='utf-8', index=False)

D_test_X.to_csv('D_test_X.csv', sep=',', encoding='utf-8', index=False)
D_test_y.to_csv('D_test_y.csv', sep=',', encoding='utf-8', index=False)



## Random data split ##

df_normalised_X = df_normalised.drop(columns = ['Life Expectancy']).copy()
df_normalised_y = df_normalised['Life Expectancy']

train_X, test_X, train_y, test_y = train_test_split(df_normalised_X, df_normalised_y, test_size=0.2, random_state=1)

test_X, val_X, test_y, val_y = train_test_split(test_X, test_y, test_size=0.5, random_state=1)

## Output ##
train_X.to_csv('train_X.csv', sep=',', encoding='utf-8', index=False)
train_y.to_csv('train_y.csv', sep=',', encoding='utf-8', index=False)

val_X.to_csv('val_X.csv', sep=',', encoding='utf-8', index=False)
val_y.to_csv('val_y.csv', sep=',', encoding='utf-8', index=False)

test_X.to_csv('test_X.csv', sep=',', encoding='utf-8', index=False)
test_y.to_csv('test_y.csv', sep=',', encoding='utf-8', index=False)