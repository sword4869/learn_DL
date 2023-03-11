- [1. pandas](#1-pandas)
  - [1.1. one-shot](#11-one-shot)
  - [1.2. Nan](#12-nan)

# 1. pandas

## 1.1. one-shot

```python
data = {
    'Sex': ['male','female', 'male'],
    'Alive': [True, False, False],
    'Age': [18, 22, 19]
}
df = pd.DataFrame(data)
print(df)

# 只转化默认的string类型
df_default = pd.get_dummies(df)
print(df_default)

# 指定必须转化
df_select = pd.get_dummies(df, columns=['Sex', 'Alive', 'Age'])
print(df_select)

'''
      Sex  Alive  Age
0    male   True   18
1  female  False   22
2    male  False   19

   Alive  Age  Sex_female  Sex_male
0   True   18           0         1
1  False   22           1         0
2  False   19           0         1

   Sex_female  Sex_male  Alive_False  Alive_True  Age_18  Age_19  Age_22
0           0         1            0           1       1       0       0
1           1         0            1           0       0       0       1
2           0         1            1           0       0       1       0
'''
```


## 1.2. Nan

单个元素判断
```python
# 用numpy赋值
x = np.nan
# 但是x == np.nan不行，因为np.nan == np.nan是False...

# 用pandas的
if pd.isna(x):
    pass
if pd.notna(x):
    pass
```
查找
```python
df[pd.isna(df).values]
'''
      Sex  Alive   Age
1  female    NaN  22.0
2    male  False   NaN
'''
```
(1) 填充
```python
df.fillna(-1)
```
        Sex  Alive   Age
    0    male   True  18.0
    1  female     -1  22.0
    2    male  False  -1.0
(2) 直接扔掉含Nan的行
```python
df = df.dropna()

# 如果后续要用index的话，要重新排列index
# 因为扔掉行的`index`是直接扔掉，不是自动连续的。
df.index = [i for i in range(df.index.size)]
```

(3) 用one-shot表示Nan：但是显然含有nan的Age不应该用one-shot表示，这就是局限
```python
data = {
    'Sex': ['male','female', 'male'],
    'Alive': [True, np.nan, False],
    'Age': [18, 22, np.nan]
}
df = pd.DataFrame(data)
print(df)

# Age并不会动
df_default = pd.get_dummies(df, dummy_na=True)
print(df_default)

# dummy_na 只在要转化的列里生效，所以我们指定columns
df_select = pd.get_dummies(df, columns=['Sex', 'Alive', 'Age'], dummy_na=True)
print(df_select)
```
          Sex  Alive   Age
    0    male   True  18.0
    1  female    NaN  22.0
    2    male  False   NaN
    
        Age  Sex_female  Sex_male  Sex_nan  Alive_False  Alive_True  Alive_nan
    0  18.0           0         1        0            0           1          0
    1  22.0           1         0        0            0           0          1
    2   NaN           0         1        0            1           0          0
    
       Sex_female  Sex_male  Sex_nan  Alive_False  Alive_True  Alive_nan  \
    0           0         1        0            0           1          0   
    1           1         0        0            0           0          1   
    2           0         1        0            1           0          0   
       Age_18.0  Age_22.0  Age_nan  
    0         1         0        0  
    1         0         1        0  
    2         0         0        1  
    