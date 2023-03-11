```python
import pandas as pd
import numpy as np
```


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
    