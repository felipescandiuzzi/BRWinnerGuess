def read_treat_csv(file_path):
    '''Function that gets the path of the csv file and returns
    the dataframe all cleaned and treated, at the same time it 
    prints the main exploratory analysis'''

    #Importing required libraries

    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns

    #Reading dataframe

    df = pd.read_csv(file_path,encoding='utf-8')

    #Getting first samples

    print(f"First Samples:\n {df.head(5)} \n")

    #Getting df info

    print(f"df Info:\n {df.info()} \n")

    #Getting df description

    print(f"df Description:\n {df.describe()} \n")

    #Getting null values (if any)

    print(f"Checking for null values: \n {df.isnull().sum().max()} occurence(s) \n")

    #Removing null values

    df = df.dropna()

    #Checking null values after dropna

    print(f"After treatment, we now have {df.isnull().sum().max()} occurrence(s) \n")

    #Creating winner column

    df["hresult"]=np.where(df['goalsht']>df['goalsvt'],'W',np.where(df['goalsht']<df['goalsvt'],'L','D'))
    print(df.head(5))

    # Histogram to goalsht
    plt.figure(figsize=(8, 6))
    sns.histplot(df['goalsht'].dropna(), kde=True)
    plt.title('Home Team Goals')
    #plt.show()

    # Histogram to goalsvt
    plt.figure(figsize=(8, 6))
    sns.histplot(df['goalsvt'].dropna(), kde=True)
    plt.title('Visiting Team Goals')
    #plt.show()

    return df