def transform(file_path):
    """Function that gets dataframe already cleaned
    and creates new arguments to the ML model, accordingly"""

    #Reading and treating basic csv file

    from eda import read_treat_csv

    df = read_treat_csv(file_path)

    print(df.head(5))

transform('data.csv')