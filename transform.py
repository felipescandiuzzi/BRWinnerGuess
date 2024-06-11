def transform(file_path,lag):
    """Function that gets dataframe already cleaned
    and creates new arguments to the ML model, accordingly"""

    #Reading and treating basic csv file

    from eda import read_treat_csv
    import pandas as pd
    from collections import deque

    df = read_treat_csv(file_path)

    #Creating date column to sort by it

    df['date'] = pd.to_datetime(df[['year', 'month', 'day']])

    #Sorting dataframe accordingly, for ascendent date

    df = df.sort_values(by='date').reset_index(drop=True)

    #Starting getting last 5 results, creating a dict to store it

    team_results = {}

    for team in pd.concat([df['hometeam'],df['visitingteam']]).unique():
        team_results[team]=deque(maxlen=lag)

    #Tracking last 5 results
    
    last_lag_h_results = []
    last_lag_v_results = []

    for index, row in df.iterrows():
        hteam=row['hometeam']
        vteam=row['visitingteam']

        last_lag_h_results.append(list(team_results[hteam]))
        last_lag_v_results.append(list(team_results[vteam]))

        team_results[hteam].append(row['hresult'])
        team_results[vteam].append('W' if row['hresult']=='L' else 'L' if row ['hresult']=='W' else 'D')

    df['llagh']=last_lag_h_results
    df['llagv']=last_lag_v_results

    for i in range(lag):
        df[f'hresult_{i+1}'] = df['llagh'].apply(lambda x: x[i] if len(x) > i else None)
        df[f'vresult_{i+1}'] = df['llagv'].apply(lambda x: x[i] if len(x) > i else None)

    # Drop intermediate columns
    df = df.drop(columns=['llagh', 'llagv'])

    print(df.head(5))

transform('data.csv',lag=5)