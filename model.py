def generate_model (file_path,lag):
    """Function that generates and evaluates a ML model
    from the dataframe with all the data"""

    #Getting transformed dataframe

    from transform import transform

    df = transform(file_path,lag)

    #Importing required libraries

    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import classification_report, accuracy_score

    #Encoding categorical values for desired columns

    columns_to_encode=['hresult','hresult_1',  'hresult_2', 
                       'hresult_3','hresult_4','hresult_5', 
                       'vresult_1','vresult_2',  'vresult_3', 
                       'vresult_4', 'vresult_5']
    encoded_df = df[columns_to_encode].copy()

    label_encoder=LabelEncoder()

    for column in encoded_df.columns:
        encoded_df[column] = label_encoder.fit_transform(encoded_df[column])
    
    # Split data into features and target

    X = encoded_df[columns_to_encode[1:]] 
    y = encoded_df['hresult']

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the model
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    # Predict on the test set
    y_pred = model.predict(X_test)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Accuracy: {accuracy}')

    print(X_train)
    print(y_train)
    #print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

df = generate_model('data.csv',5)
print(df)