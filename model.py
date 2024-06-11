from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle

def generate_model (file_path,lag):
    """Function that generates and evaluates a ML model
    from the dataframe with all the data"""

    #Getting transformed dataframe

    from transform import transform

    df = transform(file_path,lag)

    #Importing required libraries

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

    # Save the model and label encoder for future use
    with open(f'model.pkl', 'wb') as f:
        pickle.dump((model, label_encoder), f)
    
    # Predict on the test set
    y_pred = model.predict(X_test)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Accuracy: {accuracy}')


def predict_result(home_results, away_results):
    """"
    Function to predict based on user input
    """

    # Load the model and label encoder for the given season
    with open(f'model.pkl', 'rb') as f:
        model, label_encoder = pickle.load(f)
    
    # Encode the user input
    encoded_home_results = label_encoder.transform(home_results)
    encoded_away_results = label_encoder.transform(away_results)

    # Create the feature vector (using both home and away results)
    features = list(encoded_home_results) + list(encoded_away_results)
    features = [features]  # Make it 2D array

    # Predict the result
    prediction = model.predict(features)
    predicted_label = label_encoder.inverse_transform(prediction)

    return predicted_label[0]

msg = input("Please, enter the last 5 results of the home team (e.g., 'W L D W L'): ")
home_results = msg.split()

msg_2 = input("Now, enter the 5 last results of the visiting team (e.g., 'L W W D L'): ")
away_results = msg_2.split()

# Predict the result based on user input
predicted_result = predict_result(home_results, away_results)
print(f'Predicted result for home team: {predicted_result}')