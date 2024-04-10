import joblib
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder
from pymongo import MongoClient

def predict_apps(domain_file, model_file, vectorizer_file, data_collection):
    # Load model and vectorizer
    model = joblib.load(model_file)
    vectorizer = joblib.load(vectorizer_file)

    # Load domains from file
    with open(domain_file, 'r') as file:
        domains = file.readlines()
    app_names = [app['App'] for app in data_collection.find() if 'Domain' in app]
    label_encoder = LabelEncoder()
    label_encoder.fit(app_names)

    # Predict apps for each domain
    predictions = []
    for domain in domains:
        domain = domain.strip()  # Remove leading/trailing whitespaces and newlines

        # Transform domain using the loaded vectorizer
        X_transformed = vectorizer.transform([domain])

        # Predict the app for the domain
        prediction_encoded = model.predict(X_transformed)
        app_name = label_encoder.inverse_transform(prediction_encoded)[0]

        predictions.append((domain, app_name))
        #data_collection.insert_one({'domain': domain, 'app_name': app_name})

    return predictions

def add_data(data_collection, domain, app_name):
    # Check if domain already exists in data
    # If domain does not exist, ask whether to add it to data
    query = {'domain': domain}
    result = data_collection.find_one(query)

    if not result:
        add_to_data = input(f"Do you want to add {domain} => {app_name} to data? (1/0): ")
        if add_to_data == '1':
            data_collection.insert_one({'domain': domain, 'app_name': app_name})
    else:
        return


# Connect to MongoDB
client = MongoClient('mongodb://localhost:27017/')
db = client['admin']  # Change 'your_database_name' to your actual database name
data_collection = db['Domain_Classify']  # Change 'your_collection_name' to your actual collection name

# File paths
domain_file = 'New_Domain.txt'
model_file = 'model.sav'
vectorizer_file = 'vectorizer.sav'

# Perform prediction
predictions = predict_apps(domain_file, model_file, vectorizer_file, data_collection)

# Print predictions
for domain, app_name in predictions:
    print(f"Domain: {domain} => Predicted App: {app_name}")
    add_data(data_collection, domain, app_name)  # Add data to db if it doesn't exist
