import json
import csv
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, RobustScaler

# Config Variables
NUM_COLS = 8 # The number of columns in the original data set. Used to discard the original columns when extracting features.
NUM_RECS = 100 # The number of recommendations to generate.

# HTML Template Variables
HTML_HEADER = """
<html>

<title>Assignment Example Output</title>
<link rel="stylesheet" href="index.css">


<body>

  <h1>Dresses picked for: {0}</h1>
  <div class="dressesContainer">
"""
DRESS_DIV_TEMPLATE = """
    <div class="dressGridItem">
      <div class="dressGridImgContainer">
        <img src="https://qny.queenly.com/wardrobe_grid/{0}.jpg" width="200"/>
      </div>
      <div>${1}</div>
      <div>Size {2}</div>
      <div>Color: {3}</div>
    </div>
"""
CLOSING_TAGS = """
    </div>
</body>
</html>
"""


def add_features(dress_data):

    def _to_np_array(column):
        """
        Helper function to convert a single column data frame to a numpy array.
        """
        return np.array(column).reshape(-1, 1)

    def encode_variables(dress_data):
        """
        One-Hot encode all categorical variables in the dataset. Returns the concatenated
        result of the data set with the One-Hot encodings.
        """
        encoded_color = pd.get_dummies(dress_data.wardrobe_color)
        encoded_designer = pd.get_dummies(dress_data.wardrobe_designer)
        encoded_style = pd.get_dummies(dress_data.wardrobe_style)

        return pd.concat((dress_data, encoded_color,
                                      encoded_designer,
                                      encoded_style), axis=1)


    def scale_variables(dress_data):
        # Calls to_np_array to convert wardrobe_price column into a numpy array.
        scaled_prices = RobustScaler().fit_transform(_to_np_array(dress_data.wardrobe_price)) # Scale values to adjust for outliers
        scaled_sizes = MinMaxScaler(feature_range=(-1,1)).fit_transform(_to_np_array(dress_data.wardrobe_size)) * 100

        return pd.concat((dress_data, pd.DataFrame(scaled_sizes, columns=['scaled_size']),
                                    pd.DataFrame(scaled_prices, columns=['scaled_price'])), axis=1)


    dress_data = encode_variables(dress_data)
    dress_data = scale_variables(dress_data)

    return dress_data


def extract_features(data):
    """
    Returns only the features that were extracted, removing the original columns.
    """
    return data.iloc[:, NUM_COLS:].to_numpy() # Makes sure we're only dealing with the encoded or scaled values (and no original data) that we originally concatenated onto


def dress_distance(d1, d2):
    """
    Calculate the cosine similarity between the two dresses. The larger the distance, the more similar the dresses
    """
    return float(np.dot(d1, d2.T)/(np.linalg.norm(d1)*np.linalg.norm(d2))) # Simply uing the cosine distance formula to calculate distance between two dresses


def calc_recommendation_score(dress, user_dress_features):
    """
    Calculates the dress distance between the user's dress and a dress in the larger database with 8,000 dresses
    """
    score = 0.0
    for user_dress in user_dress_features:
        score += dress_distance(dress, user_dress)
    return score


def output_recommendations(user_name, dress_data, text_file_src, html_file_src, max_recommendations=NUM_RECS):
    # Opening the HTML files:
    text_file = open(text_file_src, 'w')
    html_file = open(html_file_src, 'w')

    html_file.write(HTML_HEADER.format(user_name)) # Writing the user's name as a header/title

    for idx, recco in dress_data.sort_values('score', ascending=False).dropna().iterrows(): # Sorting the dresses from most similar to least similar
        if idx == 100: # Makes sure we only get 100 dresses
            break

        # Write the wardrobe id, picture, price, size and color for each dress to the HTML webpage:
        text_file.write(recco.wardrobe_id + '\n')
        html_file.write(DRESS_DIV_TEMPLATE.format(recco.wardrobe_photo_signature,
                                                  recco.wardrobe_price,
                                                  recco.wardrobe_size,
                                                  recco.wardrobe_color))

    html_file.write(CLOSING_TAGS) # Closing the div and the body tags
    text_file.close()
    html_file.close()


def generate_dress_recommendations(user_data_source, dress_data_source):
    # Load data
    dress_data = pd.read_csv(dress_data_source)
    user_data = json.load(open(user_data_source))

    user_name = user_data['name'] # Gets the user's name
    user_saved_dresses = user_data['last_saved_dresses'] # Gets the wardrobe ids of all dresses the user saved

    # Extract features from dress_data
    dress_data = add_features(dress_data) # Call add_features with the original csv file
    dress_features = extract_features(dress_data) # Call extract_features with the original csv file
    # Getting all the dress data where wardrobe id of user's saved dresses matches wardrobe id in csv file:
    user_dress_features = extract_features(dress_data[dress_data.wardrobe_id.isin(user_saved_dresses)])

    # Iterate through dress list, and calculate the recommendation score for each dress
    recommendation_scores = []
    for dress in dress_features:
        # Calculates the recommendation score between each of the user's dresses and all dresses in the database:
        recommendation_scores.append(calc_recommendation_score(dress, user_dress_features))
    dress_data = pd.concat((dress_data, pd.DataFrame(recommendation_scores, columns=['score'])), axis=1) # Concatenating the recommendation scores to the encoded data table

    # We only want recommendations that the user hasn't saved already
    output_recommendations(
        user_name,
        dress_data[~dress_data.wardrobe_id.isin(user_saved_dresses)], # Only picks the dresses that the user didn't already save
        user_data_source.replace('json', 'txt'),
        user_data_source.replace('json', 'html')
    )

generate_dress_recommendations("emily.json", "wardrobe_items_downloaded_data.csv")
generate_dress_recommendations("trisha.json", "wardrobe_items_downloaded_data.csv")
