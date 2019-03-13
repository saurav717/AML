import numpy as np 
import pandas as pd
import json
 

data_training = pd.read_json('train.json')

# print "data" , data_training

cuisines =  data_training['cuisine'].unique()

data_training['all_ingredients'] = data_training['ingredients'].map(",".join)

ingredients = data_training.values
print ingredients


# ingredients = data_training['all_ingredients'].str.contains('garlic cloves')
# print ingredients

