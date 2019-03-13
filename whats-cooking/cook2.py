import numpy as np
import pandas as pd 

def ingredients_data(jsonfile):

	with open(jsonfile) as training_data: 
		training_data = js.load