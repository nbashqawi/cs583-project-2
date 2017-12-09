'''
Created on Dec 4, 2017

@author: nbash
'''
import data.utils as data

from sklearn import metrics
import matplotlib.pyplot as plt

train_file_location = r"..\resources\training-Obama-Romney-tweets.xlsx"
test_file_location = r"..\resources\final-testData-no-label-Obama-Romney-tweets.xlsx"
#obama_test_file_location = r"..\resources\sample-testdata-obama.xlsx"
#romney_test_file_location = r"..\resources\sample-testdata-romney.xlsx"

obama_train, romney_train = data.import_and_filter(train_file_location, dropnan = True)
obama_test, romney_test = data.import_test_data(test_file_location)

obo_model = data.select_model("log_reg")
obo_model.fit(obama_train['Annotated Tweet'], obama_train['Class'])
obo_predicted = obo_model.predict(obama_test['tweet'])

rom_model = data.select_model("log_reg")
rom_model.fit(romney_train['Annotated Tweet'], romney_train['Class'])
rom_predicted = rom_model.predict(romney_test['tweet'])

data.export_classifier_results(obama_test, obo_predicted, r'../resources/Nicholas_Bashqawi_Obama.txt')
data.export_classifier_results(romney_test, rom_predicted, r'../resources/Nicholas_Bashqawi_Romney.txt')