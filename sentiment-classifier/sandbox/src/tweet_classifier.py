'''
Created on Dec 4, 2017

@author: nbash
'''
import data.utils as data

from sklearn import metrics
import matplotlib.pyplot as plt

train_file_location = r"..\resources\training-Obama-Romney-tweets.xlsx"
obama_test_file_location = r"..\resources\sample-testdata-obama.xlsx"
romney_test_file_location = r"..\resources\sample-testdata-romney.xlsx"

obama_train, romney_train = data.import_and_filter(train_file_location, dropnan = True)
obama_test, romney_test = data.import_test_data_separate(obama_test_file_location, romney_test_file_location)

obo_model = data.select_model("log_reg")
obo_model.fit(obama_train['Annotated Tweet'], obama_train['Class'])
obo_predicted = obo_model.predict(obama_test['tweet'])

rom_model = data.select_model("log_reg")
rom_model.fit(romney_train['Annotated Tweet'], romney_train['Class'])
rom_predicted = rom_model.predict(romney_test['tweet'])

data.export_classifier_results(obama_test, obo_predicted, r'../resources/obama_out.txt')
data.export_classifier_results(romney_test, rom_predicted, r'../resources/romney_out.txt')