'''
Created on Dec 4, 2017

@author: nbash
'''
import data.utils as data

test_file_location = r"..\resources\sample-testdata.xlsx"

obama_test, romney_test = data.import_test_data(test_file_location)
data.export_classifier_results(obama_test, ['1','0','-1','-1'], r'../resources/out.txt')