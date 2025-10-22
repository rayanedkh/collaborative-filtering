import numpy as np


predictions = np.load("output.npy")  
test_set = np.load("ratings_test.npy")  

mask_test = ~np.isnan(test_set)

diff = predictions - test_set
rmse = np.sqrt(np.sum((diff[mask_test])**2) / np.sum(mask_test))

accuracy = np.sum(predictions[mask_test] == test_set[mask_test]) / np.sum(mask_test)

print(f"RMSE sur le test set : {rmse:.4f}")
print(f"Accuracy sur le test set : {accuracy:.4f}")
