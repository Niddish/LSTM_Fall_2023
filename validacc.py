from tslearn.datasets import UCR_UEA_datasets

# Initialize the dataset repository
uea_ucr = UCR_UEA_datasets()

# Get the baseline accuracy for the ElectricDevices dataset
# Assuming the function accepts a single string when querying one dataset
dict_acc = uea_ucr.baseline_accuracy(list_datasets=["ElectricDevices"])

# Access the baseline accuracy for the ElectricDevices dataset
electric_devices_acc = dict_acc.get("ElectricDevices")

# Print out the baseline accuracy
print("Baseline accuracies for ElectricDevices:")
for method, accuracy in electric_devices_acc.items():
    print(f"{method}: {accuracy:.4f}")

x_train, y_train, x_test, y_test = uea_ucr.load_dataset("ElectricDevices")

print(f"Number of training samples: {x_train.shape[0]}")
print(f"Number of testing samples: {x_test.shape[0]}")
print(f"Length of time series: {x_train.shape[1]}")
print(f"Number of features per time step: {x_train.shape[2] if x_train.ndim > 2 else 1}")
print(f"Number of classes: {len(set(y_train))}")
