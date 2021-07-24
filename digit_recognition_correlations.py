import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("mnist_train.csv")

arr = df.values
labels_arr = arr[:, 0]
pixels_arr = arr[:, 1:]

split_size = round(0.80 * len(labels_arr))

training_labels = labels_arr[:split_size]
testing_labels = labels_arr[split_size:]

training_pixels = pixels_arr[:split_size, :]
testing_pixels = pixels_arr[split_size:, :]

index = int(input("Enter the index of digit in testing data "))
max_corr = -1
for i,img in enumerate(training_pixels):
    corr = np.corrcoef(testing_pixels[index], img)[1][0]
    if corr > max_corr:
        max_corr = corr
        digit = training_labels[i]


plt.title(f"predicted outcome- {round(digit)}")
plt.imshow(testing_pixels[index].reshape(28, 28), cmap='gray')
plt.show()