import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("mnist_train.csv")

arr = df.values
labels_arr = arr[:, 0]
pixels_arr = arr[:, 1:]


# splitting the data into training data and testing data
split_size = round(0.80 * len(labels_arr))

training_labels = labels_arr[:split_size]
testing_labels = labels_arr[split_size:]

training_pixels = pixels_arr[:split_size, :]
testing_pixels = pixels_arr[split_size:, :]


# method to return the euclidean distance between two points 
def dist(pt_1, pt_2):
    return np.sqrt(sum((pt_1 - pt_2)**2))


def knn(training_pixels, label_arr, query_channel, k):
    distance_list = []
    for i, channel in enumerate(training_pixels):
        distance_list.append([dist(channel, query_channel), label_arr[i]])
                              
    distance_list.sort()
    distance_list = np.array(distance_list)
    distance_list = distance_list[:k, 1]
    b = np.unique(distance_list,return_counts=True)
    labels_list = b[0]
    freq_list = b[1]
    max_index = freq_list.argmax()
    pred = labels_list[max_index]
    return pred


index = int(input("Enter the index of digit in testing data "))
digit = knn(training_pixels, training_labels, testing_pixels[index], 100)

plt.title(f"predicted outcome- {round(digit)}")
plt.imshow(testing_pixels[index].reshape(28, 28), cmap='gray')
plt.show()