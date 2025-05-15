import csv
import math
import random



dataset = []
with open("Iris.csv", "r") as file: 
    reader = csv.reader(file)
    next(reader)  # Skip header if present
    for row in reader:
        dataset.append([float(x) for x in row[:-1]] + [row[-1]])  # Convert features to float, keep class label

# Shuffle and split dataset (80% training, 20% testing)
random.shuffle(dataset)
split_index = int(len(dataset) * 0.8)
train_set = dataset[:split_index]
test_set = dataset[split_index:]

# k-NN Classification
k = 11 # Number of neighbors
correct = 0  # Count of correct predictions
total = len(test_set)  # Total test instances

for test_instance in test_set:
    # Calculate distance from test_instance to all training instances
    distances = []
    for train_instance in train_set:
        distance = math.sqrt(sum((test_instance[i] - train_instance[i]) ** 2 for i in range(len(test_instance) - 1)))
        distances.append((train_instance, distance))

    # Sort distances and select k nearest neighbors
    distances.sort(key=lambda x: x[1])
    neighbors = [distances[i][0][-1] for i in range(k)]  # Extract class labels of k nearest neighbors

    # Predict the most common class
    prediction = max(set(neighbors), key=neighbors.count)

    # Check correctness
    if prediction == test_instance[-1]:
        correct += 1  # Increase count if prediction is correct

    # Print result
    print(f"Predicted: {prediction}, Actual: {test_instance[-1]} {'(Correct)' if prediction == test_instance[-1] else '(Wrong)'}")

# Calculate and print accuracy
accuracy = (correct / total) * 100
print(f"\nOverall Accuracy: {accuracy:.2f}%")
