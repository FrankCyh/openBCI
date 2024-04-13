import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Define the confusion matrix data
conf_matrix = np.array([[80, 46],  # TN, FP
                        [23, 86]]) # FN, TP

# Initialize a plot with a specific size
plt.figure(figsize=(8, 6))

# Create a heatmap from the confusion matrix
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Predicted Non-stimulus', 'Predicted Stimulus'],
            yticklabels=['Actual Non-stimulus', 'Actual Stimulus'])

# Add labels to the x and y axis
plt.xlabel('Predicted labels')
plt.ylabel('Ground Truth labels')

# Add a title to the heatmap
plt.title('P300 Classification Confusion Matrix')

# Display the plot
plt.show()
