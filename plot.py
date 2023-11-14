# Print the correlation matrix with heat map
import seaborn as sns
import matplotlib.pyplot as plt

class plot_heatmap:
    def __init__(self, df):
        self.df = df
    def plot_heatmap(self):
        corr = self.df.corr(method='pearson')
        plt.figure(figsize=(10, 10))
        sns.heatmap(corr, annot=True, cmap='coolwarm')
        plt.show()

'''
# Pearson Correlation Analysis
corr = df.corr(method='pearson')
plt.figure(figsize=(10, 10))
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.show()
'''

'''
# Visualize the X_train and X_test data
def plot_predictions(train_data=X_train, 
                     train_labels=y_train, 
                     test_data=X_test, 
                     test_labels=y_test, 
                     predictions=None):
  """
  Plots training data, test data and compares predictions.
  """
  plt.figure(figsize=(10, 7))

  # Plot training data in blue
  plt.scatter(train_data, train_labels, c="b", s=4, label="Training data")
  
  # Plot test data in green
  plt.scatter(test_data, test_labels, c="g", s=4, label="Testing data")

  if predictions is not None:
    # Plot the predictions in red (predictions were made on the test data)
    plt.scatter(test_data, predictions, c="r", s=4, label="Predictions")

  # Show the legend
  plt.legend(prop={"size": 14})
  plt.show()

plot_predictions()
'''
