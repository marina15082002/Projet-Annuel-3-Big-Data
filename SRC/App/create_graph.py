import matplotlib.pyplot as plt

def plot_learning_curve(train_accuracy, test_accuracy):
    iterations = range(len(train_accuracy))

    plt.plot(iterations, train_accuracy, label='Training Accuracy')
    plt.plot(iterations, test_accuracy, label='Testing Accuracy')
    plt.xlabel('Iteration')
    plt.ylabel('Accuracy')
    plt.title('Learning Curve')
    plt.legend()
    plt.show()

def plot_loss_curve(train_loss, test_loss, epochs):
    plt.plot(epochs, train_loss, label='Training Loss')
    plt.plot(epochs, test_loss, label='Testing Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss Curve')
    plt.legend()
    plt.show()