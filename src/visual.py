import matplotlib.pyplot as plt


def plot_epoch_loss(train, val):
    plt.plot(train, label="Train Loss")
    plt.plot(val, label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Loss Convergence")
    plt.show()