import numpy as np

def initW(images, labels):
    num_classes = labels.shape[1]
    W = np.random.uniform(-1, 1, (num_classes, images.shape[1] + 1))

    for _ in range(len(images)):
        k = np.random.randint(0, len(labels))
        Yk = labels[k] - 0.5  # Étiquettes ajustées à l'intervalle [-0.5, 0.5]
        Xk = np.concatenate(([1], images[k]))  # Pixels de l'image
        signal = np.dot(W, Xk)
        gXk = np.where(signal >= 0, 1.0, -1.0)
        W = W + 0.01 * np.outer(Yk - gXk, Xk)

    return W

def linearModel(W, images):
    predicted_scores = []

    for image in images:
        scores = np.dot(W, np.concatenate(([1], image)))
        exp_scores = np.exp(scores)
        probabilities = exp_scores / (np.sum(exp_scores) + np.finfo(float).eps) * 100
        predicted_scores.append(probabilities.tolist())

    return predicted_scores