import numpy as np


def softmax(x):
    x = x - np.max(x, axis=-1, keepdims=True)
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


def main():
    Q = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
    ], dtype=np.float32)

    K = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
    ], dtype=np.float32)

    V = np.array([
        [10, 0, 0, 0],
        [0, 20, 0, 0],
    ], dtype=np.float32)

    D = Q.shape[1]

    scores = Q @ K.T / np.sqrt(D)
    weights = softmax(scores)
    output = weights @ V

    print("Scores:")
    print(scores)

    print("\nSoftmax weights:")
    print(weights)

    print("\nAttention output:")
    print(output)


if __name__ == "__main__":
    main()