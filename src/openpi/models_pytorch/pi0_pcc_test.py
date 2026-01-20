import numpy as np

def pcc(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float64).ravel()
    b = b.astype(np.float64).ravel()

    a_mean = a.mean()
    b_mean = b.mean()

    numerator = np.sum((a - a_mean) * (b - b_mean))
    denominator = np.sqrt(
        np.sum((a - a_mean) ** 2) * np.sum((b - b_mean) ** 2)
    )

    if denominator == 0:
        return 0.0

    return numerator / denominator


def main():
    torch_actions = np.load("pytorch_actions.npy")
    jax_actions = np.load("pytorch_actions_ttnn.npy")
    print(torch_actions.shape, jax_actions.shape)
    print(torch_actions.dtype, jax_actions.dtype)

    score = pcc(torch_actions, jax_actions)
    print("PCC:", score)


if __name__ == "__main__":
    main()
