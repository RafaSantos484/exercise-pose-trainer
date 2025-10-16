import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=str, help="Path to the features folder")
    parser.add_argument(
        "model",
        type=str,
        choices=["random_forest", "svm"],
        help="Model type to use. Choices are: 'svm' or 'random_forest'",
    )
    parser.add_argument("--seed", type=int,
                        help="Seed used for reproducibility")

    args = parser.parse_args()

    base_path = args.path
    seed = args.seed
