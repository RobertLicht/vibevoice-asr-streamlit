import torch


def main():
    print(f"Utilized version of torch: {torch.__version__}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Selected hardware accelerator: {device}")
    if device == "cuda":
        print(f"Utilized hardware: {torch.cuda.get_device_name(0)}")


if __name__ == "__main__":
    main()
