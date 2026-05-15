import os
import gdown
def main():
    check = "model_weights/model_weights.pth"
    if not os.path.exists(check):
        os.makedirs("model_weights", exist_ok=True)
        print("Downloading model weights...")
        file_id = "1M_wyp9oSxldQ49ezXeErBeu-F0fJ9fa0"
        gdown.download(id=file_id, output=check, quiet=False)
        print("Successfully saved weights")
    else:
        print("Weights already downloaded")

if __name__ == "__main__":
    main()