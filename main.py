# --- IMPORTS
import torch, clip
import matplotlib.pyplot as plt

# SCRIPTS
import FONTS.get_fonts
import UTILS.set_up_matplotlib
import DATASETS.get_dataset
from DATASETS.load_dataset import get_images
import DATASETS.LABELS.load_labels
from UTILS.evaluation import cal_evaluation

if __name__ == '__main__':
    images = get_images()
    # LABELS
    labels = torch.load("DATASETS/LABELS/labels.pt")
    super_labels, basic_labels, hierarchy = labels["SUPERORDINATES"], labels["BASICS"], labels["CLUSTERING"]

    # MODEL
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device, jit=False)
    model_name = "clip"  # Name of the model
    preprocess_fn = preprocess  # Preprocessing to be applied on raw images
    tokenize_fn = clip.tokenize  # Tokenize function

    contexts = ["a photo of a "]
    dataset_size = 274

    cal_evaluation('', device, model, preprocess, images, model_name, contexts, hierarchy, tokenize_fn) #  the results are saved with the model name in the data directory.


