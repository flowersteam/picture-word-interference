import os
import torch
import urllib
import zipfile
import json
import torchvision

# ----- LOAD/CREATE LABELS
if not os.path.exists("DATASETS/LABELS/labels.pt"):
    super_labels = set()
    basic_labels = set()
    # --- ADDING LABELS FROM MS-COCO
    if not os.path.exists("DATASETS/LABELS/annotations.json"):
        urllib.request.urlretrieve("http://images.cocodataset.org/annotations/annotations_trainval2017.zip",
                                   "DATASETS/LABELS/mscoco_annotations.zip")
        with zipfile.ZipFile("DATASETS/LABELS/mscoco_annotations.zip", 'r') as zip_file:
            with open("DATASETS/LABELS/annotations.json", 'wb') as f:
                f.write(zip_file.read('annotations/instances_val2017.json'))
    with open("DATASETS/LABELS/annotations.json") as an_file:
        annotations_coco = json.load(an_file)
        for labels in annotations_coco["categories"]:
            super_labels.add(labels["supercategory"])
            basic_labels.add(labels["name"])

    # --- MANUALY ADDING LABELS FROM CIFAR-100
    # "person" from ms-coco should not be a basic label, as CIFAR-100 has "man","woman","boy","girl","baby"
    basic_labels.remove("person")
    cifar100 = torchvision.datasets.CIFAR100(root="DATASETS/LABELS", train=False, download=True)
    basic_labels.update(cifar100.classes)

    # --- REMOVING MULTI-WORDS AND "s" DUPLICATES
    remove_list = []

    for label in basic_labels:
        if label + "s" in basic_labels:
            remove_list.append(label + "s")
        elif " " in label or "_" in label:
            remove_list.append(label)

    for remove_label in remove_list:
        basic_labels.remove(remove_label)

    # --- HIERARCHY
    hierarchy = {super: set() for super in sorted(list(super_labels))}
    cifar_manual_clustering = {
        "outdoor": ["bridge", "castle", "cloud", "forest", "house", "mountain", "orchid", "plain", "poppy", "road",
                    "rose", "sea", "skyscraper", "sunflower", "tulip"],
        "animal": ["bear", "beaver", "bee", "beetle", "butterfly", "camel", "caterpillar", "cattle", "chimpanzee",
                   "cockroach", "crab", "crocodile", "dinosaur", "dolphin", "elephant", "flatfish", "fox", "hamster",
                   "kangaroo", "leopard", "lion", "lizard", "lobster", "mouse", "otter", "porcupine", "possum",
                   "rabbit", "raccoon", "ray", "seal", "shark", "shrew", "skunk", "snail", "snake", "spider",
                   "squirrel", "tiger", "trout", "turtle", "whale", "wolf", "worm"],
        "vehicle": ["bicyle", "bus", "motorcycle", "rocket", "streetcar", "tank", "tractor", "train"],
        "kitchen": ["bottle", "bowl", "can", "cup", "plate"],
        "electronic": ["clock", "keyboard", "lamp", "telephone", "television"],
        "person": ["baby", "boy", "girl", "man", "woman"],
        "food": ["apple", "mushroom", "orange", "pear"],
        "furniture": ["bed", "chair", "couch", "table", "wardrobe"]
    }

    for basic_label in basic_labels:
        basic_in_hierarchy = False

        for labels in annotations_coco["categories"]:
            if basic_label == labels["name"]:
                hierarchy[labels["supercategory"]].add(basic_label)
                basic_in_hierarchy = True

        if not basic_in_hierarchy:
            for category in cifar_manual_clustering:
                if basic_label in cifar_manual_clustering[category]:
                    hierarchy[category].add(basic_label)
                    basic_in_hierarchy = True

        assert (basic_in_hierarchy)

    # --- SAVE LABELS
    super_labels = list(hierarchy.keys())
    basic_labels = []

    for key in hierarchy:
        for basic_label in hierarchy[key]:
            basic_labels.append(basic_label)

    torch.save({"SUPERORDINATES": super_labels, "BASICS": basic_labels, "CLUSTERING": hierarchy},
               "DATASETS/LABELS/labels.pt")