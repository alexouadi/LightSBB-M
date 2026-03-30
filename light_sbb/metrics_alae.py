import os

import cv2
import numpy as np
from PIL import Image
from insightface.app import FaceAnalysis
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm


def compute_cosine_similarity(folder1, folder2):
    """Compute cosine similarity for paired PNG images from two folders.

    Args:
        folder1: Path to folder 1.
        folder2: Path to folder 2.

    Returns:
        Float similarity score.
    """
    similarities = {}
    images1 = sorted([f for f in os.listdir(folder1) if f.endswith(".png")])
    images2 = sorted([f for f in os.listdir(folder2) if f.endswith(".png")])

    if len(images1) != len(images2):
        raise ValueError("Les deux dossiers n'ont pas le même nombre d'images.")
    sim_values = []

    for img_name1, img_name2 in tqdm(zip(images1, images2)):
        path1 = os.path.join(folder1, img_name1)
        path2 = os.path.join(folder2, img_name2)

        img1 = Image.open(path1).convert("RGB")
        img2 = Image.open(path2).convert("RGB")

        vec1 = np.array(img1).flatten().reshape(1, -1)
        vec2 = np.array(img2).flatten().reshape(1, -1)

        sim = cosine_similarity(vec1, vec2)[0][0]
        similarities[img_name1] = sim
        sim_values.append(sim)

    mean_sim = np.mean(sim_values)
    std_sim = np.std(sim_values)
    print(f"Nombre d'images : {len(sim_values)}")
    print(f"Cosine Similarity moyenne : {mean_sim:.4f}")
    print(f"Écart-type (std) : {std_sim:.4f}")

    return similarities


def compute_average_age(folder_path):
    """Estimate ages from PNG face images in a folder using InsightFace.
    
    Args:
        folder1: Path to folder.
    
    Returns:
        np.array of estimated ages for all images in folder.
    """
    app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
    app.prepare(ctx_id=0)

    ages = []

    for file_name in tqdm(os.listdir(folder_path)):
        if not file_name.lower().endswith(".png"):
            continue

        img_path = os.path.join(folder_path, file_name)
        img = cv2.imread(img_path)
        if img is None:
            continue

        faces = app.get(img)
        for face in faces:
            if hasattr(face, 'age') and face.age is not None:
                ages.append(face.age)

    print(f"Overall average child age: {float(np.mean(ages)):.2f}")
    print(f"Numbers of age: {len(ages)}")
    print(f"Overall median child age: {float(np.median(np.array(ages))):.2f}")
    return ages
