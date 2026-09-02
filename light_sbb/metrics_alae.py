import os

import cv2
import numpy as np
import onnxruntime
from PIL import Image
from insightface.app import FaceAnalysis
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

        # Flattened 1024x1024 RGB vectors are large, so the cosine is computed in
        # float32 in place rather than through sklearn's float64 copies.
        vec1 = np.asarray(img1, dtype=np.float32).ravel()
        vec2 = np.asarray(img2, dtype=np.float32).ravel()

        sim = float(vec1 @ vec2 / (np.linalg.norm(vec1) * np.linalg.norm(vec2)))
        similarities[img_name1] = sim
        sim_values.append(sim)

    mean_sim = np.mean(sim_values)
    std_sim = np.std(sim_values)
    print(f"Nombre d'images : {len(sim_values)}")
    print(f"Cosine Similarity moyenne : {mean_sim:.4f}")
    print(f"Écart-type (std) : {std_sim:.4f}")

    return similarities


def compute_average_age(folder_path, n_threads=None):
    """Estimate ages from PNG face images in a folder using InsightFace.

    Args:
        folder1: Path to folder.
        n_threads: Cap on CPU threads; None leaves ONNX Runtime and OpenCV free to
            take every core, which starves a shared machine.

    Returns:
        np.array of estimated ages for all images in folder.
    """
    session_options = None
    if n_threads is not None:
        # ONNX Runtime and OpenCV keep their own pools and ignore OMP_NUM_THREADS.
        cv2.setNumThreads(n_threads)
        session_options = onnxruntime.SessionOptions()
        session_options.intra_op_num_threads = n_threads
        session_options.inter_op_num_threads = n_threads

    try:
        app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'],
                           session_options=session_options)
    except TypeError:
        # Older insightface builds do not forward session options.
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
