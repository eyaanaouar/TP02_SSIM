import os
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as compare_ssim
from PIL import Image, ImageEnhance, ImageOps, ImageFilter


# =========================
# 1) PRETRAITEMENT
# =========================
def preprocess(image_path: str) -> np.ndarray:
    
    # Charger image
    image = Image.open(image_path)
    
    # 1. Conversion en niveaux de gris
    image = image.convert("L")
    
    # 2. Redimensionnement 300x300
    image = image.resize((300, 300))
    
    # 3. Egalisation histogramme
    image = ImageOps.equalize(image)
    
    # 4. Amélioration du contraste
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(2.0)
    
    # 5. Binarisation (seuil = 128)
    image = image.point(lambda x: 255 if x > 128 else 0)
    
    # 6. Extraction des contours
    image = image.filter(ImageFilter.FIND_EDGES)
    
    return np.array(image)


# =========================
# 2) CALCUL DE SIMILARITE
# =========================
def compute_similarity(img1_path: str, img2_path: str):
    
    img1 = preprocess(img1_path)
    img2 = preprocess(img2_path)
    
    similarity = compare_ssim(img1, img2, data_range=255)
    
    return similarity, img1, img2


# =========================
# 3) DECISION
# =========================
def decision(similarity, threshold=0.75):
    
    if similarity >= threshold:
        return "ACCEPTEE"
    else:
        return "REJETEE"


# =========================
# 4) TEST PRINCIPAL
# =========================
if __name__ == "__main__":
    
    img1_path = "empreinte1.png"
    img2_path = "empreinte2.png"
    
    similarity, img1, img2 = compute_similarity(img1_path, img2_path)
    
    result = decision(similarity)
    
    print("Score SSIM :", round(similarity, 4))
    print("Decision :", result)
    
    # =========================
    # AFFICHAGE
    # =========================
    plt.figure(figsize=(12,4))
    
    plt.subplot(1,3,1)
    plt.imshow(img1, cmap='gray')
    plt.title("Image 1 Traitee")
    plt.axis("off")
    
    plt.subplot(1,3,2)
    plt.imshow(img2, cmap='gray')
    plt.title("Image 2 Traitee")
    plt.axis("off")
    
    plt.subplot(1,3,3)
    plt.text(0.1, 0.5, f"SSIM = {round(similarity,4)}\n\nDecision : {result}", fontsize=12)
    plt.axis("off")
    
    plt.tight_layout()
    plt.show()