import os
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as compare_ssim
from PIL import Image, ImageEnhance, ImageOps, ImageFilter


# 1 PRETRAITEMENT
def preprocess(image_path: str) -> np.ndarray:
    
    # Charger image
    image = Image.open(image_path)
    
    # Conversion en niveaux de gris
    image = image.convert("L")
    
    # Redimensionnement 
    image = image.resize((300, 300))
    
    #  Egalisation histogramme
    image = ImageOps.equalize(image)
    
    #  Amélioration du contraste
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(2.0)
    
    #  Binarisation 
    image = image.point(lambda x: 255 if x > 128 else 0)
    
    # Extraction des contours
    image = image.filter(ImageFilter.FIND_EDGES)
    
    return np.array(image)

