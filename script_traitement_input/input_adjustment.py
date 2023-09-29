import os
import cv2
import numpy as np


def augmenter_contraste_et_saturation(image_path, output_folder, alpha_contraste=1, beta_contraste=-30, alpha_saturation=1.5):
    # Lire l'image
    image = cv2.imread(image_path)

    # Appliquer une augmentation de contraste à chaque canal
    canal_blue = cv2.convertScaleAbs(image[:, :, 0], alpha=alpha_contraste, beta=beta_contraste)
    canal_green = cv2.convertScaleAbs(image[:, :, 1], alpha=alpha_contraste, beta=beta_contraste)
    canal_red = cv2.convertScaleAbs(image[:, :, 2], alpha=alpha_contraste, beta=beta_contraste)

    # Fusionner les canaux avec augmentation de contraste en une image RGB
    image_contrast = cv2.merge([canal_blue, canal_green, canal_red])

    # Convertir l'image en espace colorimétrique HSV
    image_hsv = cv2.cvtColor(image_contrast, cv2.COLOR_BGR2HSV)

    # Augmenter la saturation
    image_hsv[:, :, 1] = np.clip(alpha_saturation * image_hsv[:, :, 1], 0, 255).astype(np.uint8)

    # Revenir à l'espace colorimétrique BGR
    image_output = cv2.cvtColor(image_hsv, cv2.COLOR_HSV2BGR)

    # Créer le chemin de sortie et sauvegarder l'image
    output_path = os.path.join(output_folder, os.path.basename(image_path))
    cv2.imwrite(output_path, image_output)

# Dossier d'entrée contenant les images JPEG RGB
dossier_entree = r"/Users/anton/Desktop/script_magick/input_jpg"

# Dossier de sortie où vous souhaitez enregistrer les images JPEG égalisées
dossier_sortie = r"/Users/anton/Desktop/script_magick/output_jpg"

# Créer le dossier de sortie s'il n'existe pas
if not os.path.exists(dossier_sortie):
    os.makedirs(dossier_sortie)

# Liste les fichiers JPEG dans le dossier d'entrée
fichiers_jpeg = [f for f in os.listdir(dossier_entree) if f.endswith('.jpg')]

# Parcourir tous les fichiers JPEG
for fichier_jpeg in fichiers_jpeg:
    chemin_entree = os.path.join(dossier_entree, fichier_jpeg)
    #normaliser_canaux(chemin_entree, dossier_sortie)
    augmenter_contraste_et_saturation(chemin_entree, dossier_sortie)

print("Normalisation des canaux terminée.")
