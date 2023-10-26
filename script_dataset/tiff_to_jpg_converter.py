import os
import tifffile
from osgeo import gdal
from PIL import Image
import numpy as np
import cv2

# Dossier d'entrée contenant les images GeoTIFF
dossier_entree = r"/Users/anton/Desktop/IA/challenge/dataset/test/images"

# Dossier de sortie où vous souhaitez enregistrer les images JPEG
dossier_sortie =  r"/Users/anton/Desktop/IA/challenge/dataset/test1/images"

# Créer le dossier de sortie s'il n'existe pas
if not os.path.exists(dossier_sortie):
    os.makedirs(dossier_sortie)

# Liste les fichiers GeoTIFF dans le dossier d'entrée
fichiers_geotiff = [f for f in os.listdir(dossier_entree) if f.endswith('.tif')]

# Parcours de tous les fichiers GeoTIFF
for fichier_geotiff in fichiers_geotiff:
    fichier_entree = os.path.join(dossier_entree, fichier_geotiff)


    imageData = tifffile.imread(fichier_entree)


    # Extraire les canaux RGB
    channel_0 = imageData[:, :, 0]  
    channel_1 = imageData[:, :, 1] 
    channel_2 = imageData[:, :, 2]

    # Créer une image en couleur en combinant les canaux
    rgb_image = cv2.merge([channel_0, channel_1, channel_2])

    cv2.imwrite('imageRGBTemp.tif', rgb_image)


    # Ouvrir le fichier GeoTIFF en utilisant GDAL
    dataset = gdal.Open('imageRGBTemp.tif')

    # Lire la bande RGB (Red-Green-Blue)
    bande_red = dataset.GetRasterBand(1).ReadAsArray()
    bande_green = dataset.GetRasterBand(2).ReadAsArray()
    bande_blue = dataset.GetRasterBand(3).ReadAsArray()

    # Normalisez les valeurs des canaux pour qu'elles se situent dans la plage [0, 255]
    bande_red = np.interp(bande_red, (bande_red.min(), bande_red.max()), (0, 255))
    bande_green = np.interp(bande_green, (bande_green.min(), bande_green.max()), (0, 255))
    bande_blue = np.interp(bande_blue, (bande_blue.min(), bande_blue.max()), (0, 255))

    # Convertissez vos bandes en images avec le mode 'L' (niveaux de gris)
    image_red = Image.fromarray(bande_red.astype(np.uint8), mode='L')
    image_green = Image.fromarray(bande_green.astype(np.uint8), mode='L')
    image_blue = Image.fromarray(bande_blue.astype(np.uint8), mode='L')

    # Fusionnez les images en tant qu'image RGB
    image_rgb = Image.merge('RGB', (image_red, image_green, image_blue))
    
    # Créer le nom du fichier de sortie en remplaçant l'extension
    nom_fichier_sortie = os.path.splitext(fichier_geotiff)[0] + '.jpg'
    fichier_sortie = os.path.join(dossier_sortie, nom_fichier_sortie)
    
    # Enregistrer l'image au format JPEG
    image_rgb.save(fichier_sortie, 'JPEG')

# Fermer le dataset GDAL
dataset = None
os.remove("imageRGBTemp.tif") 

print("Conversion terminée.")