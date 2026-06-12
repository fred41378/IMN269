import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

# Import de l'image
img = Image.open("./Sol.png")

# Parametres de l'image
height = img.height
width = img.width

# Initialisation des points caracteristiques p.
# Dans notre cas, les points sont donnés dans les notes de cours
p1 = np.array([0.45, 1.28, 1])
p2 = np.array([2.58, 0.49, 1])
p3 = np.array([2.11, 2.66, 1])
p4 = np.array([4.40, 1.41, 1])


# Creation des droites parallèlles d.

d1 = np.cross(p1, p2)
d2 = np.cross(p3, p4)
d3 = np.cross(p1, p3)
d4 = np.cross(p2, p4)
print("d1 =", d1)
print("d2 =", d2)
print("d3 =", d3)
print("d4 =", d4)

# Identifications des points de fuite à l'aide des droite parallèlles

f1 = np.cross(d1, d2)
k1 = 100 / f1[2]
f1_norm = f1 * k1
f2 = np.cross(d3, d4)
k2 = 100 / f2[2]
f2_norm = f2 * k2
print("f1 =", f1)
print("f1 normalisé =", f1_norm)
print("f2 =", f2)
print("f2 normalisé =", f2_norm)

# Une fois fait, il faut trouver la droite à l'infini (l'horizon)
# à l'aide des points de fuite

d = np.cross(f1, f2)
print("d =", d)


# Création de la matrice H2 à l'aide de d (normalisé)
H2 = np.identity(3, dtype=float)

d_norm = d / d[2]
H2[-1, :] = d_norm
print("H2 =", H2)

# Création de l'image corrigé
H2_inv = np.linalg.inv(H2)
output = np.zeros_like(np.array(img))


for x in range(width):
    for y in range(height):
        p_out = np.array([x/100, y/100, 1], dtype=float)

        p_in = H2_inv @ p_out

        p_in = p_in / p_in[2]

        x_in = int(round(p_in[0] * 100))
        y_in = int(round(p_in[1] * 100))

        if 0 <= x_in < width and 0 <= y_in < height:
            output[y, x] = np.array(img)[y_in, x_in]

print("H2f1 :", H2 @ f1)
print("H2f2 :", H2 @ f2)
print("H2⁻ᵀ d : ", H2_inv.T @ d_norm)

# Affichage des résultats
plt.imshow(img)
plt.show()

plt.imshow(output)
plt.show()