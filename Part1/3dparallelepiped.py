from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
import matplotlib.pyplot as plt

from Part1.parallelepiped import Parallelepiped, project, draw_projection


def draw_3d_parallelepiped(parallelepiped: Parallelepiped, title: str):
    """
    Zeichnet das Parallelepiped im 3D-Raum.

    :param parallelepiped: Ein Parallelepiped-Objekt, das mehrere 3D-Punkte enthält,
                           die die Eckpunkte des Parallelepipeds im 3D-Raum darstellen.
    :param title: Ein String, der den Titel des Plots angibt.
    """
    points = parallelepiped.points
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Extrahiere die einzelnen Punkte
    point_s, point_a, point_b, point_c, point_d, point_e, point_f, point_g = points.values()

    # Definiere die Flächen des Parallelepipeds (jede Fläche wird durch vier Punkte beschrieben)
    faces = [
        [point_s, point_a, point_d, point_b],  # Unterseite
        [point_c, point_f, point_g, point_e],  # Oberseite
        [point_a, point_f, point_g, point_d],  # Rechte Seite
        [point_s, point_b, point_e, point_c],  # Linke Seite
        [point_s, point_c, point_f, point_a],  # Vorderseite
        [point_b, point_d, point_g, point_e]  # Rückseite
    ]

    # Farben für jede Fläche
    face_colors = ['cyan', 'magenta', 'yellow', 'red', 'green', 'blue']

    # Zeichne die Flächen
    for i, face in enumerate(faces):
        poly3d = [[face[0], face[1], face[2], face[3]]]  # Jede Fläche besteht aus 4 Punkten
        ax.add_collection3d(Poly3DCollection(poly3d, facecolors=face_colors[i], linewidths=1, edgecolors='k', alpha=0.1))

    # Zeichne die Kanten des Parallelepipeds
    def draw_3d_line(start, end, color):
        """Hilfsfunktion, um eine Linie in 3D zu zeichnen."""
        ax.plot([start[0], end[0]], [start[1], end[1]], [start[2], end[2]], color)

    # Grün: Kanten SA, BD, CF, EG
    draw_3d_line(point_s, point_a, 'g')  # s -> a
    draw_3d_line(point_b, point_d, 'g')  # b -> d
    draw_3d_line(point_c, point_f, 'g')  # c -> f
    draw_3d_line(point_e, point_g, 'g')  # e -> g

    # Rot: Kanten SB, AD, CE, FG
    draw_3d_line(point_s, point_b, 'r')  # s -> b
    draw_3d_line(point_a, point_d, 'r')  # a -> d
    draw_3d_line(point_c, point_e, 'r')  # c -> e
    draw_3d_line(point_f, point_g, 'r')  # f -> g

    # Blau: Kanten SC, AF; BE, DG
    draw_3d_line(point_s, point_c, 'b')  # s -> c
    draw_3d_line(point_a, point_f, 'b')  # a -> f
    draw_3d_line(point_b, point_e, 'b')  # b -> e
    draw_3d_line(point_d, point_g, 'b')  # d -> g

    # Achsen beschriften
    ax.set_xlabel('X-Achse')
    ax.set_ylabel('Y-Achse')
    ax.set_zlabel('Z-Achse')

    # Diagrammtitel und Anzeige
    plt.title(title)
    plt.show()


def main():
    # Parallelepiped-Daten
    vector_s = np.array([1, 1, 1])  # Stützvektor
    vector_a = np.array([1, 2, 3])  # Vektor a
    vector_b = np.array([3, 1, 3])  # Vektor b
    vector_c = np.array([3, 2, 1])  # Vektor c

    # Erstellung des Parallelpiped Objekt
    parallelepiped_object = Parallelepiped(vector_s, vector_a, vector_b, vector_c)

    # Parallelepiped im 3D Raum zeichnen
    draw_3d_parallelepiped(parallelepiped_object, "3D Parallelepiped")

    # Projektionszentrum (camera_point1 = np.array([x, y, z])) beeinflusst die Projektion
    camera_point1 = np.array([10, 5, 20])
    camera_point2 = np.array([1, 5, 20])

    # Projektionen berechnen
    projection1 = project(parallelepiped_object, camera_point1)
    projection2 = project(parallelepiped_object, camera_point2)

    # Projektionen zeichnen
    draw_projection(projection1, "Zentralprojektion 1")
    draw_projection(projection2, "Zentralprojektion 2")


if __name__ == '__main__':
    main()
