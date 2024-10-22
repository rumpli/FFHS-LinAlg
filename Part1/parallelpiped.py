import numpy as np
import matplotlib.pyplot as plt


class Parallelepiped:
    def __init__(self, support_vector: np.ndarray, vector_a: np.ndarray, vector_b: np.ndarray, vector_c: np.ndarray):
        """
        Initialisiert das Parallelepiped mit einem Stützpunkt s und drei Vektoren a, b, c, die
        das Parallelepiped definieren.
        Erwartet NumPy-Arrays mit genau 3 Elementen.

        :param support_vector: Numpy Array des Stützvektor
        :param vector_a: Numpy Array des Vektors a
        :param vector_b: Numpy Array des Vektors b
        :param vector_c: Numpy Array des Vektors c
        """
        self.support_vector = support_vector
        self.vector_a = vector_a
        self.vector_b = vector_b
        self.vector_c = vector_c

        # Überprüfen der Vektoren
        self.validate_vectors()

        # Punkte berechnen
        self.points = self.calculate_corners()

        # Überprüfen, ob alle Punkte im ersten Oktant liegen
        self.check_first_octant()

    def validate_vectors(self):
        """Überprüft, ob alle Vektoren (s, a, b, c) NumPy-Arrays mit genau 3 Elementen sind."""
        vectors = {
            "Stützvektor s": self.support_vector,
            "Vektor a": self.vector_a,
            "Vektor b": self.vector_b,
            "Vektor c": self.vector_c
        }
        for name, vec in vectors.items():
            if not isinstance(vec, np.ndarray):
                raise TypeError(f"{name} muss ein NumPy-Array sein.")
            if vec.shape != (3,):
                raise ValueError(f"{name} muss genau 3 Elemente haben, hat aber: {vec.shape[0]}.")

    def calculate_corners(self) -> dict:
        """
        Berechnet die Eckpunkte des Parallelepipeds basierend auf den Vektoren.

        :return: Dictionary der berechneten Eckpunkte des Parallelepipeds
        """
        points = {
            's': self.support_vector,
            'a': self.support_vector + self.vector_a,
            'b': self.support_vector + self.vector_b,
            'c': self.support_vector + self.vector_c,
            'd': self.support_vector + self.vector_a + self.vector_b,  # a + b
            'e': self.support_vector + self.vector_b + self.vector_c,  # b + c
            'f': self.support_vector + self.vector_a + self.vector_c,  # a + c
            'g': self.support_vector + self.vector_a + self.vector_b + self.vector_c  # a + b + c
        }
        return points

    def check_first_octant(self):
        """Überprüft, ob alle Punkte im ersten Oktanten liegen."""
        for name, point in self.points.items():
            if any(n <= 0 for n in point):
                raise ValueError(f"Das Objekt befindet sich nicht vollständig im ersten Oktanten des R^3. Punkt {name}: {point}")


def check_projection_center(camera_position: np.ndarray, parallelepiped: Parallelepiped):
    """
    Überprüft, ob das Projektionszentrum im ersten Oktanten liegt
    und das Parallelepiped zwischen Zentrum und xy-Ebene liegt.

    :param camera_position: Das Projektionszentrum (die "Kamera").
    :param parallelepiped: Ein Parallelepiped-Objekt, das mehrere 3D-Punkte enthält,
                           die die Eckpunkte des Parallelepipeds im 3D-Raum darstellen.
    """
    if any(n <= 0 for n in camera_position):
        raise ValueError("Das Projektionszentrum muss im ersten Oktanten des R^3 liegen.")

    # Höchsten z-Wert des Parallelepipeds finden
    max_z = max(point[2] for point in parallelepiped.points.values())

    if camera_position[2] <= max_z:
        raise ValueError(
            "Das Projektionszentrum muss über dem Parallelepiped liegen "
            "(z-Wert des Zentrums muss größer als der höchste Punkt des Parallelepipeds sein).")


def calculate_coordinate(x_p: float, x_u: float, z_p: float, z_u: float) -> float:
    """
    Berechnet die projizierte Koordinate entlang einer Achse (x oder y) in einer Zentralprojektion auf die xy-Ebene.

    :param x_p: Die Koordinate des Punktes entlang der betrachteten Achse (x oder y) im 3D-Raum.
    :param x_u: Die entsprechende Koordinate des Projektionszentrums entlang der betrachteten Achse (x oder y).
    :param z_p: Die z-Koordinate des Punktes im 3D-Raum.
    :param z_u: Die z-Koordinate des Projektionszentrums im 3D-Raum.
    :return: Die projizierte Koordinate des Punktes entlang der betrachteten Achse (x oder y) auf die xy-Ebene (z = 0).
    """
    return (x_p - x_u * z_p / z_u) / (1 - z_p / z_u)


def project_point(point: np.ndarray, camera_position: np.ndarray) -> np.ndarray:
    """
    Projiziert einen Punkt aus dem 3D-Raum auf die xy-Ebene (z = 0) basierend auf dem gegebenen Projektionszentrum.

    :param point: Ein 3D-Punkt, der als NumPy-Array der Form [x_p, y_p, z_p] vorliegt.
                  Er repräsentiert den Punkt im 3D-Raum, der projiziert werden soll.
    :param camera_position: Ein 3D-Punkt, der als Projektionszentrum dient, angegeben als NumPy-Array der Form [x_u, y_u, z_u].
                            Dieser Punkt beschreibt die Position des Projektionszentrums ("Kamera") im 3D-Raum.
    :return: Ein NumPy-Array mit zwei Werten [x_proj, y_proj], die die projizierten Koordinaten des Punktes
             auf die xy-Ebene (z = 0) darstellen.
    """
    x_u, y_u, z_u = camera_position
    x_p, y_p, z_p = point

    # Berechnung der projizierten x- und y-Koordinaten auf die xy-Ebene
    return np.array([
        calculate_coordinate(x_p, x_u, z_p, z_u),  # Projiziere die x-Koordinate
        calculate_coordinate(y_p, y_u, z_p, z_u)  # Projiziere die y-Koordinate
    ])


def project(parallelepiped: Parallelepiped, camera_point: np.ndarray) -> np.ndarray:
    """
    Projiziert alle Eckpunkte eines Parallelepipeds auf die xy-Ebene (z = 0) basierend auf dem Projektionszentrum (Kamera).

    :param parallelepiped: Ein Parallelepiped-Objekt, das mehrere 3D-Punkte enthält,
                           die die Eckpunkte des Parallelepipeds im 3D-Raum darstellen.
    :param camera_point: Ein 3D-Punkt, der das Projektionszentrum (Kamera) darstellt, angegeben als NumPy-Array [x_u, y_u, z_u].
                         Dieser Punkt beschreibt die Position der Kamera bzw. des Projektionszentrums im 3D-Raum.
    :return: Ein NumPy-Array, das die projizierten Eckpunkte des Parallelepipeds auf die xy-Ebene enthält.
             Jeder projizierte Punkt wird als ein Array der Form [x_proj, y_proj] dargestellt.
    """

    # Überprüfen der Projektionszentren
    check_projection_center(camera_point, parallelepiped)
    return np.array([project_point(point, camera_point) for point in parallelepiped.points.values()])


def draw_line(start: np.ndarray, finish: np.ndarray, color: str):
    """
    Zeichnet eine Linie zwischen zwei Punkten im 2D-Raum.

    :param start: Ein NumPy-Array der Form [x_start, y_start], das die Startkoordinate der Linie im 2D-Raum repräsentiert.
    :param finish: Ein NumPy-Array der Form [x_finish, y_finish], das die Endkoordinate der Linie im 2D-Raum repräsentiert.
    :param color: Ein String, der die Farbe der Linie angibt (z. B. 'r' für rot, 'g' für grün, 'b' für blau).
    """
    plt.plot([start[0], finish[0]], [start[1], finish[1]], color, linestyle='-')


def draw_projection(projection: np.ndarray, title: str):
    """
    Zeichnet die Projektion eines Parallelepipeds auf die xy-Ebene (z = 0).

    :param projection: Ein NumPy-Array, das die projizierten 2D-Koordinaten der Eckpunkte des Parallelepipeds enthält.
                       Jede projizierte Koordinate wird als ein Array der Form [x_proj, y_proj] dargestellt.
    :param title: Ein String, der den Titel des Plots angibt.
    """

    fig, ax = plt.subplots()

    # Setze den Hintergrund auf ein helles Grau
    ax.set_facecolor('#F8F8F8')

    # Extrahiere die projizierten Punkte
    point_s, point_a, point_b, point_c, point_d, point_e, point_f, point_g = projection

    # Definiere die Flächen im 2D-Projektionsraum (jede Fläche wird durch vier Punkte beschrieben)
    faces = [
        [point_s, point_a, point_d, point_b],  # Unterseite
        [point_c, point_f, point_g, point_e],  # Oberseite
        [point_a, point_f, point_g, point_d],  # Rechte Seite
        [point_s, point_b, point_e, point_c],  # Linke Seite
        [point_s, point_c, point_f, point_a],  # Vorderseite
        [point_b, point_d, point_g, point_e]   # Rückseite
    ]

    # Farben für jede Fläche
    face_colors = ['cyan', 'magenta', 'yellow', 'red', 'green', 'blue']

    # Zeichne die Flächen in der Projektion
    for i, face in enumerate(faces):
        # Extrahiere die x- und y-Koordinaten der projizierten Punkte
        xs = [p[0] for p in face]
        ys = [p[1] for p in face]
        ax.fill(xs, ys, face_colors[i], alpha=0.1, edgecolor='k')

    # Grün: Kanten SA, BD, CF, EG
    draw_line(point_s, point_a, 'g')  # s -> a
    draw_line(point_b, point_d, 'g')  # b -> d
    draw_line(point_c, point_f, 'g')  # c -> f
    draw_line(point_e, point_g, 'g')  # e -> g

    # Rot: Kanten SB, AD, CE, FG
    draw_line(point_s, point_b, 'r')  # s -> b
    draw_line(point_a, point_d, 'r')  # a -> d
    draw_line(point_c, point_e, 'r')  # c -> e
    draw_line(point_f, point_g, 'r')  # f -> g

    # Blau: Kanten SC, AF, BE, DG
    draw_line(point_s, point_c, 'b')  # s -> c
    draw_line(point_a, point_f, 'b')  # a -> f
    draw_line(point_b, point_e, 'b')  # b -> e
    draw_line(point_d, point_g, 'b')  # d -> g

    plt.title(title)
    plt.xlabel("X-Achse")
    plt.ylabel("Y-Achse")
    plt.axis('equal')
    plt.grid(True)
    plt.show()


def main():
    # Parallelepiped-Daten
    vector_s = np.array([1, 1, 1])  # Stützvektor
    vector_a = np.array([1, 2, 3])  # Vektor a
    vector_b = np.array([3, 1, 3])  # Vektor b
    vector_c = np.array([3, 2, 1])  # Vektor c

    # Erstellung des Parallelpiped Objekt
    parallelepiped_object = Parallelepiped(vector_s, vector_a, vector_b, vector_c)

    # Projektionszentrum (camera_point = np.array([x, y, z])) beeinflusst die Projektion
    camera_point1 = np.array([10, 5, 20])
    camera_point2 = np.array([1, 4, 20])

    # Projektionen berechnen
    projection1 = project(parallelepiped_object, camera_point1)
    projection2 = project(parallelepiped_object, camera_point2)

    # Projektionen zeichnen
    draw_projection(projection1, "Zentralprojektion 1")
    draw_projection(projection2, "Zentralprojektion 2")


if __name__ == '__main__':
    main()
