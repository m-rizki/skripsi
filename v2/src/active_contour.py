import numpy as np
from scipy.interpolate import RectBivariateSpline
from skimage.filters import gaussian, sobel
from skimage.util import img_as_float


class ActiveContour:
    """Class dasar untuk Algoritma Active Contour (Snake)."""

    def __init__(
        self,
        alpha=0.015,
        beta=10,
        gamma=0.001,
        sigma=3.5,
        max_iterations=500,
        sample_points=400,
    ):
        """
        Inisialisasi snake

        Args:
            alpha (float): Parameter yg mengontrol elastisitas (elasticity)
            beta (float): Parameter yg mengontrol kekakuan (stiffness)
            gamma (float): Langkah Waktu (Time Step)
            sigma (float): Sigma Gaussian Blur untuk external energy
            max_iterations (int): Maksimum Iterasi
            sample_points (int): Jumlah titik yang membentuk kurva snake
        """
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.sigma = sigma
        self.max_iterations = max_iterations
        self.sample_points = sample_points

    def initialize_snake(self, center_row, center_col, radius):
        """
        Inisialisasi snake

        Args:
            center_row (int): Koordinat baris pusat
            center_col (int): Koordinat kolom pusat
            radius (int): Radius lingkaran awal

        Returns:
            np.ndarray: Koordinat snake awal (N, 2) sebagai [x, y]
        """
        theta = np.linspace(0, 2 * np.pi, self.sample_points)
        r = center_row + radius * np.sin(theta)
        c = center_col + radius * np.cos(theta)
        snake_init = np.array([r, c]).T
        # konversi ke format [x, y]
        return snake_init[:, ::-1]

    def compute_external_energy(self, image):
        """
        Abstract method sebagai placeholder

        Hitung energi eksternal dari citra.

        Args:
            image (np.ndarray): Citra grayscale

        Returns:
            np.ndarray: Bidang energi eksternal
        """
        raise NotImplementedError(
            "Subclass harus mengimplementasikan compute_external_energy"
        )

    def compute_internal_energy_matrix(self, n_points):
        """
        Hitung matriks energi internal untuk deformasi snake.

        Args:
            n_points (int): Jumlah titik dalam snake

        Returns:
            np.ndarray: Matriks energi internal
        """
        a = self.beta
        b = -(4 * self.beta + self.alpha)
        c = 6 * self.beta + 2 * self.alpha

        eye_n = np.eye(n_points, dtype=float)
        c_axis = c * eye_n
        b_axis = b * (np.roll(eye_n, -1, axis=0) + np.roll(eye_n, -1, axis=1))
        a_axis = a * (np.roll(eye_n, -2, axis=0) + np.roll(eye_n, -2, axis=1))

        return c_axis + b_axis + a_axis

    def evolve(self, initial_snake, external_energy):
        """
        Evolusi snake menggunakan algoritma active contour.

        Args:
            initial_snake (np.ndarray): Koordinat snake awal
            external_energy (np.ndarray): Bidang energi eksternal

        Returns:
            np.ndarray: Koordinat snake akhir
        """
        raise NotImplementedError("Subclass harus mengimplementasikan evolve")


class IntegerActiveContour(ActiveContour):
    """Implementasi Active Contour integer."""

    def compute_external_energy(self, image):
        ext = gaussian(image, self.sigma)
        ext = sobel(ext)
        return -(ext**2)

    def evolve(self, initial_snake, external_energy):
        x = initial_snake[:, 0]
        y = initial_snake[:, 1]
        n = len(x)

        # hitung matrix energi internal dan inversnya
        A = self.compute_internal_energy_matrix(n)
        eye_n = np.eye(n, dtype=float)
        inv = np.linalg.inv(eye_n + self.gamma * A)  # Acton, Ivins matrix equation

        # potential force
        gy, gx = np.gradient(external_energy)

        # Evolution
        xt, yt = np.copy(x), np.copy(y)

        for iteration in range(self.max_iterations):
            fx = np.array([])
            fy = np.array([])

            for i in range(n):
                row_idx = np.round(yt[i]).astype(int)
                col_idx = np.round(xt[i]).astype(int)

                row_idx = np.clip(row_idx, 0, gx.shape[0] - 1)
                col_idx = np.clip(col_idx, 0, gx.shape[1] - 1)

                fx = np.append(fx, gx[row_idx, col_idx])
                fy = np.append(fy, gy[row_idx, col_idx])

            # Update
            xn = np.dot(inv, xt + self.gamma * fx)
            yn = np.dot(inv, yt + self.gamma * fy)

            xt = xn
            yt = yn

        return np.array([xt, yt]).T


class InterpolationActiveContour(ActiveContour):
    """Implementasi Active Contour yg ditambahkan interpolasi."""

    def __init__(self, max_px_move=1.0, **kwargs):
        """
        Inisialisasi dengan parameter khusus interpolasi.

        Args:
            max_px_move (float): Pergerakan piksel maksimum per iterasi
            **kwargs: Parameter kelas dasar
        """
        super().__init__(**kwargs)
        self.max_px_move = max_px_move

    def compute_external_energy(self, image):
        """Hitung energi eksternal menggunakan Gaussian blur dan Sobel edge detection."""
        ext = gaussian(image, self.sigma)
        ext = img_as_float(ext)
        ext = ext.astype(float, copy=False)
        return sobel(ext)

    def evolve(self, initial_snake, external_energy):
        x = initial_snake[:, 0].astype(float)
        y = initial_snake[:, 1].astype(float)
        n = len(x)

        # hitung matrix energi internal dan inversnya
        A = self.compute_internal_energy_matrix(n)
        eye_n = np.eye(n, dtype=float)
        inv = np.linalg.inv(A + self.gamma * eye_n)
        inv = inv.astype(float, copy=False)

        # potential force
        gy, gx = np.gradient(external_energy)

        # Fungsi interpolasi
        intp_gx = RectBivariateSpline(
            np.arange(gx.shape[1]), np.arange(gx.shape[0]), gx.T, kx=2, ky=2, s=0
        )

        intp_gy = RectBivariateSpline(
            np.arange(gy.shape[1]), np.arange(gy.shape[0]), gy.T, kx=2, ky=2, s=0
        )

        # Evolution
        xt, yt = np.copy(x), np.copy(y)

        for i in range(self.max_iterations):
            fx = intp_gx(xt, yt, dx=0, grid=False).astype(float, copy=False)
            fy = intp_gy(xt, yt, dy=0, grid=False).astype(float, copy=False)

            xn = np.dot(inv, self.gamma * xt + fx)
            yn = np.dot(inv, self.gamma * yt + fy)

            dx = self.max_px_move * np.tanh(xn - xt)
            dy = self.max_px_move * np.tanh(yn - yt)

            xt += dx
            yt += dy

        return np.array([xt, yt]).T
