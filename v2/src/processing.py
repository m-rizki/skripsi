from skimage.color import rgb2gray
from skimage.io import imread
from active_contour import InterpolationActiveContour, IntegerActiveContour


class SnakeProcessor:
    def __init__(self, dpi=96):
        """
        Inisialisasi prosesor.

        Args:
            dpi (int): DPI untuk menyimpan gambar
        """
        self.dpi = dpi
        self.snake_contour = None
        self.initial_snake = None
        self.external_energy = None

    def load_image(self, image_path):
        """
        Muat dan konversi citra ke grayscale.

        Args:
            image_path (str): Path ke file citra

        Returns:
            tuple: (citra_asli, citra_grayscale)
        """
        img = imread(image_path)
        img_gray = rgb2gray(img)
        return img, img_gray

    def process_with_integer(
        self,
        image_path,
        center_row,
        center_col,
        radius,
        alpha=1,
        beta=10,
        gamma=2,
        sigma=3.5,
        max_iterations=100,
        sample_points=100,
    ):
        """
        Proses citra menggunakan integer active contour.

        Args:
            image_path (str): Path ke citra input
            center_row, center_col (int): Pusat snake awal
            radius (int): Radius snake awal
            alpha, beta, gamma (float): Parameter snake
            sigma (float): Parameter Gaussian blur
            max_iterations (int): Iterasi maksimum
            sample_points (int): Jumlah titik snake

        Returns:
            tuple: (citra_asli, koordinat_snake_akhir, energi_eksternal)
        """
        img, img_gray = self.load_image(image_path)

        snake = IntegerActiveContour(
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            sigma=sigma,
            max_iterations=max_iterations,
            sample_points=sample_points,
        )

        initial_snake = snake.initialize_snake(center_row, center_col, radius)

        external_energy = snake.compute_external_energy(img_gray)

        final_snake = snake.evolve(initial_snake, external_energy)

        self.snake_contour = final_snake
        self.initial_snake = initial_snake
        self.external_energy = external_energy

        return img, final_snake, external_energy

    def process_with_interpolation(
        self,
        image_path,
        center_row,
        center_col,
        radius,
        alpha=0.015,
        beta=10,
        gamma=0.001,
        sigma=3.5,
        max_iterations=500,
        sample_points=400,
        max_px_move=1.0,
    ):
        """
        Proses citra menggunakan interpolation active contour.

        Args:
            image_path (str): Path ke citra input
            center_row, center_col (int): Pusat snake awal
            radius (int): Radius snake awal
            alpha, beta, gamma (float): Parameter snake
            sigma (float): Parameter Gaussian blur
            max_iterations (int): Iterasi maksimum
            sample_points (int): Jumlah titik snake
            max_px_move (float): Pergerakan piksel maksimum per iterasi

        Returns:
            tuple: (citra_asli, koordinat_snake_akhir, energi_eksternal)
        """
        img, img_gray = self.load_image(image_path)

        snake = InterpolationActiveContour(
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            sigma=sigma,
            max_iterations=max_iterations,
            sample_points=sample_points,
            max_px_move=max_px_move,
        )

        initial_snake = snake.initialize_snake(center_row, center_col, radius)

        external_energy = snake.compute_external_energy(img_gray)

        final_snake = snake.evolve(initial_snake, external_energy)

        self.snake_contour = final_snake
        self.initial_snake = initial_snake
        self.external_energy = external_energy

        return img, final_snake, external_energy
