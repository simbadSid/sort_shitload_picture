import cv2
import os

from image_compare import ImageComparer


class ImageCompareHistogramIntersection(ImageComparer):
    def parse_image(self, image_path: str, image_name: str):
        file_path = os.path.join(image_path, image_name)
        return cv2.imread(file_path)

    def compute_likelihood(self, image_path: str, image_name: str):
        """
        Computes the likelihood between the input image and the benchmark picture (set using self.set_benchmark_image).

        This function computes the likeness between two images by comparing their grayscale histograms
        using histogram intersection. It converts the images to grayscale, computes their histograms,
        and calculates the histogram intersection, providing a likelihood score.

        Returns:
        - likelihood: Histogram intersection likelihood value (float)

        The likelihood value ranges between 0 (completely dissimilar) and 1 (identical).
        A higher value indicates higher similarity between the images.
        """

        image2 = self.parse_image(image_path=image_path, image_name=image_name)

        # Convert images to grayscale
        gray_image1 = cv2.cvtColor(self.benchmark_image,    cv2.COLOR_BGR2GRAY)
        gray_image2 = cv2.cvtColor(image2,                  cv2.COLOR_BGR2GRAY)

        # Compute histogram intersection
        hist1 = cv2.calcHist([gray_image1], [0], None, [256], [0, 256])
        hist2 = cv2.calcHist([gray_image2], [0], None, [256], [0, 256])

        likelihood = cv2.compareHist(hist1, hist2, cv2.HISTCMP_INTERSECT)
        return likelihood
