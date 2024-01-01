import cv2
import os

from image_compare import ImageComparer


class ImageCompareORB(ImageComparer):
    def parse_image(self, image_path: str, image_name: str):
        file_path = os.path.join(image_path, image_name)
        return cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)

    def compute_likelihood(self, image_path: str, image_name: str):
        """
        Computes the likelihood between the input image and the benchmark picture (set using self.set_benchmark_image).
        Compares two images using ORB feature matching.

        Returns:
        - similarity_score: Similarity score based on number of matches (int) using ORB feature matching
        """
        # Read images
        img2 = self.parse_image(image_path=image_path, image_name=image_name)

        # Initialize ORB detector
        orb = cv2.ORB_create()

        # Find key points and descriptors
        kp1, des1 = orb.detectAndCompute(self.benchmark_image,  None)
        kp2, des2 = orb.detectAndCompute(img2,                  None)

        # Create Brute Force matcher
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        # Match descriptors
        matches = bf.match(des1, des2)

        # Sort matches based on distance
        matches = sorted(matches, key=lambda x: x.distance)

        # Draw first 10 matches (just for visualization)
        img_matches = cv2.drawMatches(
            self.benchmark_image, kp1, img2, kp2, matches[:10], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
        )

        # Calculate similarity score based on number of matches
        similarity_score = len(matches)

        # Show matches (visualization purposes)
        """
        cv2.imshow('Matches', img_matches)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        """

        return similarity_score
