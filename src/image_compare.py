from abc import ABC, abstractmethod


class ImageComparer(ABC):
    benchmark_image = None

    @abstractmethod
    def parse_image(self, image_path: str, image_name: str):
        pass

    def set_benchmark_image(self, image_path: str, image_name: str):
        self.benchmark_image = self.parse_image(image_path=image_path, image_name=image_name)

    @abstractmethod
    def compute_likelihood(self, image_path: str, image_name: str):
        pass
