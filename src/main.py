import os
import shutil
from typing import Dict, List

from image_compare import ImageComparer
from image_compare_ORB import ImageCompareORB

LIKELIHOOD_THRESHOLD    = 0.3

IMAGE_EXTENSION_LIST    = [e.lower() for e in [
    ".jpg",
    ".jpeg",
    ".png",
    ".gif",
#    '.heic'
]]

# Directories
DIR_DATA                = os.path.join('./', 'data')
DIR_RESULT              = os.path.join('./', 'result')
DIR_ALL                 = os.path.join(DIR_DATA, 'dir_all')
DIR_GROUPS              = os.path.join(DIR_DATA, 'dir_groups')
DIR_GROUPS_RESULT       = os.path.join(DIR_RESULT, 'dir_groups_result')
DIR_GROUPS_UNMATCHED    = os.path.join(DIR_RESULT, 'dir_group_unmatched')


def is_image(path: str, file: str) -> bool:
    file_path = os.path.join(path, file)
    return os.path.isfile(file_path) and os.path.splitext(file_path)[1].lower() in IMAGE_EXTENSION_LIST


def create_or_empty_dir(dir_path):
    # Check if directory exists
    if os.path.exists(dir_path):
        shutil.rmtree(dir_path, ignore_errors=False)

    os.makedirs(dir_path, exist_ok=False)

    return


def init_result_directories() -> Dict[str, List[str]]:
    """
    Create result directory for each group and for unmatched images

    :returns
        A map of each group with list of absolute path of each image file
    """

    # Create result directory for unmatched images
    create_or_empty_dir(DIR_GROUPS_UNMATCHED)

    result = {}
    # Iterate through each group directory
    for group_dir in os.listdir(DIR_GROUPS):
        group_path = os.path.join(DIR_GROUPS, group_dir)
        if not os.path.isdir(group_path):
            print(f'Unknown file type in group dir: {group_path}')
            continue

        # Create the directory corresponding to the result group
        result_group_path = os.path.join(DIR_GROUPS_RESULT, group_dir)
        create_or_empty_dir(result_group_path)

        # Copy all images from the original group directory to the result directory
        group_images = []
        for file in os.listdir(group_path):
            if not is_image(path=group_path, file=file):
                print(f'Unknown file type in group {group_dir}: {file}')
            else:
                group_images.append(file)
                shutil.copy2(os.path.join(group_path, file), result_group_path)

        # Check that two groups don't have the same key
        assert group_dir not in result

        result[group_dir] = group_images

    return result


def compare_with_all_groups(file_all: str):
    if not is_image(path=DIR_ALL, file=file_all):
        print(f'Unknown file type in images to sort: {file_all}')
        return

    image_comparer.set_benchmark_image(image_path=DIR_ALL, image_name=file_all)

    best_likelihood = 0
    matched_group = None

    # Iterate through each group directory
    for group, images_in_group in group_image_list.items():
        group_path = os.path.join(DIR_GROUPS, group)
        for img_file in images_in_group:
            likelihood = image_comparer.compute_likelihood(image_path=group_path, image_name=img_file)

            if likelihood > best_likelihood:
                best_likelihood = likelihood
                matched_group = group

    # Check if likelihood is higher than 30%
    if best_likelihood > LIKELIHOOD_THRESHOLD:
        shutil.copy2(os.path.join(DIR_ALL, file_all), os.path.join(DIR_GROUPS_RESULT, matched_group))
    else:
        shutil.copy2(os.path.join(DIR_ALL, file_all), DIR_GROUPS_UNMATCHED)


if __name__ == "__main__":
    # image_comparer: ImageComparer = ImageCompareHistogramIntersection()
    image_comparer: ImageComparer = ImageCompareORB()

    # Create result directory for each group and for unmatched images
    group_image_list = init_result_directories()

    # Iterate through images in DIR_ALL and find best match in the group
    for fic in os.listdir(DIR_ALL):
        compare_with_all_groups(file_all=fic)
