import os
import glob
from skimage import io, color, transform
import urllib.request
import tarfile
import numpy as np


def download_extract_data():
    url = "http://vis-www.cs.umass.edu/lfw/lfw-funneled.tgz"
    if not os.path.isfile("lfw-funneled.tgz"):
        print("Downloading: " + url)
        urllib.request.urlretrieve(url, "lfw-funneled.tgz")
    if not os.path.isdir("lfw-funneled"):
        tar = tarfile.open("lfw-funneled.tgz", "r:gz")
        tar.extractall()
        tar.close()


def image_crop_to_square(image):  # expects greyscale image
    height, width = image.shape
    if height > width:
        offset = int(round(((height-width)/2)))
        image = image[offset:height-offset, 0:width]
    else:
        offset = int(round(((width-height)/2)))
        image = image[0:height, offset:width-offset]
    return image


def image_crop_face(image):  # expects greyscale image
    height, width = image.shape
    if not height == width:
        print("Image is not suqare!")
    else:
        fourth = int(round(height/4))
        end = fourth*3
        return image[fourth:end, fourth:end]


def image_scale_to_32x32(image):  # expects greyscale image
    return transform.resize(image, (32, 32))


def stack_image(image):
    return np.reshape(image, -1)


def main():
    array = []
    persons = {}

    download_extract_data()
    os.chdir("lfw_funneled")
    dirs = glob.glob("*/")      # List all directories

    for current_dir in dirs:    # Get all filenames in all directories
        person_images = []
        os.chdir(current_dir)
        print("Now in dir: " + current_dir)

        # Read in all except one image in each dir
        images = glob.glob("*.jpg")
        images.pop(0)
        #print(*images, sep='\n')
        for img in images:
            image = io.imread(img)
            image = color.rgb2gray(image)
            cropped = image_crop_to_square(image)
            cropped = image_crop_face(cropped)
            cropped = image_scale_to_32x32(cropped)

            person_images.append(cropped)

            array.append(stack_image(cropped))

        persons[current_dir] = person_images
        os.chdir("../")
    print(persons.__sizeof__())
    to_nd_array = np.array(array)


if __name__ == "__main__":
    main()
