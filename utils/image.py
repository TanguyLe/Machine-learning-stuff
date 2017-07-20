from PIL import Image
import numpy as np


class PIL_Image:
    def __init__(self, array=None, path=None):

        if array is not None:
            if type(array) is not np.ndarray:
                raise TypeError("First argument is not a numpy.ndarray, use path= or from_array()")

            try:
                self.img = Image.fromarray(array)
            except TypeError as e:
                print("Not a proper numpy array :")
                print(e)
        elif path is not None:
            try:
                self.img = Image.open(path)
            except FileNotFoundError as e:
                print("Not a proper path :")
                print(e)
        else:
            raise TypeError("Nothing specified to create the image from")

    @classmethod
    def from_array(cls, array):
        return cls(array=array)

    @classmethod
    def from_file(cls, path):
        return cls(path=path)

    def show(self):
        self.img.show()

    def get_array(self):
        return self.img.getdata()

