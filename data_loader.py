import numpy as np
from sklearn.datasets import fetch_mldata
from sklearn.model_selection import train_test_split

class MNISTLoader:

    class Container:

        def __init__(self, images, labels, shuffle=True):
            self.images = images
            self.labels = labels
            self.size = self.images.shape[0]
            self.shuffle = shuffle

            self._cur = 0

        def next_batch(self, batch_size):
            if self._cur + batch_size > self.size:
                if self.shuffle:
                    self._shuffle()
                self._cur = 0

            self._cur += batch_size
            return (
                    self.images[self._cur - batch_size:self._cur],
                    self.labels[self._cur - batch_size:self._cur])

        def _shuffle(self):
            perm = np.random.permutation(self.size)
            self.images = self.images[perm]
            self.labels = self.labels[perm]

    def __init__(self, dataset_dir='dataset', shuffle=True):
        mnist = fetch_mldata('MNIST original', data_home=dataset_dir)
        X_train, X_test, y_train, y_test = train_test_split(
                mnist.data, mnist.target, test_size=20000, shuffle=False)
        self.train = self.Container(X_train, y_train, shuffle=True)
        self.test = self.Container(X_test, y_test, shuffle=False)
