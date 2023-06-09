from bookshelf import Bookshelf, Book
import numpy as np
from torchvision.datasets import MNIST

def _dataset_to_large_ndarrays(dataset, xshape: tuple[int]) -> np.ndarray:
    n = len(dataset)
    X = np.empty((n, *xshape), dtype=np.float32)
    y = np.empty(n, dtype=np.int64)
    for i, (x, y_) in enumerate(dataset):
        X[i] = x
        y[i] = y_
    return X, y

def _data_mnist(bs, data_dir:str) -> tuple[Book,Book]:
    train = MNIST(data_dir, train=True, download=True)
    test = MNIST(data_dir, train=False, download=True)
    bs.add_dimension_const('split', 2)
    bs.add_dimension_dependent('record', ('split',), [len(train), len(test)])
    w = 28
    h = 28
    bs.add_dimension_const('x', w)
    bs.add_dimension_const('y', h)
    train_X, train_y = _dataset_to_large_ndarrays(train, (w,h))
    test_X, test_y = _dataset_to_large_ndarrays(test, (w,h))
    book_X = bs.import_ndarray_list('split', ('record', 'x', 'y'), [train_X, test_X])
    book_y = bs.import_ndarray_list('split', ('record',), [train_y, test_y])
    return book_X, book_y

def data(bs: Bookshelf, name: str, data_dir:str = 'data') -> tuple[Book,Book]:
    if name == 'mnist':
        return _data_mnist(bs, data_dir)
    else:
        raise ValueError(f'Unknown dataset: {name}')