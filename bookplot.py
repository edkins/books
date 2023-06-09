from bookshelf import Book
import matplotlib.pyplot as plt
import math

def plot_some(b: Book, x:str='x', y:str='y'):
    bs = b.bookshelf
    dims = [d for d in b.get_dims() if d not in {x,y}]
    a = []
    ixss = []
    for ixs in bs.enumerate_through(dims, debug_limit=3):
        a.append(b.get_ndarray_at_ixs(dims, ixs))
        ixss.append(ixs)

    xplots = math.floor(len(a)**0.5)
    yplots = math.ceil(len(a)/xplots)
    plt.subplots(yplots, xplots)
    for i, (arr,ixs) in enumerate(zip(a,ixss)):
        plt.subplot(yplots, xplots, i+1)
        plt.title(str(dict(zip(dims, ixs))))
        plt.imshow(arr)
    plt.show()