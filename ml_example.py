from bookshelf import Bookshelf
import booktorch
import bookplot

def main():
    bs = Bookshelf()
    xs, ys = booktorch.data(bs, 'mnist')
    print(xs)
    print(ys)
    bookplot.plot_some(xs, x='x', y='y')

if __name__ == '__main__':
    main()
