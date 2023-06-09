from bookshelf import Bookshelf
import booktorch

def main():
    bs = Bookshelf()
    xs, ys = booktorch.data(bs, 'mnist')
    print(xs)
    print(ys)

if __name__ == '__main__':
    main()
