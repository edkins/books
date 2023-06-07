from __future__ import annotations
import numpy as np
import pytest
from typing import Any

class _Dimension:
    def __init__(self, name: str, size: Book):
        self.name = name
        self.size = size

    def depends_on(self, other: str) -> bool:
        return other in self.size.dims

class Book:
    def __init__(self, bookshelf: Bookshelf, dims: tuple[str], values: dict[tuple[int],np.ndarray]):
        self.bookshelf = bookshelf
        self.dims = dims
        self.values = values

        expected_sizes = list(bookshelf.leaf_through(dims))
        if len(expected_sizes) != len(values):
            raise ValueError(f"Expected {len(expected_sizes)} values, got {len(values)}")

        for ix,size in expected_sizes:
            if size != values[ix].shape:
                raise ValueError(f"Value at {ix} has incorrect size. Expected: {size}, got: {values[ix].shape}")

    def is_constant(self) -> bool:
        return len(self.dims) == 0

    def get_constant(self) -> Any:
        if not self.is_constant():
            raise ValueError("Book is not constant")
        return self.values[()]

    def get_ndarray(self) -> np.ndarray:
        if not all(self.bookshelf.dim(d).size.is_constant() for d in self.dims):
            raise ValueError("Book size is not rectangular")
        return self.values[()]

    # Context can include irrelevant dimensions
    def get_scalar_at(self, context: dict[str,int]) -> Any:
        if not set(self.dims).issubset(context.keys()):
            raise ValueError(f"Context does not include all book dimensions. Context: {context}, book: {self.dims}")

        non_const_dims = [d for d in self.dims if not self.bookshelf.dim(d).size.is_constant()]
        non_const_ixs = tuple(context[d] for d in non_const_dims)

        const_dims = [d for d in self.dims if self.bookshelf.dim(d).size.is_constant()]
        const_ixs = tuple(context[d] for d in const_dims)

        return self.values[non_const_ixs][const_ixs]

class Bookshelf:
    def __init__(self):
        self.dimensions = []

    def dim(self, name: str):
        for dim in self.dimensions:
            if dim.name == name:
                return dim
        raise ValueError(f"Dimension {dim} not found")

    def check_dimension_order(self, dims: tuple[str]):
        index = -1
        for dim in dims:
            for i,d in enumerate(self.dimensions):
                if d.name == dim:
                    if i < index:
                        raise ValueError(f"Dimensions must be ordered: {dims}")
                    index = i
                    break
            else:
                raise ValueError(f"Dimension {dim} not found")

    def dim_index(self, name: str) -> int:
        for i,d in enumerate(self.dimensions):
            if d.name == name:
                return i
        raise ValueError(f"Dimension {name} not found")

    def insert_dimension_order(self, dims: tuple[str], extra: str) -> tuple[str]:
        self.check_dimension_order(dims)
        for dim in dims:
            if dim == extra:
                raise ValueError(f"Dimension {extra} already in {dims}")
        
        extra_index = self.dim_index(extra)
        result = []
        for i in range(len(dims)):
            dim = dims[i]
            if self.dim_index(dim) < extra_index:
                result.append(dim)
            else:
                result.append(extra)
                result += dims[i:]
                break
        else:
            result.append(extra)
        self.check_dimension_order(result)
        return tuple(result)

    def create_constant(self, value: np.ndarray) -> Book:
        if len(value.shape) != 0:
            raise ValueError("Value must be a scalar")
        return Book(self, (), {():value})

    def create_constant_uint32(self, value: int) -> Book:
        return self.create_constant(np.array(value, dtype=np.uint32))

    def create_from_list(self, dim: str, values: list[int]) -> Book:
        return self.import_ndarray((dim,), np.array(values, dtype=np.uint32))

    # def combine_books(self, dims: tuple[str], books: dict[tuple[int],Book]) -> Book:
    #     self.check_dimension_order(dims)

    #     if len(books) == 0:
    #         raise NotImplementedError()

    #     # Check all the books have the same dims and they're a subset of the ones given
    #     dims1 = book[0].dims
    #     for book in books:
    #         if book.dims != dims1:
    #             raise ValueError("Books have different dimensions")
    #         if not set(dims1).issubset(dims):
    #             raise ValueError("Books have dimensions not in dims")
        
    #     dims0 = tuple(d for d in dims if d not in dims1)
    #     for ix in self.enumerate_through(dims0):
    #         if ix not in books:
    #             raise ValueError(f"Missing book at {ix}")
        
    #     return Book(self, dims, books)


    def import_ndarray(self, dims: tuple[str], value: np.ndarray) -> Book:
        self.check_dimension_order(dims)
        if len(dims) != len(value.shape):
            raise ValueError(f"Dimension mismatch: {dims} vs {value.shape}")
        if not all(self.dim(d).size.is_constant() for d in dims):
            raise ValueError(f"Dimensions must be constant in import_ndarray: {dims}")
        return Book(self, dims, {():value})

    # def import_multi_array(self, dims: tuple[str], values: dict[tuple[int], np.ndarray]) -> Book:
    #     return Book(self, dims, values)

    def _consult_multi_list(self, dims: tuple[str], dtype, values: Any, context: dict[tuple[int],int]) -> Any:
        self.check_dimension_order(dims)
        if len(dims) == 0:
            result = dtype(values)
            if result.shape != ():
                raise ValueError(f"Expected scalar but got {result.shape}")
            return result

        d0 = dims[0]
        size = self.dim(d0).size.get_scalar_at(context)
        if len(values) != size:
            raise ValueError(f"Dimension {d0} has size {size} but got {len(values)} values")

        if d0 in context:
            ix = context[d0]
            return self._consult_multi_list(dims[1:], dtype, values[ix], context)
        else:
            return [self._consult_multi_list(dims[1:], dtype, values[i], context) for i in range(size)]

    def import_multi_list(self, dims: tuple[str], dtype, values: Any) -> Book:
        self.check_dimension_order(dims)
        result = {}
        dims0, _ = self.separate_dimensions(dims)
        for ix,_ in self.leaf_through(dims):
            context = dict(zip(dims0, ix))
            result[ix] = np.array(self._consult_multi_list(dims, dtype, values, context), dtype=dtype)
        return Book(self, dims, result)

    # Return the list of dimensions which other given dimensions depend on
    # and then the rest.
    # The second list will be nonempty if the input is nonempty.
    def separate_dimensions(self, dims: tuple[str]) -> tuple[tuple[str], tuple[str]]:
        self.check_dimension_order(dims)
        result0 = []
        result1 = []
        for i,dim in enumerate(dims):
            if any(self.dim(d).depends_on(dim) for d in dims[i+1:]):
                result0.append(dim)
            else:
                result1.append(dim)
        return tuple(result0), tuple(result1)

    def enumerate_through(self, dims: tuple[str], context: dict[str,int]={}):
        self.check_dimension_order(dims)
        if len(dims) == 0:
            yield ()
        else:
            d0 = dims[0]
            d1 = dims[1:]
            for i in range(self.dim(d0).size.get_scalar_at(context)):
                for stuff in self.enumerate_through(d1, context={**context, d0: i}):
                    yield (i,) + stuff

    def leaf_through(self, dims: tuple[str]):
        self.check_dimension_order(dims)

        dims0, dims1 = self.separate_dimensions(dims)
        for ix in self.enumerate_through(dims0):
            context = {d: ix[i] for i,d in enumerate(dims0)}
            sizes = tuple(self.dim(d).size.get_scalar_at(context) for d in dims1)
            yield ix, sizes

    def add_dimension(self, name: str, size: Book):
        if size.bookshelf != self:
            raise ValueError(f"Bookshelf mismatch for size of {name}")

        # Check name is unique
        for dim in self.dimensions:
            if dim.name == name:
                raise ValueError(f"Dimension name must be unique: {name}")

        self.dimensions.append(_Dimension(name, size))

    def add_dimension_const(self, name: str, size: int):
        self.add_dimension(name, self.create_constant_uint32(size))



def test_import_ndarray_constant():
    bs = Bookshelf()
    bs.add_dimension_const("x", 2)
    bs.add_dimension_const("y", 3)
    bs.add_dimension_const("z", 4)
    dims = ("x", "y", "z")
    value = np.zeros((2, 3, 4))
    book = bs.import_ndarray(dims, value)
    assert book.dims == dims
    assert list(bs.leaf_through(dims)) == [((),(2, 3, 4))]
    assert list(bs.leaf_through(("x",))) == [((),(2,))]
    assert not book.is_constant()

def test_add_dimension_constant():
    bs = Bookshelf()
    bs.add_dimension_const("x", 2)
    bs.add_dimension_const("y", 3)
    bs.add_dimension_const("z", 4)
    dims = ("x", "y", "z")
    value = np.zeros((2, 3, 4))
    book = bs.import_ndarray(dims, value)
    bs.add_dimension("t", bs.create_constant_uint32(1))
    assert [dim.name for dim in bs.dimensions] == ["x", "y", "z", "t"]
    assert bs.dim("t").size.get_constant() == 1

def test_empty_bookshelf_dims():
    bs = Bookshelf()
    book = bs.create_constant_uint32(123)
    assert list(bs.leaf_through(())) == [((),())]
    assert book.is_constant()
    assert book.get_constant() == 123

def test_swapped_dimension_error():
    bs = Bookshelf()
    bs.add_dimension_const("x", 2)
    bs.add_dimension_const("y", 3)
    bs.add_dimension_const("z", 4)
    dims = ("x", "y", "z")
    value = np.zeros((2, 3, 4))
    with pytest.raises(ValueError):
        bs.import_ndarray(("y", "x", "z"), value)

def test_dimension_insert():
    bs = Bookshelf()
    bs.add_dimension_const("x", 2)
    bs.add_dimension_const("y", 3)
    bs.add_dimension_const("z", 4)
    assert bs.insert_dimension_order(("x","y"), "z") == ("x", "y", "z")
    assert bs.insert_dimension_order(("x","z"), "y") == ("x", "y", "z")
    assert bs.insert_dimension_order(("y","z"), "x") == ("x", "y", "z")

def test_dimension_already_exists():
    bs = Bookshelf()
    bs.add_dimension_const("x", 2)
    bs.add_dimension_const("y", 3)
    bs.add_dimension_const("z", 4)
    with pytest.raises(ValueError):
        bs.add_dimension_const("x", 5)

def test_create_from_list_wrong_size():
    bs = Bookshelf()
    bs.add_dimension_const("x", 2)
    with pytest.raises(ValueError):
        bs.create_from_list("x", [1, 2, 3])

def test_import_multi_list():
    bs = Bookshelf()
    bs.add_dimension_const("x", 2)
    ny = bs.create_from_list("x", [4, 3])
    bs.add_dimension("y", ny)
    b = bs.import_multi_list(("x", "y"), np.float32, [[1, 2, 3, 4], [5, 6, 7]])

def test_import_multi_list_wrong_ysize():
    bs = Bookshelf()
    bs.add_dimension_const("x", 2)
    ny = bs.create_from_list("x", [4, 3])
    bs.add_dimension("y", ny)
    with pytest.raises(ValueError):
        b = bs.import_multi_list(("x", "y"), np.float32, [[1, 2, 3, 4], [5, 6, 7, 8]])

def test_import_multi_list_wrong_xsize():
    bs = Bookshelf()
    bs.add_dimension_const("x", 2)
    ny = bs.create_from_list("x", [4, 3])
    bs.add_dimension("y", ny)
    with pytest.raises(ValueError):
        b = bs.import_multi_list(("x", "y"), np.float32, [[1, 2, 3, 4], [5, 6, 7], [8,9]])

def test_import_multi_list_too_much_rank():
    bs = Bookshelf()
    bs.add_dimension_const("x", 2)
    ny = bs.create_from_list("x", [4, 3])
    bs.add_dimension("y", ny)
    with pytest.raises(ValueError):
        b = bs.import_multi_list(("x", "y"), np.float32, [[[1], [2], [3], [4]], [[5], [6], [7]]])

def test_import_multi_empty_x():
    bs = Bookshelf()
    bs.add_dimension_const("x", 0)
    ny = bs.create_from_list("x", [])
    bs.add_dimension("y", ny)
    b = bs.import_multi_list(("x", "y"), np.float32, [])

def test_import_multi_empty_y():
    bs = Bookshelf()
    bs.add_dimension_const("x", 2)
    ny = bs.create_from_list("x", [0, 0])
    bs.add_dimension("y", ny)
    b = bs.import_multi_list(("x", "y"), np.float32, [[], []])

def test_import_multi_rectangular():
    bs = Bookshelf()
    bs.add_dimension_const("x", 2)
    bs.add_dimension_const("y", 4)
    b = bs.import_multi_list(("x", "y"), np.float32, [[1, 2, 3, 4], [5, 6, 7, 8]])