from __future__ import annotations
import numpy as np
from typing import Any

class _Dimension:
    def __init__(self, name: str, size: Book):
        self.name = name
        self.size = size

    def depends_on(self, other: str) -> bool:
        return self.size.has_dim(other)

class Book:
    def __init__(self, bookshelf: Bookshelf, outer_dims: tuple[str], inner_dims: tuple[str], values: dict[tuple[int],np.ndarray], is_mutable: bool):
        bookshelf._check_split_dims(outer_dims, inner_dims)

        # Error if we specify the wrong number of values
        expected_sizes = list(bookshelf._leaf_through(outer_dims, inner_dims))
        if len(expected_sizes) != len(values):
            raise ValueError(f"Expected {len(expected_sizes)} values, got {len(values)}")

        # Error if a value is the wrong size
        for ix,size in expected_sizes:
            if size != values[ix].shape:
                raise ValueError(f"Value at {ix} has incorrect size. Expected: {size}, got: {values[ix].shape}")

        self.bookshelf = bookshelf
        self.outer_dims = outer_dims
        self.inner_dims = inner_dims
        self.values = values
        self.is_mutable = is_mutable

    def has_dim(self, dim: str) -> bool:
        return dim in self.outer_dims or dim in self.inner_dims

    # There can be extra dims in the list given
    def defined_by_dims(self, dims: tuple[str]) -> bool:
        return set(self.outer_dims).issubset(dims) and set(self.inner_dims).issubset(dims)

    def is_scalar(self) -> bool:
        return len(self.outer_dims) == 0 and len(self.inner_dims) == 0

    def is_rectangular(self) -> bool:
        return len(self.outer_dims) == 0

    def get_scalar(self) -> Any:
        if not self.is_scalar():
            raise ValueError("Book is not scalar")
        return self.values[()]

    def get_ndarray(self) -> np.ndarray:
        if not self.is_rectangular():
            raise ValueError("Book size is not rectangular")
        return self.values[()]

    # Context can include irrelevant dimensions
    def get_scalar_at(self, **kwargs: int) -> Any:
        if not set(self.outer_dims).issubset(kwargs.keys()):
            raise ValueError(f"Context does not include all book outer dimensions. Context: {kwargs}, book: {self.outer_dims}")
        if not set(self.inner_dims).issubset(kwargs.keys()):
            raise ValueError(f"Context does not include all book inner dimensions. Context: {kwargs}, book: {self.inner_dims}")

        outer_ixs = tuple(kwargs[d] for d in self.outer_dims)
        inner_ixs = tuple(kwargs[d] for d in self.inner_dims)

        return self.values[outer_ixs][inner_ixs]

    def get_ndarray_at(self, **kwargs: int) -> np.ndarray:
        if not set(self.outer_dims).issubset(kwargs.keys()):
            raise ValueError(f"Context does not include all book outer dimensions. Context: {kwargs}, book: {self.outer_dims}")

        outer_ixs = tuple(kwargs[d] for d in self.outer_dims)
        inner_ixs = tuple(kwargs.get(d,slice(None)) for d in self.inner_dims)

        return self.values[outer_ixs][inner_ixs]

    def get_ndarray_at_ixs(self, dims: tuple[str], ixs: tuple[int]) -> np.ndarray:
        if len(dims) != len(ixs):
            raise ValueError(f"Expected {len(dims)} dimensions, got {len(ixs)}")
        return self.get_ndarray_at(**dict(zip(dims,ixs)))

    # Context can include irrelevant dimensions or missing dimensions
    #def get_book_at(self, **kwargs: int) -> Book:

    def __repr__(self) -> str:
        return f"Book({self.outer_dims}, {self.inner_dims}, ...)"

    def get_dims(self) -> tuple[str]:
        return self.outer_dims + self.inner_dims

class Bookshelf:
    def __init__(self):
        self.dimensions = []

    def dim(self, name: str):
        for dim in self.dimensions:
            if dim.name == name:
                return dim
        raise ValueError(f"Dimension {dim} not found")

    def has_dim(self, name: str) -> bool:
        for dim in self.dimensions:
            if dim.name == name:
                return True
        return False

    def has_dims(self, dims: tuple[str]) -> bool:
        return all(self.has_dim(d) for d in dims)

    def _check_split_dims(self, outer_dims: tuple[str], inner_dims: tuple[str]) -> None:
        # Error if any dimensions are included more than once
        if len(set(outer_dims)) != len(outer_dims):
            raise ValueError(f"Duplicate outer dimensions: {outer_dims}")
        if len(set(inner_dims)) != len(inner_dims):
            raise ValueError(f"Duplicate inner dimensions: {inner_dims}")
        if len(set(outer_dims).intersection(inner_dims)) > 0:
            raise ValueError(f"Outer and inner dimensions overlap: {outer_dims} and {inner_dims}")

        # Error if any dimensions are not in the bookshelf
        for dim in outer_dims:
            if not self.has_dim(dim):
                raise ValueError(f"Outer dimension {dim} not in bookshelf")
        for dim in inner_dims:
            if not self.has_dim(dim):
                raise ValueError(f"Inner dimension {dim} not in bookshelf")

        # Error if any inner dimension depends on a bookshelf dimension not listed in outer_dims
        for dim in inner_dims:
            if not self.dim(dim).size.defined_by_dims(outer_dims):
                raise ValueError(f"Inner dimension {dim} depends on dimensions not listed in outer_dims {outer_dims}")

        # Error if there's an outer dim that no inner dim depends on. (Such dimensions should be pushed inwards)
        for dim in outer_dims:
            if not any(self.dim(d).depends_on(dim) for d in inner_dims):
                raise ValueError(f"Outer dimension {dim} is not used by any inner dimension. Push it inwards.")

    def create_constant_uint32(self, value: int) -> Book:
        return self.import_ndarray((), np.array(value, dtype=np.uint32), is_mutable=False)

    def create_from_uint32s(self, dim: str, values: list[int]) -> Book:
        return self.import_ndarray((dim,), np.array(values, dtype=np.uint32), is_mutable=False)

    def import_ndarray(self, dims: tuple[str], value: np.ndarray, is_mutable: bool = False) -> Book:
        if len(dims) != len(value.shape):
            raise ValueError(f"Dimension count mismatch: {dims} vs {value.shape}")
        if not all(self.dim(d).size.is_scalar() for d in dims):
            raise ValueError(f"Dimensions must be scalar in import_ndarray: {dims}")
        return Book(self, (), dims, {():value}, is_mutable)

    def import_multi_ndarray(self, outer_dims: tuple[str], inner_dims: tuple[str], ixs: list[tuple[int]], values: list[np.ndarray], is_mutable: bool = False) -> Book:
        self._check_split_dims(outer_dims, inner_dims)
        if len(ixs) != len(values):
            raise ValueError(f"Index count mismatch: {len(ixs)} vs {len(values)}")
        return Book(self, outer_dims, inner_dims, {ixs[i]:values[i] for i in range(len(ixs))}, is_mutable)

    def import_ndarray_list(self, outer_dim: str, inner_dims: tuple[str], values: list[np.ndarray], is_mutable: bool = False) -> Book:
        self._check_split_dims((outer_dim,), inner_dims)
        return Book(self, (outer_dim,), inner_dims, {(i,):values[i] for i in range(len(values))}, is_mutable)

    def _consult_multi_list(self, dims: tuple[str], dtype, values: Any, context: dict[tuple[int],int]) -> Any:
        if len(dims) == 0:
            result = dtype(values)
            if result.shape != ():
                raise ValueError(f"Expected scalar but got {result.shape}")
            return result

        d0 = dims[0]
        size = self.dim(d0).size.get_scalar_at(**context)
        if len(values) != size:
            raise ValueError(f"Dimension {d0} has size {size} but got {len(values)} values")

        if d0 in context:
            ix = context[d0]
            return self._consult_multi_list(dims[1:], dtype, values[ix], context)
        else:
            return [self._consult_multi_list(dims[1:], dtype, values[i], context) for i in range(size)]

    def import_multi_list(self, dims: tuple[str], dtype, values: Any) -> Book:
        if len(set(dims)) != len(dims):
            raise ValueError(f"Duplicate dimensions: {dims}")
        if not self.has_dims(dims):
            raise ValueError(f"Not all dimensions are known: {dims}")

        # Error if any dimension depends on a later one
        for i,d0 in enumerate(dims):
            for d1 in dims[i+1:]:
                if self.dim(d0).depends_on(d1):
                    raise ValueError(f"Dimension {d0} depends on {d1} but {d1} comes later in the list")
        
        result = {}
        dims0, dims1 = self._separate_dimensions(dims)
        for ix,_ in self._leaf_through(dims0, dims1):
            if len(dims0) != len(ix):
                raise UnexpectedError("leaf_through returned wrong number of dimensions")
            context = dict(zip(dims0, ix))
            result[ix] = np.array(self._consult_multi_list(dims, dtype, values, context), dtype=dtype)
        return Book(self, dims0, dims1, result, is_mutable=False)

    # Return the list of dimensions which other given dimensions depend on
    # and then the rest.
    # The second list will be nonempty if the input is nonempty.
    def _separate_dimensions(self, dims: tuple[str]) -> tuple[tuple[str], tuple[str]]:
        result0 = []
        result1 = []
        for i,dim in enumerate(dims):
            if any(self.dim(d).depends_on(dim) for d in dims):
                result0.append(dim)
            else:
                result1.append(dim)
        result0 = tuple(result0)
        result1 = tuple(result1)
        self._check_split_dims(result0, result1)
        return result0, result1

    def enumerate_through(self, dims: tuple[str], context: dict[str,int]={}, debug_limit=None):
        if len(dims) == 0:
            yield ()
        else:
            d0 = dims[0]
            d1 = dims[1:]
            n = self.dim(d0).size.get_scalar_at(**context)
            if debug_limit is not None:
                n = min(n, debug_limit)
            for i in range(n):
                for stuff in self.enumerate_through(d1, context={**context, d0: i}, debug_limit=debug_limit):
                    yield (i,) + stuff

    def _leaf_through(self, dims0: tuple[str], dims1: tuple[str]):
        for ix in self.enumerate_through(dims0):
            context = {d: ix[i] for i,d in enumerate(dims0)}
            sizes = tuple(self.dim(d).size.get_scalar_at(**context) for d in dims1)
            yield ix, sizes

    def add_dimension(self, name: str, size: Book):
        if size.bookshelf != self:
            raise ValueError(f"Bookshelf mismatch for size of {name}")

        # Check name is unique
        for dim in self.dimensions:
            if dim.name == name:
                raise ValueError(f"Dimension name must be unique: {name}")

        # Check size book is immutable
        if size.is_mutable:
            raise ValueError(f"Size book must be immutable")

        self.dimensions.append(_Dimension(name, size))

    def add_dimension_const(self, name: str, size: int):
        self.add_dimension(name, self.create_constant_uint32(size))

    def add_dimension_dependent(self, name: str, deps: tuple[str], values: list[int]):
        self.add_dimension(name, self.import_multi_list(deps, np.uint32, values))
