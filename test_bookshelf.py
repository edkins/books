from bookshelf import Bookshelf
import numpy as np
import pytest


def test_import_ndarray_constant():
    bs = Bookshelf()
    bs.add_dimension_const("x", 2)
    bs.add_dimension_const("y", 3)
    bs.add_dimension_const("z", 4)
    dims = ("x", "y", "z")
    value = np.zeros((2, 3, 4))
    book = bs.import_ndarray(dims, value, False)
    assert book.inner_dims == dims
    assert list(bs._leaf_through((),dims)) == [((),(2, 3, 4))]
    assert list(bs._leaf_through((),("x",))) == [((),(2,))]
    assert not book.is_scalar()

def test_add_dimension_constant():
    bs = Bookshelf()
    bs.add_dimension_const("x", 2)
    bs.add_dimension_const("y", 3)
    bs.add_dimension_const("z", 4)
    dims = ("x", "y", "z")
    value = np.zeros((2, 3, 4))
    book = bs.import_ndarray(dims, value, False)
    bs.add_dimension("t", bs.create_constant_uint32(1))
    assert [dim.name for dim in bs.dimensions] == ["x", "y", "z", "t"]
    assert bs.dim("t").size.get_scalar() == 1

def test_empty_bookshelf_dims():
    bs = Bookshelf()
    book = bs.create_constant_uint32(123)
    assert list(bs._leaf_through((),())) == [((),())]
    assert book.is_scalar()
    assert book.get_scalar() == 123

def test_swapped_dimension():
    bs = Bookshelf()
    bs.add_dimension_const("x", 2)
    bs.add_dimension_const("y", 3)
    bs.add_dimension_const("z", 4)
    dims = ("x", "y", "z")
    value = np.zeros((3, 2, 4))
    book = bs.import_ndarray(("y", "x", "z"), value, False)
    assert book.inner_dims == ("y", "x", "z")

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
        bs.create_from_uint32s("x", [1, 2, 3])

def test_import_multi_list():
    bs = Bookshelf()
    bs.add_dimension_const("x", 2)
    ny = bs.create_from_uint32s("x", [4, 3])
    bs.add_dimension("y", ny)
    b = bs.import_multi_list(("x", "y"), np.float32, [[1, 2, 3, 4], [5, 6, 7]])

def test_import_multi_list_wrong_ysize():
    bs = Bookshelf()
    bs.add_dimension_const("x", 2)
    ny = bs.create_from_uint32s("x", [4, 3])
    bs.add_dimension("y", ny)
    with pytest.raises(ValueError):
        b = bs.import_multi_list(("x", "y"), np.float32, [[1, 2, 3, 4], [5, 6, 7, 8]])

def test_import_multi_list_wrong_xsize():
    bs = Bookshelf()
    bs.add_dimension_const("x", 2)
    ny = bs.create_from_uint32s("x", [4, 3])
    bs.add_dimension("y", ny)
    with pytest.raises(ValueError):
        b = bs.import_multi_list(("x", "y"), np.float32, [[1, 2, 3, 4], [5, 6, 7], [8,9]])

def test_import_multi_list_too_much_rank():
    bs = Bookshelf()
    bs.add_dimension_const("x", 2)
    ny = bs.create_from_uint32s("x", [4, 3])
    bs.add_dimension("y", ny)
    with pytest.raises(ValueError):
        b = bs.import_multi_list(("x", "y"), np.float32, [[[1], [2], [3], [4]], [[5], [6], [7]]])

def test_import_multi_empty_x():
    bs = Bookshelf()
    bs.add_dimension_const("x", 0)
    ny = bs.create_from_uint32s("x", [])
    bs.add_dimension("y", ny)
    b = bs.import_multi_list(("x", "y"), np.float32, [])

def test_import_multi_empty_y():
    bs = Bookshelf()
    bs.add_dimension_const("x", 2)
    ny = bs.create_from_uint32s("x", [0, 0])
    bs.add_dimension("y", ny)
    b = bs.import_multi_list(("x", "y"), np.float32, [[], []])

def test_import_multi_rectangular():
    bs = Bookshelf()
    bs.add_dimension_const("x", 2)
    bs.add_dimension_const("y", 4)
    b = bs.import_multi_list(("x", "y"), np.float32, [[1, 2, 3, 4], [5, 6, 7, 8]])

def test_import_multi_rectangular_flipped():
    bs = Bookshelf()
    bs.add_dimension_const("x", 2)
    bs.add_dimension_const("y", 4)
    b = bs.import_multi_list(("y","x"), np.float32, [[1, 2], [3,4], [5,6], [7,8]])

def test_import_multi_rectangular_yonly():
    bs = Bookshelf()
    bs.add_dimension_const("x", 2)
    bs.add_dimension_const("y", 4)
    b = bs.import_multi_list(("y",), np.float32, [1, 2, 3, 4])

def test_import_multi_list_pseudo_rectangular():
    bs = Bookshelf()
    bs.add_dimension_const("x", 2)
    ny = bs.create_from_uint32s("x", [4, 4])
    bs.add_dimension("y", ny)
    b = bs.import_multi_list(("x", "y"), np.float32, [[1, 2, 3, 4], [5, 6, 7, 8]])

def test_import_multi_list_pseudo_rectangular_flipped():
    bs = Bookshelf()
    bs.add_dimension_const("x", 2)
    ny = bs.create_from_uint32s("x", [4, 4])
    bs.add_dimension("y", ny)
    with pytest.raises(ValueError):
        b = bs.import_multi_list(("y","x"), np.float32, [[1, 2], [3,4], [5,6], [7,8]])

def test_import_multi_list_pseudo_rectangular_yonly():
    bs = Bookshelf()
    bs.add_dimension_const("x", 2)
    ny = bs.create_from_uint32s("x", [4, 4])
    bs.add_dimension("y", ny)
    with pytest.raises(ValueError):
        b = bs.import_multi_list(("y",), np.float32, [1,2,3,4])

def test_dependent_dim_from_ndarray():
    bs = Bookshelf()
    bs.add_dimension_const("x", 2)
    a = np.zeros((2,))
    a[0] = 3
    a[1] = 6
    ny = bs.import_ndarray("x", a, False)
    bs.add_dimension("y", ny)

def test_dependent_dim_from_ndarray_mutable():
    bs = Bookshelf()
    bs.add_dimension_const("x", 2)
    a = np.zeros((2,))
    a[0] = 3
    a[1] = 6
    ny = bs.import_ndarray("x", a, True)
    with pytest.raises(ValueError):
        bs.add_dimension("y", ny)

def test_get_ndarray_at():
    bs = Bookshelf()
    bs.add_dimension_const("x", 2)
    bs.add_dimension_dependent("y", "x", [3, 4])
    b = bs.import_multi_list(("x", "y"), np.float32, [[1, 2, 3], [4, 5, 6, 7]])
    assert np.array_equal(b.get_ndarray_at(x=0), np.array([1,2,3]))
    assert np.array_equal(b.get_ndarray_at(x=1), np.array([4,5,6,7]))
    assert b.get_ndarray_at(x=0, y=2) == 3

def test_get_ndarray_rectangular():
    bs = Bookshelf()
    bs.add_dimension_const("x", 2)
    bs.add_dimension_const("y", 3)
    b = bs.import_multi_list(("x", "y"), np.float32, [[1, 2, 3], [4, 5, 6]])
    assert np.array_equal(b.get_ndarray_at(x=0), np.array([1,2,3]))
    assert np.array_equal(b.get_ndarray_at(x=1), np.array([4,5,6]))
    assert b.get_ndarray_at(x=0, y=2) == 3
    assert np.array_equal(b.get_ndarray_at(y=2), np.array([3,6]))
    assert np.array_equal(b.get_ndarray_at(), np.array([[1,2,3], [4,5,6]]))

def test_get_ndarray_at_fail():
    bs = Bookshelf()
    bs.add_dimension_const("x", 2)
    bs.add_dimension_dependent("y", "x", [3, 4])
    b = bs.import_multi_list(("x", "y"), np.float32, [[1, 2, 3], [4, 5, 6, 7]])
    with pytest.raises(KeyError):
        b.get_ndarray_at(x=2)
    with pytest.raises(ValueError):
        b.get_ndarray_at(y=1)
    with pytest.raises(ValueError):
        b.get_ndarray_at()
