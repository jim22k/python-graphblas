from warnings import warn as _warn

import numpy as _np

from . import backend as _backend
from .core.matrix import Matrix as _Matrix
from .core.utils import normalize_values as _normalize_values
from .core.utils import output_type as _output_type
from .core.vector import Vector as _Vector
from .dtypes import lookup_dtype as _lookup_dtype
from .exceptions import GraphblasException as _GraphblasException


def draw(m):  # pragma: no cover (deprecated)
    """Draw a square adjacency Matrix as a graph.

    Requires `networkx <https://networkx.org/>`_ and
    `matplotlib <https://matplotlib.org/>`_ to be installed.

    Example output:

    .. image:: /_static/img/draw-example.png
    """
    from . import viz

    _warn(
        "`graphblas.io.draw` is deprecated; it has been moved to `graphblas.viz.draw`",
        DeprecationWarning,
        stacklevel=2,
    )
    viz.draw(m)


def from_networkx(G, nodelist=None, dtype=None, weight="weight", name=None):
    """Create a square adjacency Matrix from a networkx Graph.

    Parameters
    ----------
    G : nx.Graph
        Graph to convert
    nodelist : list, optional
        List of nodes in the nx.Graph. If not provided, all nodes will be used.
    dtype :
        Data type
    weight : str, default="weight"
        Weight attribute
    name : str, optional
        Name of resulting Matrix

    Returns
    -------
    :class:`~graphblas.Matrix`
    """
    import networkx as nx

    if dtype is not None:
        dtype = _lookup_dtype(dtype).np_type
    A = nx.to_scipy_sparse_array(G, nodelist=nodelist, dtype=dtype, weight=weight)
    return from_scipy_sparse(A, name=name)


def from_numpy(m):  # pragma: no cover (deprecated)
    """Create a sparse Vector or Matrix from a dense numpy array.

    .. deprecated:: 2023.2.0
        `from_numpy` will be removed in a future release.
        Use `Vector.from_dense` or `Matrix.from_dense` instead.
        Will be removed in version 2023.10.0 or later

    A value of 0 is considered as "missing".

    - m.ndim == 1 returns a `Vector`
    - m.ndim == 2 returns a `Matrix`
    - m.ndim > 2 raises an error

    dtype is inferred from m.dtype

    Parameters
    ----------
    m : np.ndarray
        Input array

    See Also
    --------
    Matrix.from_dense
    Vector.from_dense
    from_scipy_sparse

    Returns
    -------
    Vector or Matrix
    """
    _warn(
        "`graphblas.io.from_numpy` is deprecated; "
        "use `Matrix.from_dense` and `Vector.from_dense` instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    if m.ndim > 2:
        raise _GraphblasException("m.ndim must be <= 2")

    try:
        from scipy.sparse import coo_array, csr_array
    except ImportError:  # pragma: no cover (import)
        raise ImportError("scipy is required to import from numpy") from None

    if m.ndim == 1:
        A = csr_array(m)
        _, size = A.shape
        dtype = _lookup_dtype(m.dtype)
        return _Vector.from_coo(A.indices, A.data, size=size, dtype=dtype)
    A = coo_array(m)
    return from_scipy_sparse(A)


def from_scipy_sparse(A, *, dup_op=None, name=None):
    """Create a Matrix from a scipy.sparse array or matrix.

    Input data in "csr" or "csc" format will be efficient when importing with SuiteSparse:GraphBLAS.

    Parameters
    ----------
    A : scipy.sparse
        Scipy sparse array or matrix
    dup_op : BinaryOp, optional
        Aggregation function for formats that allow duplicate entries (e.g. coo)
    name : str, optional
        Name of resulting Matrix

    Returns
    -------
    :class:`~graphblas.Matrix`
    """
    nrows, ncols = A.shape
    dtype = _lookup_dtype(A.dtype)
    if A.nnz == 0:
        return _Matrix(dtype, nrows=nrows, ncols=ncols, name=name)
    if _backend == "suitesparse" and A.format in {"csr", "csc"}:
        data = A.data
        is_iso = (data[[0]] == data).all()
        if is_iso:
            data = data[[0]]
        if A.format == "csr":
            return _Matrix.ss.import_csr(
                nrows=nrows,
                ncols=ncols,
                indptr=A.indptr,
                col_indices=A.indices,
                values=data,
                is_iso=is_iso,
                sorted_cols=getattr(A, "_has_sorted_indices", False),
                name=name,
            )
        return _Matrix.ss.import_csc(
            nrows=nrows,
            ncols=ncols,
            indptr=A.indptr,
            row_indices=A.indices,
            values=data,
            is_iso=is_iso,
            sorted_rows=getattr(A, "_has_sorted_indices", False),
            name=name,
        )
    if A.format == "csr":
        return _Matrix.from_csr(A.indptr, A.indices, A.data, ncols=ncols, name=name)
    if A.format == "csc":
        return _Matrix.from_csc(A.indptr, A.indices, A.data, nrows=nrows, name=name)
    if A.format != "coo":
        A = A.tocoo()
    return _Matrix.from_coo(
        A.row, A.col, A.data, nrows=nrows, ncols=ncols, dtype=dtype, dup_op=dup_op, name=name
    )


def from_awkward(A, *, name=None):
    """Create a Matrix or Vector from an Awkward Array.

    The Awkward Array must have top-level parameters: format, shape

    The Awkward Array must have top-level attributes based on format:
    - vec/csr/csc: values, indices
    - hypercsr/hypercsc: values, indices, offset_labels

    Parameters
    ----------
    A : awkward.Array
        Awkward Array with values and indices
    name : str, optional
        Name of resulting Matrix or Vector

    Returns
    -------
    Vector or Matrix
    """
    params = A.layout.parameters
    if missing := {"format", "shape"} - params.keys():
        raise ValueError(f"Missing parameters: {missing}")
    format = params["format"]
    shape = params["shape"]

    if len(shape) == 1:
        if format != "vec":
            raise ValueError(f"Invalid format for Vector: {format}")
        return _Vector.from_coo(
            A.indices.layout.data, A.values.layout.data, size=shape[0], name=name
        )
    nrows, ncols = shape
    values = A.values.layout.content.data
    indptr = A.values.layout.offsets.data
    if format == "csr":
        cols = A.indices.layout.content.data
        return _Matrix.from_csr(indptr, cols, values, ncols=ncols, name=name)
    if format == "csc":
        rows = A.indices.layout.content.data
        return _Matrix.from_csc(indptr, rows, values, nrows=nrows, name=name)
    if format == "hypercsr":
        rows = A.offset_labels.layout.data
        cols = A.indices.layout.content.data
        return _Matrix.from_dcsr(rows, indptr, cols, values, nrows=nrows, ncols=ncols, name=name)
    if format == "hypercsc":
        cols = A.offset_labels.layout.data
        rows = A.indices.layout.content.data
        return _Matrix.from_dcsc(cols, indptr, rows, values, nrows=nrows, ncols=ncols, name=name)
    raise ValueError(f"Invalid format for Matrix: {format}")


def from_pydata_sparse(s, *, dup_op=None, name=None):
    """Create a Vector or a Matrix from a pydata.sparse array or matrix.

    Input data in "gcxs" format will be efficient when importing with SuiteSparse:GraphBLAS.

    Parameters
    ----------
    s : sparse
        PyData sparse array or matrix (see https://sparse.pydata.org)
    dup_op : BinaryOp, optional
        Aggregation function for formats that allow duplicate entries (e.g. coo)
    name : str, optional
        Name of resulting Matrix

    Returns
    -------
    :class:`~graphblas.Vector`
    :class:`~graphblas.Matrix`
    """
    try:
        import sparse
    except ImportError:  # pragma: no cover (import)
        raise ImportError("sparse is required to import from pydata sparse") from None
    if not isinstance(s, sparse.SparseArray):
        raise TypeError(
            "from_pydata_sparse only accepts objects from the `sparse` library; "
            "see https://sparse.pydata.org"
        )
    if s.ndim > 2:
        raise _GraphblasException("m.ndim must be <= 2")

    if s.ndim == 1:
        # the .asformat('coo') makes it easier to convert dok/gcxs using a single approach
        _s = s.asformat("coo")
        return _Vector.from_coo(
            _s.coords, _s.data, dtype=_s.dtype, size=_s.shape[0], dup_op=dup_op, name=name
        )
    # handle two-dimensional arrays
    if isinstance(s, sparse.GCXS):
        return from_scipy_sparse(s.to_scipy_sparse(), dup_op=dup_op, name=name)
    if isinstance(s, (sparse.DOK, sparse.COO)):
        _s = s.asformat("coo")
        return _Matrix.from_coo(
            *_s.coords,
            _s.data,
            nrows=_s.shape[0],
            ncols=_s.shape[1],
            dtype=_s.dtype,
            dup_op=dup_op,
            name=name,
        )
    raise ValueError(f"Unknown sparse array type: {type(s).__name__}")  # pragma: no cover (safety)


# TODO: add parameters to allow different networkx classes and attribute names
def to_networkx(m, edge_attribute="weight"):
    """Create a networkx DiGraph from a square adjacency Matrix.

    Parameters
    ----------
    m : Matrix
        Square adjacency Matrix
    edge_attribute : str, optional
        Name of edge attribute from values of Matrix. If None, values will be skipped.
        Default is "weight".

    Returns
    -------
    nx.DiGraph
    """
    import networkx as nx

    rows, cols, vals = m.to_coo()
    rows = rows.tolist()
    cols = cols.tolist()
    G = nx.DiGraph()
    if edge_attribute is None:
        G.add_edges_from(zip(rows, cols))
    else:
        G.add_weighted_edges_from(zip(rows, cols, vals.tolist()), weight=edge_attribute)
    return G


def to_numpy(m):  # pragma: no cover (deprecated)
    """Create a dense numpy array from a sparse Vector or Matrix.

    .. deprecated:: 2023.2.0
        `to_numpy` will be removed in a future release.
        Use `Vector.to_dense` or `Matrix.to_dense` instead.
        Will be removed in version 2023.10.0 or later

    Missing values will become 0 in the output.

    numpy dtype will match the GraphBLAS dtype

    Parameters
    ----------
    m : Vector or Matrix
        GraphBLAS Vector or Matrix

    See Also
    --------
    to_scipy_sparse
    Matrix.to_dense
    Vector.to_dense

    Returns
    -------
    np.ndarray
    """
    _warn(
        "`graphblas.io.to_numpy` is deprecated; "
        "use `Matrix.to_dense` and `Vector.to_dense` instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    try:
        import scipy  # noqa: F401
    except ImportError:  # pragma: no cover (import)
        raise ImportError("scipy is required to export to numpy") from None
    if _output_type(m) is _Vector:
        return to_scipy_sparse(m).toarray()[0]
    sparse = to_scipy_sparse(m, "coo")
    return sparse.toarray()


def to_scipy_sparse(A, format="csr"):
    """Create a scipy.sparse array from a GraphBLAS Matrix or Vector.

    Parameters
    ----------
    A : Matrix or Vector
        GraphBLAS object to be converted
    format : str
        {'bsr', 'csr', 'csc', 'coo', 'lil', 'dia', 'dok'}

    Returns
    -------
    scipy.sparse array

    """
    import scipy.sparse as ss

    format = format.lower()
    if format not in {"bsr", "csr", "csc", "coo", "lil", "dia", "dok"}:
        raise ValueError(f"Invalid format: {format}")
    if _output_type(A) is _Vector:
        indices, data = A.to_coo()
        if format == "csc":
            return ss.csc_array((data, indices, [0, len(data)]), shape=(A._size, 1))
        rv = ss.csr_array((data, indices, [0, len(data)]), shape=(1, A._size))
        if format == "csr":
            return rv
    elif _backend == "suitesparse" and format in {"csr", "csc"}:
        if A._is_transposed:
            info = A.T.ss.export("csc" if format == "csr" else "csr", sort=True)
            if "col_indices" in info:
                info["row_indices"] = info["col_indices"]
            else:
                info["col_indices"] = info["row_indices"]
        else:
            info = A.ss.export(format, sort=True)
        values = _normalize_values(A, info["values"], None, (A._nvals,), info["is_iso"])
        if format == "csr":
            return ss.csr_array((values, info["col_indices"], info["indptr"]), shape=A.shape)
        return ss.csc_array((values, info["row_indices"], info["indptr"]), shape=A.shape)
    elif format == "csr":
        indptr, cols, vals = A.to_csr()
        return ss.csr_array((vals, cols, indptr), shape=A.shape)
    elif format == "csc":
        indptr, rows, vals = A.to_csc()
        return ss.csc_array((vals, rows, indptr), shape=A.shape)
    else:
        rows, cols, data = A.to_coo()
        rv = ss.coo_array((data, (rows, cols)), shape=A.shape)
        if format == "coo":
            return rv
    return rv.asformat(format)


_AwkwardDoublyCompressedMatrix = None


def to_awkward(A, format=None):
    """Create an Awkward Array from a GraphBLAS Matrix.

    Parameters
    ----------
    A : Matrix or Vector
        GraphBLAS object to be converted
    format : str {'csr', 'csc', 'hypercsr', 'hypercsc', 'vec}
        Default format is csr for Matrix; vec for Vector

    The Awkward Array will have top-level attributes based on format:
    - vec/csr/csc: values, indices
    - hypercsr/hypercsc: values, indices, offset_labels

    Top-level parameters will also be set: format, shape

    Returns
    -------
    awkward.Array

    """
    try:
        # awkward version 1
        # MAINT: we can probably drop awkward v1 at the end of 2024 or 2025
        import awkward._v2 as ak
        from awkward._v2.forms.listoffsetform import ListOffsetForm
        from awkward._v2.forms.numpyform import NumpyForm
        from awkward._v2.forms.recordform import RecordForm
    except ImportError:
        # awkward version 2
        import awkward as ak
        from awkward.forms.listoffsetform import ListOffsetForm
        from awkward.forms.numpyform import NumpyForm
        from awkward.forms.recordform import RecordForm

    out_type = _output_type(A)
    if format is None:
        format = "vec" if out_type is _Vector else "csr"
    format = format.lower()
    classname = None

    if out_type is _Vector:
        if format != "vec":
            raise ValueError(f"Invalid format for Vector: {format}")
        size = A.nvals
        indices, values = A.to_coo()
        form = RecordForm(
            contents=[
                NumpyForm(A.dtype.np_type.name, form_key="node1"),
                NumpyForm("int64", form_key="node0"),
            ],
            fields=["values", "indices"],
        )
        d = {"node0-data": indices, "node1-data": values}

    elif out_type is _Matrix:
        if format == "csr":
            indptr, cols, values = A.to_csr()
            d = {"node3-data": cols}
            size = A.nrows
        elif format == "csc":
            indptr, rows, values = A.to_csc()
            d = {"node3-data": rows}
            size = A.ncols
        elif format == "hypercsr":
            rows, indptr, cols, values = A.to_dcsr()
            d = {"node3-data": cols, "node5-data": rows}
            size = len(rows)
        elif format == "hypercsc":
            cols, indptr, rows, values = A.to_dcsc()
            d = {"node3-data": rows, "node5-data": cols}
            size = len(cols)
        else:
            raise ValueError(f"Invalid format for Matrix: {format}")
        d["node1-offsets"] = indptr
        d["node4-data"] = _np.ascontiguousarray(values)

        form = ListOffsetForm(
            "i64",
            RecordForm(
                contents=[
                    NumpyForm("int64", form_key="node3"),
                    NumpyForm(A.dtype.np_type.name, form_key="node4"),
                ],
                fields=["indices", "values"],
            ),
            form_key="node1",
        )
        if format.startswith("hyper"):
            global _AwkwardDoublyCompressedMatrix
            if _AwkwardDoublyCompressedMatrix is None:  # pylint: disable=used-before-assignment
                # Define behaviors to make all fields function at the top-level
                @ak.behaviors.mixins.mixin_class(ak.behavior)
                class _AwkwardDoublyCompressedMatrix:
                    @property
                    def values(self):  # pragma: no branch (???)
                        return self.data.values

                    @property
                    def indices(self):  # pragma: no branch (???)
                        return self.data.indices

            form = RecordForm(
                contents=[
                    form,
                    NumpyForm("int64", form_key="node5"),
                ],
                fields=["data", "offset_labels"],
            )
            classname = "_AwkwardDoublyCompressedMatrix"

    else:
        raise TypeError(f"A must be a Matrix or Vector, found {type(A)}")

    ret = ak.from_buffers(form, size, d)
    ret = ak.with_parameter(ret, "format", format)
    ret = ak.with_parameter(ret, "shape", list(A.shape))
    if classname:
        ret = ak.with_name(ret, classname)
    return ret


def to_pydata_sparse(A, format="coo"):
    """Create a pydata.sparse array from a GraphBLAS Matrix or Vector.

    Parameters
    ----------
    A : Matrix or Vector
        GraphBLAS object to be converted
    format : str
        {'coo', 'dok', 'gcxs'}

    Returns
    -------
    sparse array (see https://sparse.pydata.org)

    """
    try:
        from sparse import COO
    except ImportError:  # pragma: no cover (import)
        raise ImportError("sparse is required to export to pydata sparse") from None

    format = format.lower()
    if format not in {"coo", "dok", "gcxs"}:
        raise ValueError(f"Invalid format: {format}")

    if _output_type(A) is _Vector:
        indices, values = A.to_coo(sort=False)
        s = COO(indices, values, shape=A.shape)
    else:
        if format == "gcxs":
            B = to_scipy_sparse(A, format="csr")
        else:
            # obtain an intermediate conversion via hardcoded 'coo' intermediate object
            B = to_scipy_sparse(A, format="coo")
        # convert to pydata.sparse
        s = COO.from_scipy_sparse(B)

    # express in the desired format
    return s.asformat(format)


def mmread(source, engine="auto", *, dup_op=None, name=None, **kwargs):
    """Create a GraphBLAS Matrix from the contents of a Matrix Market file.

    This uses `scipy.io.mmread
    <https://docs.scipy.org/doc/scipy/reference/generated/scipy.io.mmread.html>`_
    or `fast_matrix_market.mmread
    <https://github.com/alugowski/fast_matrix_market/tree/main/python>`_.

    By default, ``fast_matrix_market`` will be used if available, because it
    is faster. Additional keyword arguments in ``**kwargs`` will be passed
    to the engine's ``mmread``. For example, ``parallelism=8`` will set the
    number of threads to use to 8 when using ``fast_matrix_market``.

    Parameters
    ----------
    source : str or file
        Filename (.mtx or .mtz.gz) or file-like object
    engine : {"auto", "scipy", "fmm", "fast_matrix_market"}, default "auto"
        How to read the matrix market file. "scipy" uses ``scipy.io.mmread``,
        "fmm" and "fast_matrix_market" uses ``fast_matrix_market.mmread``,
        and "auto" will use "fast_matrix_market" if available.
    dup_op : BinaryOp, optional
        Aggregation function for duplicate coordinates (if found)
    name : str, optional
        Name of resulting Matrix

    Returns
    -------
    :class:`~graphblas.Matrix`
    """
    try:
        # scipy is currently needed for *all* engines
        from scipy.io import mmread
        from scipy.sparse import isspmatrix_coo
    except ImportError:  # pragma: no cover (import)
        raise ImportError("scipy is required to read Matrix Market files") from None
    engine = engine.lower()
    if engine in {"auto", "fmm", "fast_matrix_market"}:
        try:
            from fast_matrix_market import mmread  # noqa: F811
        except ImportError:  # pragma: no cover (import)
            if engine != "auto":
                raise ImportError(
                    "fast_matrix_market is required to read Matrix Market files "
                    f'using the "{engine}" engine'
                ) from None
    elif engine != "scipy":
        raise ValueError(
            f'Bad engine value: {engine!r}. Must be "auto", "scipy", "fmm", or "fast_matrix_market"'
        )
    array = mmread(source, **kwargs)
    if isspmatrix_coo(array):
        nrows, ncols = array.shape
        return _Matrix.from_coo(
            array.row, array.col, array.data, nrows=nrows, ncols=ncols, dup_op=dup_op, name=name
        )
    return _Matrix.from_dense(array, name=name)


def mmwrite(
    target,
    matrix,
    engine="auto",
    *,
    comment="",
    field=None,
    precision=None,
    symmetry=None,
    **kwargs,
):
    """Write a Matrix Market file from the contents of a GraphBLAS Matrix.

    This uses `scipy.io.mmwrite
    <https://docs.scipy.org/doc/scipy/reference/generated/scipy.io.mmwrite.html>`_.

    Parameters
    ----------
    target : str or file target
        Filename (.mtx) or file-like object opened for writing
    matrix : Matrix
        Matrix to be written
    engine : {"auto", "scipy", "fmm", "fast_matrix_market"}, default "auto"
        How to read the matrix market file. "scipy" uses ``scipy.io.mmwrite``,
        "fmm" and "fast_matrix_market" uses ``fast_matrix_market.mmwrite``,
        and "auto" will use "fast_matrix_market" if available.
    comment : str, optional
        Comments to be prepended to the Matrix Market file
    field : str
        {"real", "complex", "pattern", "integer"}
    precision : int, optional
        Number of digits to write for real or complex values
    symmetry : str, optional
        {"general", "symmetric", "skew-symmetric", "hermetian"}
    """
    try:
        # scipy is currently needed for *all* engines
        from scipy.io import mmwrite
    except ImportError:  # pragma: no cover (import)
        raise ImportError("scipy is required to write Matrix Market files") from None
    engine = engine.lower()
    if engine in {"auto", "fmm", "fast_matrix_market"}:
        try:
            from fast_matrix_market import mmwrite  # noqa: F811
        except ImportError:  # pragma: no cover (import)
            if engine != "auto":
                raise ImportError(
                    "fast_matrix_market is required to write Matrix Market files "
                    f'using the "{engine}" engine'
                ) from None
    elif engine != "scipy":
        raise ValueError(
            f'Bad engine value: {engine!r}. Must be "auto", "scipy", "fmm", or "fast_matrix_market"'
        )
    if _backend == "suitesparse" and matrix.ss.format in {"fullr", "fullc"}:
        array = matrix.ss.export()["values"]
    else:
        array = to_scipy_sparse(matrix, format="coo")
    mmwrite(
        target,
        array,
        comment=comment,
        field=field,
        precision=precision,
        symmetry=symmetry,
        **kwargs,
    )
