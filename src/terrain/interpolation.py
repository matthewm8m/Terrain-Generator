"""This module includes methods to interpolate data from a regular grid of data."""

import numpy as np
import numpy.polynomial as poly

class NearestNeighborInterpolator():
    def __init__(self, data):
        """Interpolates from a given grid of data by producing the nearest grid value.
        
        Arguments:
            data {numpy.ndarray} -- The grid of data used for interpolation.
        
        Raises:
            TypeError: Argument `data` is not a numpy.ndarray.
        """

        # Data validation checks.
        if not isinstance(data, np.ndarray):
            raise TypeError("Input data must be an array.")

        self.data = data

    def __call__(self, positions):
        """Interpolates data at the given positions.
        
        Arguments:
            positions {numpy.ndarray} -- An array specifying positions to sample data from. The last dimension of the array must be the same size as the grid.
        
        Raises:
            TypeError: Argument `positions` is not a numpy.ndarray.
        """

        # Positions validation checks.
        if not isinstance(positions, np.ndarray):
            raise TypeError("Positions data must be an array.")
        elif positions.shape[-1] != self.data.ndim:
            raise ValueError("Positions data shape must have last dimension that is same as that of the grid.")
        
        # Compute the rounded positions as indices.
        positions_shape = positions.shape
        positions = np.minimum(np.array(self.data.shape) - 1, np.maximum(0, (positions + 0.5).astype(int)))
        positions = positions.reshape(-1, self.data.ndim)

        return self.data[tuple(positions[:, d] for d in range(self.data.ndim))].reshape(positions_shape[:-1])

class PolynomialInterpolator():
    def __init__(self, data, degree=1, hermite=False):
        """Interpolates from a given grid of data by producing the fitted polynomial value.
        
        Arguments:
            data {numpy.ndarray} -- The grid of data used for interpolation.
        
        Keyword Arguments:
            degree {int} -- The degree of the polynomial to fit. (default: {1})
        
        Raises:
            TypeError: Argument 'data' is not a numpy.ndarray.
            TypeError: Argument 'degree' is not an integer or an iterable.
            ValueError: Argument 'degree' is non-positive.
            ValueError: Argument 'degree' is too large to interpolate across a dimension.
            ValueError: Argument 'degree' is does not have the correct number of dimensions.
        """

        # Data validation checks.
        if not isinstance(data, np.ndarray):
            raise TypeError("Input data must be an array.")

        # Degree validation checks.
        if not isinstance(degree, int) and not isinstance(degree, tuple):
            raise TypeError("Degree must be an integer or iterable.")
        if isinstance(degree, int):
            if degree < 0:
                raise ValueError("Degree must be non-negative.")
            for d in range(data.ndim):
                if degree >= data.shape[d]:
                    raise ValueError("Degree must be less than the size of all dimensions.")
            degree = tuple(degree for _ in range(data.ndim))
        else:
            if len(degree) != data.ndim:
                raise ValueError("Number of degrees must equal the number of dimensions.")
            for d in range(data.ndim):
                if degree[d] < 0:
                    raise ValueError("Degree must be non-negative.")
                if degree[d] >= data.shape[d]:
                    raise ValueError("Degree must be less than the size of all dimensions.")

        self.data = data
        self.degree = degree
        self.hermite = hermite

    def __call__(self, positions):
        """Interpolates data at the given positions.
        
        Arguments:
            positions {numpy.ndarray} -- An array specifying positions to sample data from. The last dimension of the array must be the same size as the grid.
        
        Raises:
            TypeError: Argument `positions` is not a numpy.ndarray.
        """

        half_degrees = np.array(tuple(self.degree[d] / 2.0 for d in range(self.data.ndim)))
        lower_degrees = np.zeros(self.data.ndim)
        upper_degrees = np.array(tuple(self.data.shape[d] - self.degree[d] for d in range(self.data.ndim)))

        origins = np.around(np.clip(positions - half_degrees, lower_degrees, upper_degrees)).astype(int)

        print(origins)

        interpolators = {}
        for dim in range(self.data.ndim):
            for i, position in enumerate(positions):
                dim_coords = tuple(position[dim_k] for dim_k in range(dim))
                dim_range = list(np.arange(origins[i, dim], origins[i, dim] + self.degree[dim] + 1))
                origins_partial = np.array(tuple(map(np.ravel,
                                  np.meshgrid(*tuple(
                                  np.arange(origins[i, dim_k], origins[i, dim_k] + self.degree[dim_k])
                                  for dim_k in range(dim + 1, self.data.ndim)))))).T

                print(position)
                print(dim_coords)
                print(dim_range)
                print(origins_partial)
                print()

                for origin_partial in origins_partial:
                    op_tuple = tuple(origin_partial)

                    if op_tuple not in interpolators:
                        if dim == 0:
                            dim_values = self.data[dim_coords + (dim_range,) + op_tuple]
                        else:
                            dim_values = [interpolators[](self.position[dim - 1]) for ]

                        polyfit = poly.Polynomial.fit(dim_range, dim_values, deg=self.degree[dim])
                        interpolators[op_tuple] = polyfit

