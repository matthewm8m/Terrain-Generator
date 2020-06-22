import itertools as it

import numpy as np
import matplotlib.pyplot as plt

import numpy as np

class BlockRandomGenerator():
    def __init__(self, block_size, block_dim, block_repeat=1<<32, rv=None, seed=None, args=(), **kwargs):
        # Set block parameters.
        #   Block shape = (block size, block size, ..., block size) # block dimension times 
        self.block_size = block_size
        self.block_dimension = block_dim
        self.block_repeat = block_repeat
        self.block_shape = (block_size,) * block_dim 

        # Set random variable parameters.
        #   Default random variable generator is Uniform[0, 1].
        if rv is None:
            self.random = np.random.uniform
        else:
            self.random = rv
        self.random_args = args
        self.random_kwargs = kwargs

        # Set seeding parameters.
        #   We need the seed size to easily format seeds later.
        self.seed_size = 0
        block_repeat -= 1
        while block_repeat > 0:
            block_repeat = block_repeat >> 32
            self.seed_size += 1

        #   Default seed is random seed from Uniform[0, block repeat - 1].
        #   All seeds must be in the range [0, block repeat - 1].
        if seed is None:
            np.random.seed(None)
            self.seed = np.random.randint(self.block_repeat)
        else:
            self.seed = self.clamp_seed(seed)

    def clamp_seed(self, seed):
        return (seed % self.block_repeat + self.block_repeat) % self.block_repeat

    def format_seed(self, seed):
        seed = self.clamp_seed(seed)
        seed_max = 1 << 32
        for _ in range(self.seed_size):
            yield seed % seed_max
            seed = seed >> 32

    def generate_block(self, index):
        # Seed the random generator based on block.
        np.random.seed(
            [s for d in range(self.block_dimension) for s in self.format_seed(self.seed + index[d])]
        )

        # Generate the actual random block.
        random_block = self.random(
            *self.random_args,
            **self.random_kwargs,
            size=self.block_shape)

        return random_block
    
    def generate_blocks(self, indices):
        for index in indices:
            yield self.generate_block(index)

class ScalarNoiseGenerator():
    def __init__(self, dimension=1, trend_size=2, trend_levels=8, smoothness=0.5, seed=None):
        self.dimension = dimension
        self.trend_size = trend_size
        self.trend_levels = trend_levels
        self.smoothness = smoothness

        # Pick a random seed if none is given.
        # Make sure that the seed is within [0, 2^32-1].
        # Convert the seed to a format of (s1, s2, ..., sn) where n is divisible by d.
        seed_max = int(2**32)
        if seed is None:
            np.random.seed(None)
            self.seed = tuple(np.random.randint(seed_max, size=self.dimension))
        else:
            # Allow for tuple seeds of length divisible by dimension.
            if isinstance(tuple):
                if len(seed) % self.dimension == 0:
                    self.seed = tuple((s % seed_max + seed_max) % seed_max for s in seed)
                else:
                    raise ValueError("Invalid length of seed. Must be divisible by number of dimensions.")
            else:
                self.seed = ((seed % seed_max + seed_max) % seed_max,) * self.dimension

    def generate(self, index=None, size=None):
        # Get the size of a block.
        block_size = self.trend_size ** self.trend_levels

        # Setup the default index and size.
        if index is None:
            index = (0,) * self.dimension
        if size is None:
            size = (block_size,) * self.dimension

        # Get the bounding box of the selected location.
        bound_lower = tuple(np.floor(x / block_size) for x in index)
        bound_upper = tuple(np.ceil((x + s) / block_size) + 1 for x, s in zip(index, size))

        # Calculate the subsets of dimensions to perform each multiplication of the interpolation.
        dimension_range = range(self.dimension)
        dimension_subsets = tuple(it.chain.from_iterable(it.combinations(dimension_range, r) for r in range(self.dimension+1)))

        # Create the noise base with size of maximum level.
        seed_max = int(2**32)
        noise_base = np.zeros(tuple(int(block_size * (u - l)) for l, u in zip(bound_lower, bound_upper)))
        noise_indices = np.array(np.meshgrid(*(np.arange(l, u) for l, u in zip(bound_lower, bound_upper)))).reshape(self.dimension, -1).T
        for idx in tuple(noise_indices):
            # Seed the noise generator with the seed specified.
            block_seed = tuple(
                int(((s + idx[i % self.dimension]) % seed_max + seed_max) % seed_max)
                for i, s in enumerate(self.seed)
            )
            np.random.seed(block_seed)

            # Fill the noise base.
            idx_adjusted = tuple(idx[d] - bound_lower[d] for d in dimension_range)
            idx_slice = tuple(slice(int(block_size * idx_adj), int(block_size * (idx_adj + 1))) for idx_adj in idx_adjusted)
            noise_base[idx_slice] = np.random.rand(*((block_size,) * self.dimension))

        # Create the noise result with size of maximum level.
        noise_result = np.zeros_like(noise_base)

        # Add noise from each level of detail        
        for level in range(self.trend_levels):
            # Calculate the size and scale of the level.
            level_size = self.trend_size ** (self.trend_levels - level)
            level_scale = self.trend_size ** level

            # Allocate space for the new noise added.
            noise_added = np.zeros_like(noise_result)

            # Calculate the new base noise from the level.
            noise_base_shape = tuple(int(s) for d in dimension_range for s in ((bound_upper[d] - bound_lower[d]) * level_size, level_scale))
            noise_base = noise_base.reshape(*noise_base_shape)
            noise_base = np.mean(noise_base, axis=tuple(2 * d + 1 for d in dimension_range))
            for d in dimension_range:
                noise_base = np.repeat(noise_base, level_scale, axis=d)

            # Compute the noise added from each possible combination of dimensions.
            for dimensions in dimension_subsets:
                # Compute the shift in each dimension.
                dimensions_roll = tuple(-level_scale if d in dimensions else 0 for d in dimension_range)
                
                # Compute the noise shifted in each dimension.
                noise_rolled = np.roll(noise_base, dimensions_roll, axis=tuple(dimension_range))

                # Compute the noise multiplied by the correct linear coefficients.
                for d in dimension_range:
                    slices = tuple(slice(None) if k == d else None for k in dimension_range)

                    # Create the linear coefficient array for linear interpolation.
                    coeffs_linear = np.linspace(0.0, 1.0, level_scale, endpoint=False)
                    coeffs_linear = np.tile(coeffs_linear, int((bound_upper[d] - bound_lower[d]) * level_size))

                    if d in dimensions:
                        noise_rolled = noise_rolled * coeffs_linear[slices]
                    else:
                        noise_rolled = noise_rolled * (1 - coeffs_linear[slices])

                # Add the noise shifted to the noise to be added.
                noise_added = noise_added + noise_rolled

            # Add the noise to be added to the result noise.
            noise_result = (1 - self.smoothness) * noise_result + self.smoothness * noise_added
        
        # Clip the noise result to the original location.
        index_bounded = tuple(slice(int(i - block_size * bound_lower[k]), int(i - block_size * bound_lower[k] + s)) for k, (i, s) in enumerate(zip(index, size)))
        noise_result = noise_result[index_bounded]

        return noise_result