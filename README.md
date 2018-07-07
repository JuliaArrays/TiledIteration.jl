# TiledIteration

[![Build Status](https://travis-ci.org/JuliaArrays/TiledIteration.jl.svg?branch=master)](https://travis-ci.org/JuliaArrays/TiledIteration.jl)

[![codecov.io](http://codecov.io/github/JuliaArrays/TiledIteration.jl/coverage.svg?branch=master)](http://codecov.io/github/JuliaArrays/TiledIteration.jl?branch=master)

This Julia package handles some of the low-level details for writing
cache-efficient, possibly-multithreaded code for multidimensional
arrays. A "tile" corresponds to a chunk of a larger array, typically a
region that is large enough to encompass any "local" computations you
need to perform; some of these computations may require temporary storage.

## Usage

This package offers two basic kinds of functionality: the management
of temporary buffers for processing on tiles, and the iteration over
disjoint tiles of a larger array.

### Iteration

To iterate over disjoint tiles of a larger array, use a `TileIterator`:

```julia
using TiledIteration

A = rand(1000,1000);   # our big array
for tileaxs in TileIterator(axes(A), (128,8))
    @show tileaxs
end
```

This produces
```julia
tileaxs = (1:128,1:8)
tileaxs = (129:256,1:8)
tileaxs = (257:384,1:8)
tileaxs = (385:512,1:8)
tileaxs = (513:640,1:8)
tileaxs = (641:768,1:8)
tileaxs = (769:896,1:8)
tileaxs = (897:1000,1:8)
tileaxs = (1:128,9:16)
tileaxs = (129:256,9:16)
tileaxs = (257:384,9:16)
tileaxs = (385:512,9:16)
...
```

You can see that the total axes range is split up into chunks,
which are of size `(128,8)` except at the edges of `A`. Naturally,
these axes serve as the basis for processing individual chunks of
the array.

As a further example, suppose you've started julia with `JULIA_NUM_THREADS=4`; then

```julia
function fillid!(A, tilesz)
    tileinds_all = collect(TileIterator(axes(A), tilesz))
    Threads.@threads for i = 1:length(tileinds_all)
        tileaxs = tileinds_all[i]
        A[tileaxs...] .= Threads.threadid()
    end
    A
end

A = zeros(Int, 8, 8)
fillid!(A, (2,2))
```

would yield

```julia
8×8 Array{Int64,2}:
 1  1  2  2  3  3  4  4
 1  1  2  2  3  3  4  4
 1  1  2  2  3  3  4  4
 1  1  2  2  3  3  4  4
 1  1  2  2  3  3  4  4
 1  1  2  2  3  3  4  4
 1  1  2  2  3  3  4  4
 1  1  2  2  3  3  4  4
```

See also "EdgeIterator" below.

### Determining the chunk size

[Stencil computations](https://en.wikipedia.org/wiki/Stencil_code)
typically require "padding" values, so the inputs to a computation may
be of a different size than the resulting outputs. Naturally, you can
set the tile size manually; a simple convenience function,
`padded_tilesize`, attempts to pick reasonable choices for you
depending on the size of your kernel (stencil) and element type you'll
be using:

```julia
julia> padded_tilesize(UInt8, (3,3))
(768,18)

julia> padded_tilesize(UInt8, (3,3), 4)  # we want 4 of these to fit in L1 cache at once
(512,12)

julia> padded_tilesize(Float64, (3,3))
(96,18)

julia> padded_tilesize(Float32, (3,3,3))
(64,6,6)
```

### Allocating and managing temporary storage

To allocate temporary storage while working with tiles, use `TileBuffer`:

```julia
julia> tileaxs = (-1:15, 0:7)  # really this might have come from TileIterator

julia> buf = TileBuffer(Float32, tileaxs)
TiledIteration.TileBuffer{Float32,2,2} with indices -1:15×0:7:
 0.0  0.0          2.38221f-44  0.0          0.0          0.0          9.3887f-44   0.0
 0.0  1.26117f-44  0.0          0.0          0.0          8.26766f-44  0.0          0.0
 0.0  0.0          0.0          0.0          0.0          0.0          0.0          0.0
 0.0  0.0          0.0          6.02558f-44  0.0          0.0          0.0          0.0
 0.0  0.0          0.0          0.0          7.28675f-44  0.0          0.0          0.0
 0.0  1.54143f-44  0.0          0.0          0.0          0.0          0.0          0.0
 0.0  0.0          0.0          0.0          0.0          0.0          0.0          0.0
 0.0  0.0          0.0          0.0          0.0          0.0          0.0          0.0
 0.0  0.0          0.0          0.0          0.0          0.0          9.94922f-44  0.0
 0.0  0.0          0.0          0.0          0.0          8.82818f-44  0.0          0.0
 0.0  0.0          0.0          0.0          0.0          0.0          0.0          0.0
 0.0  0.0          0.0          0.0          0.0          0.0          0.0          0.0
 0.0  0.0          0.0          0.0          0.0          0.0          0.0          0.0
 0.0  0.0          0.0          0.0          0.0          9.10844f-44  0.0          0.0
 0.0  0.0          0.0          0.0          0.0          0.0          1.03696f-43  0.0
 0.0  0.0          0.0          0.0          0.0          0.0          0.0          0.0
 0.0  0.0          0.0          0.0          0.0          0.0          0.0          0.0
```

This returns an uninitialized buffer for use over the indicated domain. You can reuse this same storage for the next tile, even if the tile is smaller because it corresponds to the edge of the original array:

```julia
julia> pointer(buf)
Ptr{Float32} @0x00007f79131fd550

julia> buf = TileBuffer(buf, (16:20, 0:7))
TiledIteration.TileBuffer{Float32,2,2} with indices 16:20×0:7:
 0.0  0.0  0.0  0.0          0.0          0.0  0.0          0.0
 0.0  0.0  0.0  0.0          0.0          0.0  0.0          0.0
 0.0  0.0  0.0  0.0          1.54143f-44  0.0  0.0          0.0
 0.0  0.0  0.0  1.26117f-44  0.0          0.0  0.0          0.0
 0.0  0.0  0.0  0.0          0.0          0.0  2.38221f-44  0.0

julia> pointer(buf)
Ptr{Float32} @0x00007f79131fd550
```

When you use it again at the top of the next block of columns, it returns to its original size while still reusing the same memory:
```julia
julia> buf = TileBuffer(buf, (-1:15, 8:15))
TiledIteration.TileBuffer{Float32,2,2} with indices -1:15×8:15:
 0.0  0.0          2.38221f-44  0.0          0.0          0.0          9.3887f-44   0.0
 0.0  1.26117f-44  0.0          0.0          0.0          8.26766f-44  0.0          0.0
 0.0  0.0          0.0          0.0          0.0          0.0          0.0          0.0
 0.0  0.0          0.0          6.02558f-44  0.0          0.0          0.0          0.0
 0.0  0.0          0.0          0.0          7.28675f-44  0.0          0.0          0.0
 0.0  1.54143f-44  0.0          0.0          0.0          0.0          0.0          0.0
 0.0  0.0          0.0          0.0          0.0          0.0          0.0          0.0
 0.0  0.0          0.0          0.0          0.0          0.0          0.0          0.0
 0.0  0.0          0.0          0.0          0.0          0.0          9.94922f-44  0.0
 0.0  0.0          0.0          0.0          0.0          8.82818f-44  0.0          0.0
 0.0  0.0          0.0          0.0          0.0          0.0          0.0          0.0
 0.0  0.0          0.0          0.0          0.0          0.0          0.0          0.0
 0.0  0.0          0.0          0.0          0.0          0.0          0.0          0.0
 0.0  0.0          0.0          0.0          0.0          9.10844f-44  0.0          0.0
 0.0  0.0          0.0          0.0          0.0          0.0          1.03696f-43  0.0
 0.0  0.0          0.0          0.0          0.0          0.0          0.0          0.0
 0.0  0.0          0.0          0.0          0.0          0.0          0.0          0.0

julia> pointer(buf)
Ptr{Float32} @0x00007f79131fd550
```

### EdgeIterator

When performing stencil operations, oftentimes the edge of the array
requires special treatment. Several approaches to handling the edges
(adding explicit padding, or executing special code just when on the
boundaries) can slow your algorithm down because of extra steps or
branches.

This package helps support implementations which first handle the
"interior" of an array (for example using `TiledIterator` over just
the interior) using a "fast path," and then handle just the edges by a
(possibly) less carefully optimized algorithm. The key component of
this is `EdgeIterator`:

```julia
outerrange = CartesianIndices((-1:4, 0:3))
innerrange = CartesianIndices(( 1:3, 1:2))
julia> for I in EdgeIterator(outerrange, innerrange)
           @show I
       end
I = CartesianIndex(-1, 0)
I = CartesianIndex(0, 0)
I = CartesianIndex(1, 0)
I = CartesianIndex(2, 0)
I = CartesianIndex(3, 0)
I = CartesianIndex(4, 0)
I = CartesianIndex(-1, 1)
I = CartesianIndex(0, 1)
I = CartesianIndex(4, 1)
I = CartesianIndex(-1, 2)
I = CartesianIndex(0, 2)
I = CartesianIndex(4, 2)
I = CartesianIndex(-1, 3)
I = CartesianIndex(0, 3)
I = CartesianIndex(1, 3)
I = CartesianIndex(2, 3)
I = CartesianIndex(3, 3)
I = CartesianIndex(4, 3)
```

The time required to visit these edge sites is on the order of the
number of edge sites, not the order of the number of sites encompassed
by `outerrange`, and consequently is efficient.
