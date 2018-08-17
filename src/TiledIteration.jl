module TiledIteration

using OffsetArrays
using Base: tail, Indices, @propagate_inbounds
using Base.IteratorsMD: inc

export TileIterator, EdgeIterator, padded_tilesize, TileBuffer

### TileIterator ###

const L1cachesize = 2^15
const cachelinesize = 64

struct TileIterator{N,I,UR}
    inds::I
    sz::Dims{N}
    R::CartesianIndices{N,UR}
end
rangetype(::CartesianIndices{N,T}) where {N,T} = T
function TileIterator(inds::Indices{N}, sz::Dims{N}) where N
    ls = map(length, inds)
    ns = map(ceildiv, ls, sz)
    R = CartesianIndices(ns)
    TileIterator{N,typeof(inds),rangetype(R)}(inds, sz, R)
end

Iterators.IteratorEltype(::Type{<:TileIterator}) = Iterators.HasEltype()

ceildiv(l, s) = ceil(Int, l/s)

Base.length(iter::TileIterator) = length(iter.R)
Base.eltype(iter::TileIterator{N}) where {N} = NTuple{N,UnitRange{Int}}

function Base.iterate(iter::TileIterator)
    iterR = iterate(iter.R)
    iterR === nothing && return nothing
    I, state = iterR
    return getindices(iter, I), state
end
function Base.iterate(iter::TileIterator, state)
    iterR = iterate(iter.R, state)
    iterR === nothing && return nothing
    I, newstate = iterR
    return getindices(iter, I), newstate
end

Base.show(io::IO, iter::TileIterator) = print(io, "TileIterator(", iter.inds, ", ", iter.sz, ')')

@inline function getindices(iter::TileIterator, I::CartesianIndex)
    map3(_getindices, iter.inds, iter.sz, I.I)
end
_getindices(ind, s, i) = first(ind)+(i-1)*s : min(last(ind),first(ind)+i*s-1)
map3(f, ::Tuple{}, ::Tuple{}, ::Tuple{}) = ()
@inline map3(f, a::Tuple, b::Tuple, c::Tuple) = (f(a[1], b[1], c[1]), map3(f, tail(a), tail(b), tail(c))...)

### EdgeIterator ###

struct EdgeIterator{N,UR1,UR2}
    outer::CartesianIndices{N,UR1}
    inner::CartesianIndices{N,UR2}
    function EdgeIterator{N,UR1,UR2}(outer::CartesianIndices{N}, inner::CartesianIndices{N}) where {N,UR1,UR2}
        ((first(inner) ∈ outer) & (last(inner) ∈ outer)) || throw(DimensionMismatch("$inner must be in the interior of $outer"))
        new(outer, inner)
    end
end
EdgeIterator(outer::CartesianIndices{N,UR1}, inner::CartesianIndices{N,UR2}) where {N,UR1,UR2} =
    EdgeIterator{N,UR1,UR2}(outer, inner)
EdgeIterator(outer::Indices{N}, inner::Indices{N}) where N =
    EdgeIterator(promote(CartesianIndices(outer), CartesianIndices(inner))...)

"""
    EdgeIterator(outer, inner)

A Cartesian iterator that efficiently visits sites that are in `outer`
but not in `inner`. This can be useful for calculating edge values
that may require special treatment or boundary conditions.
"""
EdgeIterator

Iterators.IteratorEltype(::Type{<:EdgeIterator}) = Iterators.HasEltype()

Base.eltype(::Type{EdgeIterator{N,UR1,UR2}}) where {N,UR1,UR2} = CartesianIndex{N}
Base.length(iter::EdgeIterator) = length(iter.outer) - length(iter.inner)

function Base.iterate(iter::EdgeIterator)
    iterouter = iterate(iter.outer)
    iterouter === nothing && return nothing
    item = nextedgeitem(iter, iterouter[2])
    item ∉ iter.outer && return nothing
    return item, item
end
function Base.iterate(iter::EdgeIterator, state)
    iterouter = iterate(iter.outer, state)
    iterouter === nothing && return nothing
    item = nextedgeitem(iter, iterouter[2])
    item.I[end] > last(iter.outer.indices[end]) && return nothing
    return item, item
end

@inline function nextedgeitem(iter::EdgeIterator, I::CartesianIndex)
    !(I ∈ iter.inner) && return I
    newI = CartesianIndex(inc((last(iter.inner)[1], tail(I.I)...), first(iter.outer).I, last(iter.outer).I))
    nextedgeitem(iter, newI)
end

Base.show(io::IO, iter::EdgeIterator) = print(io, "EdgeIterator(", iter.outer.indices, ", ", iter.inner.indices, ')')

### Calculating the size of tiles ###

# If kernelsize-1 is the amount of padding, and s is the extra width of the tile along
# each axis, then the fraction of useful to total work is
# prod(s)/prod(s+kernelsize); this ratio is maximized if s is proportional to
# kernelsize.
"""
    padded_tilesize(T::Type, kernelsize::Dims, [ncache=2]) -> tilesize::Dims

Calculate a suitable tile size to approximately maximize the amount of
productive work, given a stencil of size `kernelsize`. The element
type of the array is `T`. Optionally specify `ncache`, the number of
such arrays that you'd like to have fit simultaneously in L1 cache.

This favors making the first dimension larger, since the first
dimension corresponds to individual cache lines.

# Examples
julia> padded_tilesize(UInt8, (3,3))
(768,18)

julia> padded_tilesize(UInt8, (3,3), 4)
(512,12)

julia> padded_tilesize(Float64, (3,3))
(96,18)

julia> padded_tilesize(Float32, (3,3,3))
(64,6,6)
"""
function padded_tilesize(::Type{T}, kernelsize::Dims, ncache = 2) where T
    nd = max(1, sum(x->x>1, kernelsize))
    # isbits(T) || return map(zero, kernelsize)
    # don't be too minimalist on the cache-friendly dim (use at least 2 cachelines)
    dim1minlen = 2*cachelinesize÷sizeof(T)
    psz = (max(kernelsize[1], dim1minlen), tail(kernelsize)...)
    L = sizeof(T)*prod(psz)
    # try to stay in L1 cache, but in the end we want a reasonably
    # favorable work ratio. f is the constant of proportionality in
    #      s+kernelsize ∝ kernelsize
    f = max(floor(Int, (L1cachesize/(ncache*L))^(1/nd)), 2)
    return _padded_tilesize_scale(f, psz)
end

@noinline _padded_tilesize_scale(f, psz) = map(x->x <= 1 ? x : f*x, psz) # see #15276

### Tile shaping and coordinate transformation

struct TileBuffer{T,N,P} <: AbstractArray{T,N}
    view::OffsetArray{T,N,Array{T,N}}  # the currently-active view
    buf::Array{T,P}                    # the original backing buffer
end

"""
    TileBuffer(a, inds::Indices) -> v

Return a buffer-view `v` whose indices match `inds`, using the array
`a` for storage. `inds` does not necessarily have to match the size of
`a` (which allows tiles to be of different sizes, e.g., at the edges).
"""
@inline function TileBuffer(a::Array, inds::Indices)
    l = map(length, inds)
    tilev = OffsetArray(_tileview(a, l), inds)
    TileBuffer(tilev, a)
end

"""
    TileBuffer(T, inds::Indices) -> v

Return a TileBuffer, allocating a new backing array of element type `T`
and size determined by `inds`.
"""
function TileBuffer(::Type{T}, inds::Indices) where T
    l = map(length, inds)
    TileBuffer(Array{T}(undef, l), inds)
end

TileBuffer(tb::TileBuffer, inds::Indices) = TileBuffer(tb.buf, inds)

@inline function _tileview(a::Array, l::Dims)
    if size(a) == l
        return a
    else
        # returning a SubArray would not be type-stable, we must return an Array
        prod(l) > length(a) && throw(DimensionMismatch("array of size $(size(a)) is not adequate for a tile of size $l"))
        return unsafe_wrap(Array, pointer(a), l)
    end
end

Base.axes(tb::TileBuffer) = axes(tb.view)
Base.size(tb::TileBuffer) = size(tb.view)

@inline @propagate_inbounds Base.getindex(tb::TileBuffer{T,N}, I::Vararg{Int,N}) where {T,N} = tb.view[I...]

@inline @propagate_inbounds Base.setindex!(tb::TileBuffer{T,N}, val, I::Vararg{Int,N}) where {T,N} = tb.view[I...] = val

Base.pointer(tb::TileBuffer) = pointer(parent(tb.view))

Base.parent(tb::TileBuffer) = tb.buf

end # module
