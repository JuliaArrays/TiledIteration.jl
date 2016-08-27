__precompile__()

module TiledIteration

using OffsetArrays
using Base: tail, Indices

export TileIterator, padded_tilesize, TileView

const L1cachesize = 2^15
const cachelinesize = 64

immutable TileIterator{N,I}
    inds::I
    sz::Dims{N}
    R::CartesianRange{CartesianIndex{N}}
end

function TileIterator{N}(inds::Indices{N}, sz::Dims{N})
    ls = map(length, inds)
    ns = map(ceildiv, ls, sz)
    TileIterator{N,typeof(inds)}(inds, sz, CartesianRange(ns))
end
ceildiv(l, s) = ceil(Int, l/s)

Base.length(iter::TileIterator) = length(iter.R)
Base.eltype{N}(iter::TileIterator{N}) = NTuple{N,UnitRange{Int}}

@inline Base.start(iter::TileIterator) = start(iter.R)
@inline function Base.next(iter::TileIterator, state)
    I, newstate = next(iter.R, state)
    getindices(iter, I), newstate
end
@inline Base.done(iter::TileIterator, state) = done(iter.R, state)

@inline function getindices(iter::TileIterator, I::CartesianIndex)
    map3(_getindices, iter.inds, iter.sz, I.I)
end
_getindices(ind, s, i) = first(ind)+(i-1)*s : min(last(ind),first(ind)+i*s-1)
map3(f, ::Tuple{}, ::Tuple{}, ::Tuple{}) = ()
@inline map3(f, a::Tuple, b::Tuple, c::Tuple) = (f(a[1], b[1], c[1]), map3(f, tail(a), tail(b), tail(c))...)

### Calculating the size of tiles

# If kernelsize-1 is the amount of padding, and s is the extra width of the tile along
# each axis, then the fraction of useful to total work is
# prod(s)/prod(s+kernelsize); this ratio is maximized if s is proportional to
# kernelsize.
function padded_tilesize{T}(::Type{T}, kernelsize::Dims)
    nd = max(1, sum(x->x>1, kernelsize))
    # isbits(T) || return map(zero, kernelsize)
    # don't be too minimalist on the cache-friendly dim (use at least 2 cachelines)
    dim1minlen = 2*cachelinesize÷sizeof(T)
    psz = (max(kernelsize[1], dim1minlen), tail(kernelsize)...)
    L = sizeof(T)*prod(psz)
    # try to stay in L1 cache, but in the end we want a reasonably
    # favorable work ratio. f is the constant of proportionality in
    #      s+kernelsize ∝ kernelsize
    f = max(floor(Int, (L1cachesize/(2*L))^(1/nd)), 2)
    return _padded_tilesize_scale(f, psz)
end

@noinline _padded_tilesize_scale(f, psz) = map(x->x <= 1 ? x : f*x, psz) # see #15276

### Tile shaping and coordinate transformation

immutable TileView{T,N,P} <: AbstractArray{T,N}
    view::OffsetArray{T,N,Array{T,N}}  # the currently-active view
    buf::Array{T,P}                    # the original backing buffer
end

"""
    TileView(a, inds::Indices) -> v

Return a buffer-view `v` whose indices match `inds`, using the array
`a` for storage. `inds` does not necessarily have to match the size of
`a` (which allows tiles to be of different sizes, e.g., at the edges).
"""
@inline function TileView(a::Array, inds::Indices)
    l = map(length, inds)
    tilev = OffsetArray(_tileview(a, l), inds)
    TileView(tilev, a)
end

"""
    TileView(T, inds::Indices) -> v

Return a TileView, allocating a new backing array of element type `T`
and size determined by `inds`.
"""
function TileView{T}(::Type{T}, inds::Indices)
    l = map(length, inds)
    TileView(Array{T}(l), inds)
end

TileView(v::TileView, inds::Indices) = TileView(v.buf, inds)

@inline function _tileview(a::Array, l::Dims)
    if size(a) == l
        return a
    else
        # returning a SubArray would not be type-stable, we must return an Array
        prod(l) > length(a) && throw(DimensionMismatch("array of size $(size(a)) is not adequate for a tile of size $l"))
        return unsafe_wrap(Array, pointer(a), l)
    end
end

Base.indices(v::TileView) = indices(v.view)

@inline Base.getindex{T,N}(v::TileView{T,N}, I::Vararg{Int,N}) = v.view[I...]

@inline Base.setindex!{T,N}(v::TileView{T,N}, val, I::Vararg{Int,N}) = v.view[I...] = val

@inline OffsetArrays.unsafe_getindex(v::TileView, I...) = OffsetArrays.unsafe_getindex(v.view, I...)

@inline OffsetArrays.unsafe_setindex!(v::TileView, val, I...) = OffsetArrays.unsafe_setindex!(v.view, val, I...)

end # module
