__precompile__()

module TiledIteration

using OffsetArrays
using Base: tail, Indices

export TileIterator, padded_tilesize

const L1cachesize = 2^15
const cachelinesize = 64

# If sz-1 is the amount of padding, and s is the extra width of the tile along
# each axis, then the fraction of useful to total work is
# prod(s)/prod(s+sz); this ratio is maximized if s is proportional to
# sz.
function padded_tilesize{T}(::Type{T}, sz::Dims)
    nd = max(1, sum(x->x>1, sz))
    # isbits(T) || return map(zero, sz)
    # don't be too minimalist on the cache-friendly dim (use at least 2 cachelines)
    dim1minlen = 2*cachelinesize÷sizeof(T)
    psz = (max(sz[1], dim1minlen), tail(sz)...)
    L = sizeof(T)*prod(psz)
    # try to stay in L1 cache, but in the end we want a reasonably
    # favorable work ratio. f is the constant of proportionality in s+sz ∝ sz
    f = max(floor(Int, (L1cachesize/(2*L))^(1/nd)), 2)
    return _padded_tilesize_scale(f, psz)
end

@noinline _padded_tilesize_scale(f, psz) = map(x->x <= 1 ? x : f*x, psz) # see #15276

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

end # module
