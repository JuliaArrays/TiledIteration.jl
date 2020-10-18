################################################################################
##### CoveredRange
################################################################################

# a range covered by subranges
struct CoveredRange{R,S} <: AbstractVector{UnitRange{Int}}
    offsets::R
    stopping::S
end

struct FixedLength
    length::Int
end

struct LengthAtMost{N <: Union{Int, Nothing}}
    maxlength::Int
    maxstop::N
end

function compute_stop(offset, stopping::FixedLength)
    return offset+stopping.length
end
function compute_stop(offset, stopping::LengthAtMost)
    return min(offset+stopping.maxlength, stopping.maxstop)
end

compute_range(offset, stopping)::UnitRange{Int} = (offset+1):compute_stop(offset, stopping)

Base.size(o::CoveredRange,args...) = size(o.offsets, args...)
Base.@propagate_inbounds function Base.getindex(o::CoveredRange, inds...)
    offset = o.offsets[inds...]
    return compute_range(offset, o.stopping)
end

################################################################################
##### RoundedRange
################################################################################
struct RoundedRange{R} <: AbstractVector{Int}
    range::R
end

Base.@propagate_inbounds function Base.getindex(r::RoundedRange, i)
    rough = r.range[i]
    return round(Int, rough)
end

function roundedrange(start; stop, length)
    inner = LinRange(start, stop, length)
    return RoundedRange(inner)
end

Base.size(o::RoundedRange, args...) = size(o.range, args...)

################################################################################
##### TileIterator
################################################################################
struct TileIterator{N,C} <: AbstractArray{NTuple{N, UnitRange{Int}}, N}
    covers1d::C
end

function TileIterator(covers1d::NTuple{N, AbstractVector{UnitRange{Int}}}) where {N}
    C = typeof(covers1d)
    return TileIterator{N, C}(covers1d)
end

function TileIterator(axes::Indices{N}, tilesize::Dims{N}) where {N}
    tileiterator(axes, RelaxLastTile(tilesize))::TileIterator{N}
end

# strategies
export RelaxStride
"""
    RelaxStride(tilesize)

Tiling strategy, that guarantees each tile of size `tilesize`.
If the needed tiles will slightly overlap, to cover everything.

# Examples
```jldoctest
julia> using TiledIteration

julia> collect(tileiterator((1:4,), RelaxStride((2,))))
2-element Array{Tuple{UnitRange{Int64}},1}:
 (1:2,)
 (3:4,)

julia> collect(tileiterator((1:4,), RelaxStride((3,))))
2-element Array{Tuple{UnitRange{Int64}},1}:
 (1:3,)
 (2:4,)
```

See also [`tileiterator`](@ref).
"""
struct RelaxStride{N}
    tilesize::Dims{N}
end

export RelaxLastTile

"""
    RelaxLastTile(tilesize)

Tiling strategy, that premits the size of the last tiles along each dimension to be smaller
then `tilesize` if needed. All other tiles are of size `tilesize`.

# Examples
```jldoctest
julia> using TiledIteration

julia> collect(tileiterator((1:4,), RelaxLastTile((2,))))
2-element Array{Tuple{UnitRange{Int64}},1}:
 (1:2,)
 (3:4,)

julia> collect(tileiterator((1:7,), RelaxLastTile((2,))))
4-element Array{Tuple{UnitRange{Int64}},1}:
 (1:2,)
 (3:4,)
 (5:6,)
 (7:7,)
```

See also [`tileiterator`](@ref).
"""
struct RelaxLastTile{N}
    tilesize::Dims{N}
end

"""
    split(strategy)

Split an N dimensional strategy into an NTuple of 1 dimensional strategies.
"""
function split end

function split(strategy::RelaxStride)
    map(strategy.tilesize) do s
        RelaxStride((s,))
    end
end

function split(strategy::RelaxLastTile)
    map(strategy.tilesize) do s
        RelaxLastTile((s,))
    end
end

"""
    titr = tileiterator(axes::NTuple{N, AbstractUnitRange}, strategy)

Decompose `axes` into an iterator `titr` of smaller axes.

```jldoctest
julia> using TiledIteration

julia> collect(tileiterator((1:3, 0:5), RelaxLastTile((2, 3))))
2×2 Array{Tuple{UnitRange{Int64},UnitRange{Int64}},2}:
 (1:2, 0:2)  (1:2, 3:5)
 (3:3, 0:2)  (3:3, 3:5)

julia> collect(tileiterator((1:3, 0:5), RelaxStride((2, 3))))
2×2 Array{Tuple{UnitRange{Int64},UnitRange{Int64}},2}:
 (1:2, 0:2)  (1:2, 3:5)
 (2:3, 0:2)  (2:3, 3:5)
```
"""
function tileiterator(axes, strategy)
    covers1d = map(cover1d, axes, split(strategy))
    return TileIterator(covers1d)
end

function cover1d(ax, strategy::RelaxStride{1})::CoveredRange
    tilelen = stride = first(strategy.tilesize)
    firstoffset = first(ax)-1
    lastoffset = last(ax) - tilelen
    stepcount = ceil(Int, (lastoffset - firstoffset) / stride) + 1
    offsets = roundedrange(firstoffset, stop=lastoffset, length=stepcount)
    @assert last(ax) - tilelen <= last(offsets) <= last(ax)

    stopping = FixedLength(tilelen)
    return CoveredRange(offsets, stopping)
end

function cover1d(ax, strategy::RelaxLastTile{1})::CoveredRange
    tilelen = stride = first(strategy.tilesize)
    maxstop = last(ax)
    stopping = LengthAtMost(tilelen, maxstop)

    lo = first(ax)
    hi = last(ax)
    stepcount = if tilelen <= stride
        floor(Int, (hi - lo) / stride) + 1
    else
        ceil(Int, (hi + 1 - lo - tilelen) / stride) + 1
    end
    firstoffset = lo - 1
    offsets = range(firstoffset, step=tilelen, length=stepcount)
    return CoveredRange(offsets, stopping)
end

Base.@propagate_inbounds function Base.getindex(o::TileIterator, inds...)
    cis = CartesianIndices(o)[inds...]
    o[cis]
end
Base.@propagate_inbounds function Base.getindex(o::TileIterator, ci::CartesianIndex)
    map(getindex, o.covers1d, Tuple(ci))
end

Base.size(o::TileIterator) = map(length, o.covers1d)
Base.IndexStyle(o::TileIterator) = IndexCartesian()
