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
##### QuantizedRange
################################################################################
struct QuantizedRange{R} <: AbstractVector{Int}
    range::R
end

Base.@propagate_inbounds function Base.getindex(r::QuantizedRange, i)
    rough = r.range[i]
    return round(Int, rough)
end

quantizedrange(args...; kw...) = QuantizedRange(range(args...; kw...))

Base.size(o::QuantizedRange, args...) = size(o.range, args...)

################################################################################
##### TileIterator
################################################################################
struct TileIterator{N,C} <: AbstractArray{NTuple{N, UnitRange{Int}}, N}
    covers1d::C
end

export Fixed
struct Fixed{V}
    value::V
end
Fixed() = Fixed(nothing)
Fixed(o::Fixed) = o

export Balanced
struct Balanced{V}
    value::V
end
Balanced() = Balanced(nothing)
Balanced(o::Balanced) = o

function TileIterator(covers1d::NTuple{N, AbstractVector{UnitRange{Int}}}) where {N}
    C = typeof(covers1d)
    return TileIterator{N, C}(covers1d)
end

function compute_stoppings(axes, tilesize::Dims, stride::Balanced)
    map(FixedLength, tilesize)
end

function compute_stoppings(axes, tilesize::Dims, stride::Fixed)
    map(axes, tilesize) do ax, maxlen
        maxstop = last(ax)
        LengthAtMost(maxlen, maxstop)
    end
end

function compute_offsetss(axes, tilesize::Dims, stride::Fixed)
    map(axes, tilesize, stride.value) do ax, tilelen, step
        lo = first(ax)
        hi = last(ax)
        stepcount = if tilelen <= step
            floor(Int, (hi - lo) / step) + 1
        else
            ceil(Int, (hi + 1 - lo - tilelen) / step) + 1
        end
        firstoffset = lo - 1
        range(firstoffset, step=step, length=stepcount)
    end
end

function compute_offsetss(axes, tilesize::Dims, stride::Balanced)
    map(axes, tilesize, stride.value) do ax, tilelen, s
        firstoffset = first(ax)-1
        lastoffset = last(ax) - tilelen
        stepcount = ceil(Int, (lastoffset - firstoffset) / s) + 1
        r = quantizedrange(firstoffset, stop=lastoffset, length=stepcount)
        @assert last(ax) - tilelen <= last(r) <= last(ax)
        r
    end
end

function TileIterator(axes::Indices{N}, tilesize::Dims{N}) where {N}
    TileIterator(axes,tilesize=tilesize)::TileIterator{N}
end

resolve_stride(tilesize::Dims, stride::Nothing) = Fixed(tilesize)
resolve_stride(tilesize::Dims, stride::Balanced{Nothing}) = Balanced(tilesize)
resolve_stride(tilesize::Dims, stride::Fixed{Nothing}) = Fixed(tilesize)
resolve_stride(tilesize::Dims, stride::Union{Fixed, Balanced}) = stride
resolve_stride(tilesize::Dims, stride::Dims) = Fixed(stride)

function TileIterator(axes::Indices; tilesize, stride=nothing)
    stride = resolve_stride(tilesize, stride)
    offsetss = compute_offsetss(axes, tilesize, stride)
    stoppings = compute_stoppings(axes, tilesize, stride)
    covers1d = map(offsetss, stoppings) do offsets, stopping
        CoveredRange(offsets, stopping)
    end
    return TileIterator(covers1d)
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
