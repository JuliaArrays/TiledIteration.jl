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

export Balanced
struct Balanced{V}
    value::V
end

function TileIterator(covers1d::NTuple{N, AbstractVector{UnitRange{Int}}}) where {N}
    C = typeof(covers1d)
    return TileIterator{N, C}(covers1d)
end

function compute_stoppings(axes, tilesize::Fixed)
    map(FixedLength, tilesize.value)
end

function compute_stoppings(axes, tilesize::Dims)
    map(axes, tilesize) do ax, maxlen
        maxstop = last(ax)
        LengthAtMost(maxlen, maxstop)
    end
end

function compute_offsetss(axes, tilesize::Dims)
    map(axes, tilesize) do ax, tilelen
        firstoffset = first(ax) - 1
        stepcount = ceil(Int, (last(ax) - first(ax)) / tilelen)
        if first(ax) + stepcount * tilelen <= last(ax)
            stepcount += 1
        end
        r = range(firstoffset, step=tilelen, length=stepcount)
        @assert last(ax) - tilelen <= last(r) <= last(ax)
        r
    end
end

function compute_offsetss(axes, tilesize::Fixed)
    map(axes, tilesize.value) do ax, tilelen
        firstoffset = first(ax)-1
        lastoffset = last(ax) - tilelen
        stepcount = ceil(Int, (lastoffset - firstoffset) / tilelen) + 1
        r = quantizedrange(firstoffset, stop=lastoffset, length=stepcount)
        @assert last(ax) - tilelen <= last(r) <= last(ax)
        r
    end
end

function TileIterator(axes::Indices{N}, tilesize::Dims{N}) where {N}
    TileIterator(axes,tilesize=tilesize)::TileIterator{N}
end

function check_axes_tilesize_stride(axes, tilesize, stride)
    if stride === tilesize === nothing
        msg = "At least one of `tilesize` or `stride` must be provided"
        throw(ArgumentError(msg))
    else

    end
end

function TileIterator(axes::Indices; tilesize=nothing, stride=nothing)
    if stride !== nothing
        throw(ArgumentError("stride not yet implemented"))
    end
    check_axes_tilesize_stride(axes, tilesize, stride)
    offsetss = compute_offsetss(axes, tilesize)
    stoppings = compute_stoppings(axes, tilesize)
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
