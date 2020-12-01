### TiledUnitRange

struct TiledUnitRange{T, R} <: AbstractUnitRange{T}
    parent::R
    tilelength::T
    tilestride::T
    length::T

    function TiledUnitRange{T, R}(parent::R, tilelength::T, tilestride::T) where {T, R}
        n = _length(last(parent), tilelength, tilestride)
        new{T, R}(parent, tilelength, tilestride, n)
    end
end
TiledUnitRange(parent::R, l::T, s::T) where {T, R} = TiledUnitRange{T, R}(parent, l, s)
TiledUnitRange(parent, tilelength) = TiledUnitRange(parent, tilelength, tilelength)

_length(stop::T, n, Δ) where T = ceil(T, (stop - n)/Δ) + 1
_length(stop::CartesianIndex{1}, n, Δ) = _length(first(stop.I), n, Δ)

tilelength(r::TiledUnitRange{T, R}) where {T, R<:CartesianIndices} = CartesianIndex(r.tilelength)
tilelength(r::TiledUnitRange) = r.tilelength

tilestride(r::TiledUnitRange{T, R}) where {T, R<:CartesianIndices} = CartesianIndex(r.tilestride)
tilestride(r::TiledUnitRange) = r.tilestride

Base.length(r::TiledUnitRange) = r.length

function Base.first(r::TiledUnitRange{T, R}) where {T, R}
    start = first(r.parent)
    stop = min(start+tilelength(r), last(r.parent))
    return start:stop
end

function Base.last(r::TiledUnitRange{T, R}) where {T, R}
    start = first(r.parent) + (length(r)-1) * tilestride(r)
    stop = min(start+tilelength(r), last(r.parent))
    return start:stop
end

function Base.getindex(r::TiledUnitRange{T, R}, i::Int) where {T, R}
    start = first(r.parent) + (i-1)*tilestride(r)
    stop = min(start+tilelength(r), last(r.parent))
    return start:stop
end

Base.show(io::IO, r::TiledUnitRange) = print(io, "TiledUnitRange(", r.parent, ",", r.tilelength, ",", r.tilestride, ")")
function Base.show(io::IO, r::TiledUnitRange{T, R}) where {T, R<:CartesianIndices}
    print(io, "TiledUnitRange(CartesianIndices(", r.parent.indices, "),", r.tilelength, ",", r.tilestride, ")")
end

### TiledIndices

struct TiledIndices{N, T, R} <: AbstractArray{R, N}
    indices::NTuple{N, TiledUnitRange{T, R}}
end

TiledIndices(indices, tilelength) = TiledIndices(indices, tilelength, tilelength)
TiledIndices(indices, tilelength, tilestride) =
    TiledIndices(map(TiledUnitRange, indices, tilelength, tilestride))
TiledIndices(indices::CartesianIndices, tilelength, tilestride) =
    TiledIndices(map(CartesianIndices, indices.indices), tilelength, tilestride)

Base.size(iter::TiledIndices) = map(length, iter.indices)
Base.@propagate_inbounds Base.getindex(iter::TiledIndices{N}, inds::Vararg{Int, N}) where N = map(getindex, iter.indices, inds)
Base.@propagate_inbounds function Base.getindex(
        iter::TiledIndices{N, T, R},
        inds::Vararg{Int, N}) where {N, T, R<:CartesianIndices}
    tile = map(getindex, iter.indices, inds)
    # reformulate into CartesianIndices{N}
    CartesianIndices(mapreduce(I->I.indices, (i,j)->(i..., j...), tile))
end


