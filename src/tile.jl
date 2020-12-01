### TiledUnitRange

struct TiledUnitRange{T, R} <: AbstractUnitRange{T}
    parent::R
    tilelength::T
    tilestride::T
    length::T

    function TiledUnitRange{T, R}(parent::R, tilelength::T, tilestride::T) where {T, R<:AbstractUnitRange{T}}
        n = ceil(T, (last(parent) - tilelength)/tilestride) + 1
        new{T, R}(parent, tilelength, tilestride, n)
    end
end
TiledUnitRange(parent::R, l::T, s::T) where {T, R} = TiledUnitRange{T, R}(parent, l, s)
TiledUnitRange(parent, tilelength) = TiledUnitRange(parent, tilelength, tilelength)

Base.length(r::TiledUnitRange) = r.length

function Base.first(r::TiledUnitRange{T, R}) where {T, R}
    start = first(r.parent)
    return R(start, min(start+r.length, last(r.parent)))
end

function Base.last(r::TiledUnitRange{T, R}) where {T, R}
    start = first(r.parent) + (length(r)-1) * r.tilestride
    return R(start, min(start+r.length, last(r.parent)))
end

function Base.getindex(r::TiledUnitRange{T, R}, i::Int) where {T, R}
    start = first(r.parent) + (i-1)*r.tilestride
    return R(start, min(start+r.length, last(r.parent)))
end

Base.show(io::IO, r::TiledUnitRange) = print(io, "TiledUnitRange(", r.parent, ",", r.tilelength, ",", r.tilestride, ")")

### TiledIndices

struct TiledIndices{N, T, R} <: AbstractArray{R, N}
    indices::NTuple{N, TiledUnitRange{T, R}}
end

TiledIndices(indices, tilelength) = TiledIndices(indices, tilelength, tilelength)
TiledIndices(indices, tilelength, tilestride) = TiledIndices(map(TiledUnitRange, indices, tilelength, tilestride))

Base.size(iter::TiledIndices) = map(length, iter.indices)
Base.@propagate_inbounds Base.getindex(iter::TiledIndices{N}, inds::Vararg{Int, N}) where N = map(getindex, iter.indices, inds)
