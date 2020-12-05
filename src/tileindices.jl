### TileIndices

struct TileIndices{T, N, R<:AbstractTileRange} <: AbstractArray{T, N}
    indices::NTuple{N, R}
    function TileIndices(indices::NTuple{N, <:AbstractTileRange{T}}) where {N, T}
        new{NTuple{N, T}, N, eltype(indices)}(indices)
    end
end

"""
    TileIndices(indices, sz, Δ=sz; keep_last=true)

Construct a sliding tile along axes `r` with fixed sliding strides `Δ` and tile size `sz`.

# Arguments

- `sz`: The size of each tile. If keyword `keep_last=true`, the last tile size might be smaller than
  `sz`.
- `Δ=sz`: For each dimension `i` and `r = indices[i]`, the sliding stride `Δ[i]` is defined as
  `first(r[n]) - first(r[n-1])`. Using a stride `Δ[i] < sz[i]` means there are overlaps between each
  adjacent tile along this dimension.
- `keep_last=true` (keyword): this keyword affects the cases when the last tile size is smaller
  than `sz`, in which case, `true`/`false` tells `TileIndices` to keep/discard the last tile.

# Examples

```jldoctest
julia> TileIndices((1:4, 0:5), (3, 4), (2, 3))
 2×2 TileIndices{Tuple{UnitRange{Int64}, UnitRange{Int64}}, 2, FixedTileRange{UnitRange{Int64}, Int64, Val{true}, UnitRange{Int64}}}:
  (1:3, 0:3)  (1:3, 3:5)
  (3:4, 0:3)  (3:4, 3:5)

julia> TileIndices((1:4, 0:5), (3, 4), (2, 3); keep_last=false)
1×1 TileIndices{Tuple{UnitRange{Int64}, UnitRange{Int64}}, 2, FixedTileRange{UnitRange{Int64}, Int64, Val{false}, UnitRange{Int64}}}:
 (1:3, 0:3)
```

When `sz` and `Δ` are scalars, it affects each dimension equivalently.

```jldoctest
julia> TileIndices((1:4, 0:5), 3, 2)
2×3 TileIndices{Tuple{UnitRange{Int64}, UnitRange{Int64}}, 2, FixedTileRange{UnitRange{Int64}, Int64, Val{true}, UnitRange{Int64}}}:
 (1:3, 0:2)  (1:3, 2:4)  (1:3, 4:5)
 (3:4, 0:2)  (3:4, 2:4)  (3:4, 4:5)

julia> TileIndices((1:4, 0:5), 3, 2; keep_last=false)
1×2 TileIndices{Tuple{UnitRange{Int64}, UnitRange{Int64}}, 2, FixedTileRange{UnitRange{Int64}, Int64, Val{false}, UnitRange{Int64}}}:
 (1:3, 0:2)  (1:3, 2:4)
```

!!! note
    This method is equivalent to `TileIndices(indices, FixedTile(sz, Δ; keep_last=keep_last))`.
"""
TileIndices(indices, n, Δ=n; kwargs...) = TileIndices(indices, FixedTile(n, Δ; kwargs...))


"""
    TileIndices(indices, s::AbstractTileStrategy)

Construct a sliding tile along axes `r` using provided tile strategy `s`.

Currently available strategies are:

- [`FixedTile`](@ref): each tile is of the same fixed size (except the last one).

For usage examples, please refer to the docstring of each tile strategy.
"""
TileIndices(indices, s::AbstractTileStrategy) = TileIndices(s(indices))

Base.size(iter::TileIndices) = map(length, iter.indices)

Base.@propagate_inbounds function Base.getindex(
        iter::TileIndices{T, N},
        inds::Vararg{Int, N}) where {N, T}
    map(getindex, iter.indices, inds)
end
Base.@propagate_inbounds function Base.getindex(
        iter::TileIndices{T, N},
        inds::Vararg{Int, N}) where {N, T<:NTuple{N, CartesianIndices}}
    tile = map(getindex, iter.indices, inds)
    # reformulate into CartesianIndices{N}
    CartesianIndices(mapreduce(I->I.indices, (i,j)->(i..., j...), tile))
end
