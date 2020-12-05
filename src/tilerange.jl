# AbstractTileRange Protocols
#
# The parent range `R` is supposed to be an array of scalar indices, including:
#   - ranges
#   - vectors of integers
#   - CartesianIndices{1}
#
# A tile range generated from `R` is a iterator whose elements are also an array of scalar indices.
# It's an indices-specific block array.
#
# Reference: https://docs.julialang.org/en/v1.5/manual/arrays/#man-supported-index-types
#
#
# | scalar index      |   indices(tile)      |
# |-------------------|-----------------     |
# | range/vector      |   AbstractTileRange  |
# | CartesianIndices  |   TileIndices        |
#
# AbstractTileStrategy serves as an adapter for `AbstractTileRange` and `TileIndices`


abstract type AbstractTileRange{R} <: AbstractArray{R, 1} end
abstract type AbstractTileStrategy end

const Range1 = Union{AbstractUnitRange, AbstractVector}

### FixedTileRange and FixedTile

"""
    FixedTileRange(r, n, [Δ=n]; keep_last=true)

Construct a sliding tile along range `r` with fixed sliding stride `Δ` and tile length `n`.

# Arguments

- `r`: a range, `CartesianIndices` or `Vector`
- `n::Integer`: The length of each tile. If keyword `keep_last=true`, the last tile length might be
  less than `n`.
- `Δ::Union{Integer, CartesianIndex{1}}=n`: The sliding stride `Δ` is defined as `first(r[n]) -
  first(r[n-1])`. Using a stride `Δ<n` means there are overlaps between each adjacent tile. `Δ` can
  also be `CartesianIndex{1}`.
- `keep_last=true` (keyword): this keyword affects the cases when the last tile has few elements
  than `n`, in which case, `true`/`false` tells `FixedTileRange` to keep/discard the last tile.

# Examples

```jldoctest
julia> FixedTileRange(2:10, 3)
3-element FixedTileRange{UnitRange{Int64}, Int64, Val{true}, UnitRange{Int64}}:
 2:4
 5:7
 8:10

julia> FixedTileRange(1:10, 4)
3-element FixedTileRange{UnitRange{Int64}, Int64, Val{true}, UnitRange{Int64}}:
 1:4
 5:8
 9:10

julia> FixedTileRange(1:10, 4, 2)
 4-element FixedTileRange{UnitRange{Int64}, Int64, Val{true}, UnitRange{Int64}}:
  1:4
  3:6
  5:8
  7:10

julia> FixedTileRange(1:10, 4; keep_last=false)
 2-element FixedTileRange{UnitRange{Int64}, Int64, Val{false}, UnitRange{Int64}}:
  1:4
  5:8
```

Besides an `AbstractUnitRange`, the input range `r` can also be a `CartesianIndices{1}` or more
generally, an `AbstractVector{<:Integer}`:

```jldoctest
julia> FixedTileRange(CartesianIndices((1:10, )), 4)
 3-element FixedTileRange{CartesianIndices{1, Tuple{UnitRange{Int64}}}, Int64, Val{true}, CartesianIndices{1, Tuple{UnitRange{Int64}}}}:
  [CartesianIndex(1,), CartesianIndex(2,), CartesianIndex(3,), CartesianIndex(4,)]
  [CartesianIndex(5,), CartesianIndex(6,), CartesianIndex(7,), CartesianIndex(8,)]
  [CartesianIndex(9,), CartesianIndex(10,)]
```

!!! warning It usually has bad indexing performance if `r` is not lazily evaluated. For example,
    `FixedTileRange(collect(1:10), 4)` creates a new `Vector` of length `4` everytime when
    `getindex` is called.
"""
struct FixedTileRange{R, T, RP} <: AbstractTileRange{R}
    parent::RP
    n::T
    Δ::T
    keep_last::Bool

    # keep `length` information to avoid unnecessary calculation and thus is more performant
    length::T

    function FixedTileRange(parent::R, n::T, Δ; keep_last::Bool=true) where {R<:Range1, T}
        _length = _fixedtile_length(first(parent), last(parent), n, Δ, keep_last)
        new{_eltype(R), T, R}(parent, n, Δ, keep_last, _length)
    end
end
FixedTileRange(r::Range1, n::Integer; kwargs...) = FixedTileRange(r, n, n; kwargs...)

_eltype(::Type{R}) where R<:AbstractUnitRange = UnitRange{eltype(R)}
_eltype(::Type{R}) where R<:AbstractVector = R # this includes CartesianIndices{1}

_int(x::Integer) = x
_int(x::CartesianIndex{1}) = first(x.I)

function _fixedtile_length(start::T, stop::T, n, Δ, keep_last) where T<:Integer
    _round = keep_last ? ceil : floor
    return _round(T, (stop - n - start + 1)/_int(Δ)) + 1
end
_fixedtile_length(start::T, stop::T, n, Δ, keep_last) where T<:CartesianIndex{1} =
    _fixedtile_length(_int(start), _int(stop), n, Δ, keep_last)

"Get the length of the tile. The last tile might has few elements than this."
tilelength(r::FixedTileRange{<:CartesianIndices{1}}) = CartesianIndex{1}(r.n)
tilelength(r::FixedTileRange) = r.n

"Get the stride between adjacent tiles."
tilestride(r::FixedTileRange{<:CartesianIndices{1}}) = CartesianIndex{1}(r.Δ)
tilestride(r::FixedTileRange) = r.Δ

Base.size(r::FixedTileRange) = (r.length, )
Base.length(r::FixedTileRange) = r.length

Base.@propagate_inbounds function Base.getindex(r::FixedTileRange{R, T}, i::Int) where {R, T}
    @boundscheck checkbounds(r, i)
    start = first(r.parent) + (i-1)*tilestride(r)
    stop = min(start+tilelength(r)-oneunit(eltype(R)), last(r.parent))
    return convert(R, start:stop)
end


"""
    FixedTile(sz, Δ; keep_last=true)

A tile strategy that you can use to construct [`TileIndices`](@ref) using [`FixedTileRange`](@ref).

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
julia> TileIndices((1:4, 0:5), FixedTile((3, 4), (2, 3)))
 2×2 TileIndices{Tuple{UnitRange{Int64}, UnitRange{Int64}}, 2, FixedTileRange{UnitRange{Int64}, Int64, Val{true}, UnitRange{Int64}}}:
  (1:3, 0:3)  (1:3, 3:5)
  (3:4, 0:3)  (3:4, 3:5)

julia> TileIndices((1:4, 0:5), FixedTile((3, 4), (2, 3); keep_last=false))
1×1 TileIndices{Tuple{UnitRange{Int64}, UnitRange{Int64}}, 2, FixedTileRange{UnitRange{Int64}, Int64, Val{false}, UnitRange{Int64}}}:
 (1:3, 0:3)
```

When `sz` and `Δ` are scalars, it affects each dimension equivalently.

```jldoctest
julia> TileIndices((1:4, 0:5), FixedTile(3, 2))
2×3 TileIndices{Tuple{UnitRange{Int64}, UnitRange{Int64}}, 2, FixedTileRange{UnitRange{Int64}, Int64, Val{true}, UnitRange{Int64}}}:
 (1:3, 0:2)  (1:3, 2:4)  (1:3, 4:5)
 (3:4, 0:2)  (3:4, 2:4)  (3:4, 4:5)

julia> TileIndices((1:4, 0:5), FixedTile(3, 2; keep_last=false))
1×2 TileIndices{Tuple{UnitRange{Int64}, UnitRange{Int64}}, 2, FixedTileRange{UnitRange{Int64}, Int64, Val{false}, UnitRange{Int64}}}:
 (1:3, 0:2)  (1:3, 2:4)
```
"""
struct FixedTile{N, T} <: AbstractTileStrategy
    size::T
    Δ::T
    keep_last::Bool
end

FixedTile(sz::T, Δ=sz; keep_last=true) where T<:Integer = FixedTile{0, T}(sz, Δ, keep_last)
FixedTile(sz::T, Δ=sz; keep_last=true) where T = FixedTile{length(sz), T}(sz, Δ, keep_last)

(S::FixedTile{0})(r::Range1) = FixedTileRange(r, S.size, S.Δ; keep_last=S.keep_last)
(S::FixedTile{0})(r::CartesianIndices{1}) = FixedTileRange(r, S.size, S.Δ; keep_last=S.keep_last)
(S::FixedTile{0})(indices) = 
    map(r->FixedTileRange(r, S.size, S.Δ; keep_last=S.keep_last), indices)

(S::FixedTile{N})(indices) where N =
    map((args...)->FixedTileRange(args...; keep_last=S.keep_last), indices, S.size, S.Δ)
# ambiguity patch
(S::FixedTile{0})(indices::CartesianIndices{N}) where N =
    S(map(x->CartesianIndices((x, )), indices.indices))
(S::FixedTile{N})(indices::CartesianIndices{N}) where N =
    S(map(x->CartesianIndices((x, )), indices.indices))
