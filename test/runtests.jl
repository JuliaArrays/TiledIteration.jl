using TiledIteration, OffsetArrays
using Test

@testset "tiled iteration" begin
    sz = (3,5)
    for i1 = -3:2, i2 = -1:5
        for l1 = 2:13, l2 = 1:16
            inds = (i1:i1+l1-1, i2:i2+l2-1)
            A = fill!(OffsetArray{Int}(undef, inds), 0)
            k = 0
            for tileinds in TileIterator(inds, sz)
                tile = A[tileinds...]
                @test !isempty(tile)
                @test all(tile .== 0)
                A[tileinds...] .= (k+=1)
            end
            @test minimum(A) == 1
            @test eltype(collect(TileIterator(inds, sz))) == Tuple{UnitRange{Int}, UnitRange{Int}}
        end
    end
end

@testset "edge iteration" begin
    iter = EdgeIterator(CartesianIndices((-1:4,0:3)), CartesianIndices((1:3,1:2)))
    @test collect(iter) == [CartesianIndex((-1,0)),
                            CartesianIndex(( 0,0)),
                            CartesianIndex(( 1,0)),
                            CartesianIndex(( 2,0)),
                            CartesianIndex(( 3,0)),
                            CartesianIndex(( 4,0)),
                            CartesianIndex((-1,1)),
                            CartesianIndex(( 0,1)),
                            CartesianIndex(( 4,1)),
                            CartesianIndex((-1,2)),
                            CartesianIndex(( 0,2)),
                            CartesianIndex(( 4,2)),
                            CartesianIndex((-1,3)),
                            CartesianIndex(( 0,3)),
                            CartesianIndex(( 1,3)),
                            CartesianIndex(( 2,3)),
                            CartesianIndex(( 3,3)),
                            CartesianIndex(( 4,3))]
    iter = EdgeIterator((0:3,0:3), (1:3,1:2))
    @test collect(iter) == [CartesianIndex(( 0,0)),
                            CartesianIndex(( 1,0)),
                            CartesianIndex(( 2,0)),
                            CartesianIndex(( 3,0)),
                            CartesianIndex(( 0,1)),
                            CartesianIndex(( 0,2)),
                            CartesianIndex(( 0,3)),
                            CartesianIndex(( 1,3)),
                            CartesianIndex(( 2,3)),
                            CartesianIndex(( 3,3))]
    iter = EdgeIterator((1:4,0:3), (1:3,1:2))
    @test collect(iter) == [CartesianIndex(( 1,0)),
                            CartesianIndex(( 2,0)),
                            CartesianIndex(( 3,0)),
                            CartesianIndex(( 4,0)),
                            CartesianIndex(( 4,1)),
                            CartesianIndex(( 4,2)),
                            CartesianIndex(( 1,3)),
                            CartesianIndex(( 2,3)),
                            CartesianIndex(( 3,3)),
                            CartesianIndex(( 4,3))]
    iter = EdgeIterator((-1:4,1:3), (1:3,1:2))
    @test collect(iter) == [CartesianIndex((-1,1)),
                            CartesianIndex(( 0,1)),
                            CartesianIndex(( 4,1)),
                            CartesianIndex((-1,2)),
                            CartesianIndex(( 0,2)),
                            CartesianIndex(( 4,2)),
                            CartesianIndex((-1,3)),
                            CartesianIndex(( 0,3)),
                            CartesianIndex(( 1,3)),
                            CartesianIndex(( 2,3)),
                            CartesianIndex(( 3,3)),
                            CartesianIndex(( 4,3))]
    iter = EdgeIterator((-1:4,0:2), (1:3,1:2))
    @test collect(iter) == [CartesianIndex((-1,0)),
                            CartesianIndex(( 0,0)),
                            CartesianIndex(( 1,0)),
                            CartesianIndex(( 2,0)),
                            CartesianIndex(( 3,0)),
                            CartesianIndex(( 4,0)),
                            CartesianIndex((-1,1)),
                            CartesianIndex(( 0,1)),
                            CartesianIndex(( 4,1)),
                            CartesianIndex((-1,2)),
                            CartesianIndex(( 0,2)),
                            CartesianIndex(( 4,2))]
    iter = EdgeIterator((-1:4,1:2), (1:3,1:2))
    @test collect(iter) == [CartesianIndex((-1,1)),
                            CartesianIndex(( 0,1)),
                            CartesianIndex(( 4,1)),
                            CartesianIndex((-1,2)),
                            CartesianIndex(( 0,2)),
                            CartesianIndex(( 4,2))]
    iter = EdgeIterator((1:3,0:3), (1:3,1:2))
    @test collect(iter) == [CartesianIndex(( 1,0)),
                            CartesianIndex(( 2,0)),
                            CartesianIndex(( 3,0)),
                            CartesianIndex(( 1,3)),
                            CartesianIndex(( 2,3)),
                            CartesianIndex(( 3,3))]
    iter = EdgeIterator((1:3,1:2), (1:3,1:2))
    @test collect(iter) == []
    @test_throws DimensionMismatch EdgeIterator((1:3,1:1), (1:3,1:2))
    @test_throws DimensionMismatch EdgeIterator((1:3,1:2), (1:4,1:2))
    iter = EdgeIterator(CartesianIndices((0:4,)), CartesianIndices(1:3))
    @test collect(iter) == [CartesianIndex(0,),
                            CartesianIndex(4,)]
    iter = EdgeIterator((Base.OneTo(2),Base.OneTo(3)),(2:2,1:3))
    for it in iter
        @test it ∈ iter.outer
    end
    iter = EdgeIterator(CartesianIndices((1:4,)),CartesianIndices((2:4,)))
    for it in iter
        @test it ∈ iter.outer
    end
end

@testset "padded sizes" begin
    @test @inferred(padded_tilesize(UInt8, (1,))) == (2^14,)
    @test @inferred(padded_tilesize(UInt16, (1,))) == (2^13,)
    @test @inferred(padded_tilesize(Float64, (1,))) == (2^11,)
    @test @inferred(padded_tilesize(Float64, (1,1))) == (2^11, 1)
    @test @inferred(padded_tilesize(Float64, (2,1))) == (2^11, 1)
    shp = @inferred(padded_tilesize(Float64, (2,2)))
    @test all(x->x>2, shp)
    @test 2^13 <= prod(shp)*8 <= 2^14
    shp = @inferred(padded_tilesize(Float64, (3,3,3)))
    @test all(x->x>3, shp)
    @test 2^13 <= prod(shp)*8 <= 2^14
end

@testset "threads" begin
    function sumtiles(A, sz)
        indsA = axes(A)
        iter = TileIterator(indsA, sz)
        sums = zeros(eltype(A), length(iter))
        for (i,tileinds) in enumerate(iter)
            v = view(A, tileinds...)
            if sums[i] == 0
                sums[i] = sum(v)
            else
                sums[i] = -sum(v)
            end
        end
        sums
    end
    function sumtiles_threaded(A, sz)
        indsA = axes(A)
        iter = TileIterator(indsA, sz)
        allinds = collect(iter)
        sums = zeros(eltype(A), length(iter))
        Threads.@threads for i = 1:length(allinds)
            tileinds = allinds[i]
            v = view(A, tileinds...)
            if sums[i] == 0
                sums[i] = sum(v)
            else
                sums[i] = -sum(v)
            end
        end
        sums
    end

    Asz, tilesz = Base.JLOptions().can_inline == 1 ? ((1000,1000), (100,100)) : ((100,100), (10,10))
    A = rand(Float64, Asz)
    snt = sumtiles(A, tilesz)
    st = sumtiles_threaded(A, tilesz)
    @test snt == st
    @test all(st .> 0)
end

@testset "TileBuffer" begin
    a = fill!(Array{Int}(undef,5,5), 0)
    v = @inferred(TileBuffer(a, (2:3, 97:99)))
    @test ndims(v) == 2
    @test eltype(v) == Int
    @test axes(v) == (2:3, 97:99)
    v[2,97] = 1
    @test a[1] == 1
    p = pointer(v)
    @test parent(v) === v.buf
    v = @inferred(TileBuffer(v, (-1:1, 1:1)))
    @test pointer(v) == p
    @test axes(v) == (-1:1, 1:1)
    @test v[-1,1] == 1
    @test v[0,1] == v[1,1] == 0
    v[0,1] = 2
    @test a[2] == 2
    v = @inferred(TileBuffer(v, (0:23,)))
    @test pointer(v) == p
    @test eltype(v) == Int
    @test ndims(v) == 1
    @test axes(v) == (0:23,)
    @test v[1] == 2
    @test_throws BoundsError v[24]
    opts = Base.JLOptions()
    if opts.can_inline == 1 && opts.check_bounds == 0
        @inbounds v[24] = 7
        @test a[25] == 7
        @test (@inbounds v[24]) == 7
    end
    @test_throws DimensionMismatch TileBuffer(a, (1:6,1:5))
    @test_throws DimensionMismatch TileBuffer(v, (1:5,1:6))
    @test_throws DimensionMismatch TileBuffer(a, (0:25,))
    b = TileBuffer(Float64, (1:16,1:4))
    @test eltype(b) == Float64
    @test axes(b) == (1:16,1:4)
    @test pointer(b) != p
    p = pointer(b)
    b = TileBuffer(b, (17:32,1:4))
    @test pointer(b) == p
    @inbounds begin
        b[17,2] = 5.2
        @test b[17,2] == 5.2
    end
end

nothing
