using TiledIteration, OffsetArrays
using Base.Test

@testset "tiled iteration" begin
    sz = (3,5)
    for i1 = -3:2, i2 = -1:5
        for l1 = 2:13, l2 = 1:16
            inds = (i1:i1+l1-1, i2:i2+l2-1)
            A = fill!(OffsetArray{Int}(inds), 0)
            k = 0
            for tileinds in TileIterator(inds, sz)
                tile = A[tileinds...]
                @test !isempty(tile)
                @test all(tile .== 0)
                A[tileinds...] = (k+=1)
            end
            @test minimum(A) == 1
            @test eltype(collect(TileIterator(inds, sz))) == Tuple{UnitRange{Int}, UnitRange{Int}}
        end
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
        indsA = indices(A)
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
        indsA = indices(A)
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
    A = rand(Asz)
    snt = sumtiles(A, tilesz)
    st = sumtiles_threaded(A, tilesz)
    @test snt == st
    @test all(st .> 0)
end

@testset "TileBuffer" begin
    a = fill!(Array{Int}(5,5), 0)
    v = @inferred(TileBuffer(a, (2:3, 97:99)))
    @test ndims(v) == 2
    @test eltype(v) == Int
    @test indices(v) == (2:3, 97:99)
    v[2,97] = 1
    @test a[1] == 1
    p = pointer(v)
    @test parent(v) === v.buf
    v = @inferred(TileBuffer(v, (-1:1, 1:1)))
    @test pointer(v) == p
    @test indices(v) == (-1:1, 1:1)
    @test v[-1,1] == 1
    @test v[0,1] == v[1,1] == 0
    v[0,1] = 2
    @test a[2] == 2
    v = @inferred(TileBuffer(v, (0:23,)))
    @test pointer(v) == p
    @test eltype(v) == Int
    @test ndims(v) == 1
    @test indices(v) == (0:23,)
    @test v[1] == 2
    @test_throws BoundsError v[24]
    if Base.JLOptions().can_inline == 1
        @unsafe v[24] = 7
        @test a[25] == 7
        @test (@unsafe v[24]) == 7
    end
    @test_throws DimensionMismatch TileBuffer(a, (1:6,1:5))
    @test_throws DimensionMismatch TileBuffer(v, (1:5,1:6))
    @test_throws DimensionMismatch TileBuffer(a, (0:25,))
    b = TileBuffer(Float64, (1:16,1:4))
    @test eltype(b) == Float64
    @test indices(b) == (1:16,1:4)
    @test pointer(b) != p
    p = pointer(b)
    b = TileBuffer(b, (17:32,1:4))
    @test pointer(b) == p
    @unsafe begin
        b[17,2] = 5.2
        @test b[17,2] == 5.2
    end
end

nothing
