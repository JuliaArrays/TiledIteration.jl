@testset "Fixed Tile" begin
    @testset "FixedTileRange" begin
        function test_iteration(r)
            for (i, v) in enumerate(r)
                @test v == r[i]
            end
            @test collect(r) == r

            rst = zero(first(r.parent))
            for v in r
                rst += sum(v)
            end
            @test mapreduce(sum, +, r) == rst
        end

        @testset "ranges" begin
            for r0 in [
                2:10,
                UInt8(2):UInt8(10),
                Base.IdentityUnitRange(2:10),
                OffsetArrays.IdOffsetRange(2:10),
                OffsetArrays.IdOffsetRange(Base.OneTo(9), 1)
            ]
                # keep the last tile
                r = @inferred FixedTileRange(r0, 4)
                @test r == FixedTileRange(r0, 4; keep_last=true) == FixedTileRange(r0, 4, 4)
                @test r isa AbstractArray
                @test eltype(r) == UnitRange{eltype(r0)}
                @test length(r) == 3
                @test first(r) == 2:5
                @test last(r) == 10:10
                test_iteration(r)
                @test all(map(x->length(x)==4, r[1:end-1]))
                @test length(r[end]) <= 4

                # different stride
                r = @inferred FixedTileRange(r0, 4, 2)
                @test r == FixedTileRange(r0, 4, 2; keep_last=true)
                @test r isa AbstractArray
                @test eltype(r) == UnitRange{eltype(r0)}
                @test length(r) == 4
                @test first(r) == 2:5
                @test last(r) == 8:10
                test_iteration(r)
                @test all(map(x->length(x)==4, r[1:end-1]))
                @test length(r[end]) <= 4

                # discard the last tile
                r = @inferred FixedTileRange(r0, 4, keep_last=false)
                @test r isa AbstractArray
                @test eltype(r) == UnitRange{eltype(r0)}
                @test length(r) == 2
                @test first(r) == 2:5
                @test last(r) == 6:9
                test_iteration(r)
                @test all(map(x->length(x)==4, r))
            end

            # FixedTileRange only works for 1d case
            @test_throws MethodError FixedTileRange((2:10, 2:10), (4, 4), (2, 2))        
        end

        @testset "CartesianIndices" begin
            r0 = CartesianIndex(2):CartesianIndex(10)

            # keep the last tile
            r = @inferred FixedTileRange(r0, 4)
            @test r == FixedTileRange(r0, 4; keep_last=true) == FixedTileRange(r0, 4, 4)
            @test r isa AbstractArray
            @test eltype(r) == typeof(r0)
            @test length(r) == 3
            @test first(r) == CartesianIndex(2):CartesianIndex(5)
            @test last(r) == CartesianIndex(10):CartesianIndex(10)
            test_iteration(r)
            @test all(map(x->length(x)==4, r[1:end-1]))
            @test length(r[end]) <= 4

            # `Δ` can also be `CartesianIndex` when `r` is a `CartesianIndices`
            @test FixedTileRange(r0, 4, 2) == FixedTileRange(r0, 4, CartesianIndex(2))

            # different stride
            r = @inferred FixedTileRange(r0, 4, 2)
            @test r == FixedTileRange(r0, 4, 2; keep_last=true)
            @test r isa AbstractArray
            @test eltype(r) == typeof(r0)
            @test length(r) == 4
            @test first(r) == CartesianIndex(2):CartesianIndex(5)
            @test last(r) == CartesianIndex(8):CartesianIndex(10)
            test_iteration(r)
            @test all(map(x->length(x)==4, r[1:end-1]))
            @test length(r[end]) <= 4

            # discard the last tile
            r = @inferred FixedTileRange(r0, 4, keep_last=false)
            @test r isa AbstractArray
            @test eltype(r) == typeof(r0)
            @test length(r) == 2
            @test first(r) == CartesianIndex(2):CartesianIndex(5)
            @test last(r) == CartesianIndex(6):CartesianIndex(9)
            # test_iteration(r)
            @test all(map(x->length(x)==4, r))
        end

        @testset "AbstractVector" begin
            # AbstractVector can be terrible for performance if it allocates memory.

            for r0 in [
                collect(2:10),
                collect(UInt8(2):UInt8(10)),
                # Broken: https://github.com/JuliaArrays/OffsetArrays.jl/issues/171
                # OffsetArray(collect(2:10), -1)
            ]
                # keep the last tile
                r = @inferred FixedTileRange(r0, 4)
                @test r == FixedTileRange(r0, 4; keep_last=true) == FixedTileRange(r0, 4, 4)
                @test r isa AbstractArray
                @test eltype(r) == Vector{eltype(r0)}
                @test length(r) == 3
                @test first(r) == collect(2:5)
                @test last(r) == collect(10:10)
                test_iteration(r)
                @test all(map(x->length(x)==4, r[1:end-1]))
                @test length(r[end]) <= 4

                # different stride
                r = @inferred FixedTileRange(r0, 4, 2)
                @test r == FixedTileRange(r0, 4, 2; keep_last=true)
                @test r isa AbstractArray
                @test eltype(r) == Vector{eltype(r0)}
                @test length(r) == 4
                @test first(r) == collect(2:5)
                @test last(r) == collect(8:10)
                test_iteration(r)
                @test all(map(x->length(x)==4, r[1:end-1]))
                @test length(r[end]) <= 4

                # discard the last tile
                r = @inferred FixedTileRange(r0, 4, keep_last=false)
                @test r isa AbstractArray
                @test eltype(r) == Vector{eltype(r0)}
                @test length(r) == 2
                @test first(r) == collect(2:5)
                @test last(r) == collect(6:9)
                test_iteration(r)
                @test all(map(x->length(x)==4, r))
            end
        end

    end # FixedTileRange


    @testset "FixedTile" begin
        ranges = [
            2:10,
            UInt8(2):UInt8(10),
            Base.IdentityUnitRange(2:10),
            OffsetArrays.IdOffsetRange(2:10),
            OffsetArrays.IdOffsetRange(Base.OneTo(9), 1)
        ]

        # scalar case
        s = FixedTile(4, 2)
        @test s == FixedTile(4, 2; keep_last=true)
        @test s == FixedTile(4, CartesianIndex(2))

        for r0 in ranges
            s = FixedTile(4, 2)
            R = @inferred s(r0)
            @test s(r0) == FixedTileRange(r0, 4, 2)

            r1 = CartesianIndices((r0, ))
            @test s(r1) == FixedTileRange(r1, 4, 2)
        end

        # nd case
        for r0 in ranges, n in 2:3
            for (sz, Δ) in [
                    (4, 2),
                    (ntuple(_->4, n), ntuple(_->2, n)),
                    (fill(4, n), fill(2, n)),
                    (ntuple(_->4, n), ntuple(_->CartesianIndex(2), n))
            ]
                s = FixedTile(sz, Δ)
                indices = ntuple(_->r0, n)
                R = @inferred s(indices)
                @test eltype(R) <: FixedTileRange{<:UnitRange}
                @test all(map(==, R, ntuple(_->FixedTileRange(r0, 4, 2), n)))

                indices = CartesianIndices(indices)
                r1 = CartesianIndices((r0, ))
                R = @inferred s(indices)
                @test eltype(R) <: FixedTileRange{<:CartesianIndices}
                @test all(map(==, R, ntuple(_->FixedTileRange(r1, 4, 2), n)))
            end
        end

    end # FixedTile
end
