@testset "TileIndices" begin
    function test_iteration(R)
        for (i, v) in enumerate(R)
            @test v == R[i]
        end
    end

    @testset "FixedTile" begin
        @testset "axes" begin
            R0 = (1:4, 0:5)
            
            R = @inferred TileIndices(R0, FixedTile((3, 4), (2, 3)))
            @test R == TileIndices(R0, (3, 4), (2, 3))
            @test R isa AbstractArray
            @test eltype(R) == Tuple{UnitRange{Int}, UnitRange{Int}}
            @test size(R) == (2, 2)
            @test R[1] == (1:3, 0:3)
            @test R[end] == (3:4, 3:5)
            test_iteration(R)

            R = @inferred TileIndices(R0, FixedTile((3, 4), (2, 3), keep_last=false))
            @test R == TileIndices(R0, (3, 4), (2, 3), keep_last=false)
            @test R isa AbstractArray
            @test eltype(R) == Tuple{UnitRange{Int}, UnitRange{Int}}
            @test size(R) == (1, 1)
            @test R[1] == (1:3, 0:3)
            test_iteration(R)

            R = @inferred TileIndices(R0, FixedTile(3, 2))
            @test R == TileIndices(R0, 3, 2)
            @test R isa AbstractArray
            @test eltype(R) == Tuple{UnitRange{Int}, UnitRange{Int}}
            @test size(R) == (2, 3)
            @test R[1] == R[1, 1] == (1:3, 0:2)
            @test R[3] == R[1, 2] == (1:3, 2:4)
            @test R[end] == R[6] == (3:4, 4:5)
            test_iteration(R)

            R = @inferred TileIndices(R0, FixedTile(3, 2, keep_last=false))
            @test R == TileIndices(R0, 3, 2, keep_last=false)
            @test R isa AbstractArray
            @test eltype(R) == Tuple{UnitRange{Int}, UnitRange{Int}}
            @test size(R) == (1, 2)
            @test R[1] == R[1, 1] == (1:3, 0:2)
            @test R[2] == R[1, 2] == R[end] == (1:3, 2:4)
        end

        @testset "CartesianIndices" begin
            R0 = CartesianIndex(1,0):CartesianIndex(4, 5)
            
            R = @inferred TileIndices(R0, FixedTile((3, 4), (2, 3)))
            @test R == TileIndices(R0, (3, 4), (2, 3))
            @test R isa AbstractArray
            @test eltype(R) <: Tuple{<:CartesianIndices, <:CartesianIndices}
            @test size(R) == (2, 2)
            @test R[1] == CartesianIndices((1:3, 0:3))
            @test R[end] == CartesianIndices((3:4, 3:5))
            test_iteration(R)

            R = @inferred TileIndices(R0, FixedTile((3, 4), (2, 3), keep_last=false))
            @test R == TileIndices(R0, (3, 4), (2, 3), keep_last=false)
            @test R isa AbstractArray
            @test eltype(R) <: Tuple{<:CartesianIndices, <:CartesianIndices}
            @test size(R) == (1, 1)
            @test R[1] == CartesianIndices((1:3, 0:3))
            test_iteration(R)

            R = @inferred TileIndices(R0, FixedTile(3, 2))
            @test R == TileIndices(R0, 3, 2)
            @test R isa AbstractArray
            @test eltype(R) <: Tuple{<:CartesianIndices, <:CartesianIndices}
            @test size(R) == (2, 3)
            @test R[1] == R[1, 1] == CartesianIndices((1:3, 0:2))
            @test R[3] == R[1, 2] == CartesianIndices((1:3, 2:4))
            @test R[end] == R[6] == CartesianIndices((3:4, 4:5))
            test_iteration(R)

            R = @inferred TileIndices(R0, FixedTile(3, 2, keep_last=false))
            @test R == TileIndices(R0, 3, 2, keep_last=false)
            @test R isa AbstractArray
            @test eltype(R) <: Tuple{<:CartesianIndices, <:CartesianIndices}
            @test size(R) == (1, 2)
            @test R[1] == R[1, 1] == CartesianIndices((1:3, 0:2))
            @test R[2] == R[1, 2] == R[end] == CartesianIndices((1:3, 2:4))
        end
    end
end
