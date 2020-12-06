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

            # StepRange
            R0 = (1:2:8, 0:3:20)

            R = @inferred TileIndices(R0, FixedTile((3, 4), (2, 3)))
            @test R == TileIndices(R0, (3, 4), (2, 3))
            @test R isa AbstractArray
            @test eltype(R) == Tuple{StepRange{Int, Int}, StepRange{Int, Int}}
            @test size(R) == (2, 4)
            @test R[1] == (1:2:5, 0:3:9)
            @test R[end] == (3:2:7, 9:3:18)
            test_iteration(R)

            R = @inferred TileIndices(R0, FixedTile((3, 4), (2, 3), keep_last=false))
            @test R == TileIndices(R0, (3, 4), (2, 3), keep_last=false)
            @test R isa AbstractArray
            @test eltype(R) == Tuple{StepRange{Int, Int}, StepRange{Int, Int}}
            @test size(R) == (1, 3)
            @test R[1] == (1:2:5, 0:3:9)
            @test R[end] == (1:2:5, 6:3:15)
            test_iteration(R)

            R = @inferred TileIndices(R0, FixedTile(3, 2))
            @test R == TileIndices(R0, 3, 2)
            @test R isa AbstractArray
            @test eltype(R) == Tuple{StepRange{Int, Int}, StepRange{Int, Int}}
            @test size(R) == (2, 6)
            @test R[1] == R[1, 1] == (1:2:5, 0:3:6)
            @test R[3] == R[1, 2] == (1:2:5, 2:3:8)
            @test R[end] == R[12] == (3:2:7, 10:3:16)
            test_iteration(R)

            R = @inferred TileIndices(R0, FixedTile(3, 2, keep_last=false))
            @test R == TileIndices(R0, 3, 2, keep_last=false)
            @test R isa AbstractArray
            @test eltype(R) == Tuple{StepRange{Int, Int}, StepRange{Int, Int}}
            @test size(R) == (1, 6)
            @test R[1] == R[1, 1] == (1:2:5, 0:3:6)
            @test R[6] == R[1, 6] == R[end] == (1:2:5, 10:3:16)
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

            if VERSION >= v"1.6.0-DEV.1174"
                # StepRange CartesianIndices
                # https://github.com/JuliaLang/julia/pull/37829
                R0 = CartesianIndices((1:2:8, 0:3:20))

                R = @inferred TileIndices(R0, FixedTile((3, 4), (2, 3)))
                @test R == TileIndices(R0, (3, 4), (2, 3))
                @test R isa AbstractArray
                @test size(R) == (2, 4)
                @test R[1] == CartesianIndices((1:2:5, 0:3:9))
                @test R[end] == CartesianIndices((3:2:7, 9:3:18))
                test_iteration(R)

                R = @inferred TileIndices(R0, FixedTile((3, 4), (2, 3), keep_last=false))
                @test R == TileIndices(R0, (3, 4), (2, 3), keep_last=false)
                @test R isa AbstractArray
                @test size(R) == (1, 3)
                @test R[1] == CartesianIndices((1:2:5, 0:3:9))
                @test R[end] == CartesianIndices((1:2:5, 6:3:15))
                test_iteration(R)

                R = @inferred TileIndices(R0, FixedTile(3, 2))
                @test R == TileIndices(R0, 3, 2)
                @test R isa AbstractArray
                @test size(R) == (2, 6)
                @test R[1] == R[1, 1] == CartesianIndices((1:2:5, 0:3:6))
                @test R[3] == R[1, 2] == CartesianIndices((1:2:5, 2:3:8))
                @test R[end] == R[12] == CartesianIndices((3:2:7, 10:3:16))
                test_iteration(R)

                R = @inferred TileIndices(R0, FixedTile(3, 2, keep_last=false))
                @test R == TileIndices(R0, 3, 2, keep_last=false)
                @test R isa AbstractArray
                @test size(R) == (1, 6)
                @test R[1] == R[1, 1] == CartesianIndices((1:2:5, 0:3:6))
                @test R[6] == R[1, 6] == R[end] == CartesianIndices((1:2:5, 10:3:16))
            end
        end
    end
end
