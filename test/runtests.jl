using TiledIteration
using Base.Test

@testset "tiled iteration" begin
    sz = (3,5)
    for i1 = -3:2, i2 = -1:5
        for l1 = 2:13, l2 = 1:16
            inds = (i1:i1+l1-1, i2:i2+l2-1)
            A = zeros(Int, inds...)
            k = 0
            for tileinds in TileIterator(inds, sz)
                tile = A[tileinds...]
                @test !isempty(tile)
                @test all(tile .== 0)
                A[tileinds...] = (k+=1)
            end
            @test minimum(A) == 1
        end
    end
end

nothing
