# https://github.com/JuliaLang/julia/pull/29440
if VERSION < v"1.1.0-DEV.389"
    Base.:(:)(I::CartesianIndex{N}, J::CartesianIndex{N}) where N =
        CartesianIndices(map((i,j) -> i:j, Tuple(I), Tuple(J)))
end
