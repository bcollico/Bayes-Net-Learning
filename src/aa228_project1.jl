module aa228_project1

export Variable, Particle, PSO

struct Variable
    name::Symbol
    m::Int
end

mutable struct Particle
    i::Int
    pos::Vector{Int}
    vel::Vector{Float64}
    score::Float64
    pos_pbest::Vector{Int}
    score_pbest::Float64
    function Particle(i::Int, n_vars::Int)
        zn = zeros(n_vars)
        # Initalize each particle position with random permutation of
        # nodes
        pos_0 = shuffle([1:n_vars...])
        return new(i, pos_0, zn, -Inf, pos_0, -Inf)
    end
end

mutable struct PSO
    P::Vector{Particle}
    n_p::Int
    n_vars::Int
    pos_gbest::Vector{Int}
    score_gbest::Float64
    mut_frac::Float64
    mut_n::Int
    search_fcn
    gbest_i::Int
    gbest_i_max::Int
    function PSO(n_p::Int, n_vars::Int, search_fcn)
        P = [Particle(i, n_vars) for i in 1:n_p]
        return new(P, n_p, m_vars, zeros(Int, n_vars), -Inf, 0.1, 2, search_fcn, 0, 5)
    end
end

end # module
