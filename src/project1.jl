# Julia Pkgs
using LinearAlgebra
using LightGraphs
using DataFrames
using CSV
using Printf
using SpecialFunctions
using TikzGraphs
using TikzPictures
using Random

# User Pkgs
# using aa228_project1

# user structs
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
        return new(i, pos_0, rand(0:1e-5:1,n_vars), -Inf, pos_0, -Inf)
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
        return new(P, n_p, n_vars, zeros(Int, n_vars), -Inf, 0.1, 2, search_fcn, 0, 5)
    end
end

# user fcns
"""
    write_gph(dag::SimpleDiGraph, idx2names, filename)

Takes a SimpleDiGraph, a Dict of index to names and a output filename to write
the graph in `gph` format.
"""
function write_gph(dag::SimpleDiGraph, idx2names, filename)
    open(filename, "w") do io
        for edge in edges(dag)
            @printf(io, "%s,%s\n", idx2names[src(edge)], idx2names[dst(edge)])
        end
    end
end

function write_score(score, filename)
    open(filename, "w") do io
        @printf(io, "%f", score)
    end
end

function write_image(dag::SimpleDiGraph, filename)
    t = TikzGraphs.plot(dag)
    TikzPictures.save(PDF(filename),t)
end

function output(
    graph_struct::SimpleDiGraph,
    best_score::Float64,
    var_names::Array{String},
    file_output::String
)

    file_output_graph = file_output*".gph"
    file_output_fig = file_output
    file_output_score = file_output*".score"

    write_gph(graph_struct, var_names, file_output_graph)
    write_image(graph_struct, file_output_fig)
    write_score(best_score, file_output_score)
end

function get_n_var_values(
    graph_vars::Vector{Variable}, 
    graph_struct::SimpleDiGraph,
    n_vars::Int
)
    # Number of possible values (instantiations) for each variable.
    n_var_values = [graph_vars[i].m for i in 1:n_vars]

    return n_var_values
end

function get_n_pa_values(
    graph_vars::Vector{Variable}, 
    graph_struct::SimpleDiGraph,
    n_vars::Int,
    n_var_values
)
    # Number of possible values (instantiations) for the parents of each variable.
    n_pa_values = [prod([n_var_values[j] for j in inneighbors(graph_struct, i)]) for i in 1:n_vars]

    return n_pa_values
end

"""
    sub2ind(siz, x)

"""
function sub2ind(siz, x)
    k = vcat(1, cumprod(siz[1:end-1]))
    return dot(k, x .-1) + 1
end

"""
    bayes_net_counts(
        graph_vars::Vector{Variable}, 
        graph_struct::SimpleDiGraph, 
        dataset::Matrix{Int}, 
        n_vars::Int
    )

Given a set of variables, a graph structure, and discrete data set, 
compute the count matrices for each variable.
"""
function bayes_net_counts(
    graph_vars::Vector{Variable}, 
    graph_struct::SimpleDiGraph, 
    dataset::Matrix{Int}, 
    n_vars::Int
)
    # Number of possible values (instantiations) for variable and their parents
    n_var_values = get_n_var_values(graph_vars, graph_struct, n_vars)
    n_pa_values = get_n_pa_values(graph_vars, graph_struct, n_vars, n_var_values)

    # Pre-allocated matrix to store counts for variable instantiation i
    # given a parent instantiation j
    count_matrix = [zeros(n_pa_values[i], n_var_values[i]) for i in 1:n_vars]

    # eachcol creates a generator that iterates over the second dimension
    # of the dataset, returning the columns as AbstractVector views
    for obs_col in eachcol(transpose(dataset))
        # first loop:
        #   loop through each observation vector in the dataset
        for i_var in 1:n_vars
            # second loop:
            #   loop through each variable in the current observation
            #   vector
            # observation value relevant to the current looping variable
            obs_value = obs_col[i_var]

            # parents of current looping variable
            parents = inneighbors(graph_struct, i_var)

            # increment the counter for the current looping variable
            # given parental instantiation j and observation value k
            j = 1
            if !isempty(parents)
                j = sub2ind(n_var_values[parents], obs_col[parents])
            end
            count_matrix[i_var][j, obs_value] += 1.0
        end
    end
    return count_matrix
end

"""
    uniform_prior(
        graph_vars::Vector{Variable}, 
        graph_struct::SimpleDiGraph, 
        n_vars::Int
    )
Returns alpha vector for uniform Dirchlet prior
"""
function uniform_prior(
    graph_vars::Vector{Variable}, 
    graph_struct::SimpleDiGraph, 
    n_vars::Int
)
    # Number of possible values (instantiations) for variable and their parents
    n_var_values = get_n_var_values(graph_vars, graph_struct, n_vars)
    n_pa_values = get_n_pa_values(graph_vars, graph_struct, n_vars, n_var_values)
    return [ones(n_pa_values[i], n_var_values[i]) for i in 1:n_vars]
end

function bayesian_score_component(
    count_matrix,
    alpha
)
    val =  sum(loggamma.(alpha + count_matrix))
    val -= sum(loggamma.(alpha))
    val += sum(loggamma.(sum(alpha, dims=2)))
    val -= sum(loggamma.(sum(alpha, dims=2) + sum(count_matrix, dims=2)))
    return val
end

function bayesian_score(
    graph_vars::Vector{Variable}, 
    graph_struct::SimpleDiGraph, 
    dataset::Matrix{Int}, 
    n_vars::Int
)
    count_matrix = bayes_net_counts(graph_vars, graph_struct, dataset, n_vars)
    alpha = uniform_prior(graph_vars, graph_struct, n_vars)
    # alpha = [ones(size(count_matrix[i]))for i in 1:5] # should do the same thing

    # Bayesian score calculated according to P. 96, Eq. 5.5
    return sum(bayesian_score_component(count_matrix[i], alpha[i]) for i in 1:n_vars)
end

function k2(
    graph_vars::Vector{Variable}, 
    graph_struct::SimpleDiGraph,
    dataset::Matrix{Int},
    n_vars::Int,
    ordering::Vector{Int}
)
    for i in 1:n_vars
        parents = []
        current_node = ordering[i]
        prev_best_score = bayesian_score(graph_vars, graph_struct, dataset, n_vars)
        while true
            best_score = -Inf
            best_parent = 0
            for j in 1:i-1
                current_parent = ordering[j]
                if !has_edge(graph_struct, current_parent, current_node)
                    add_edge!(graph_struct, current_parent, current_node)
                    current_score = bayesian_score(graph_vars, graph_struct, dataset, n_vars)
                    if current_score > best_score
                        best_score = current_score
                        best_parent = current_parent
                    end
                    rem_edge!(graph_struct, current_parent, current_node)
                end
            end
            if best_score > prev_best_score
                prev_best_score = best_score
                add_edge!(graph_struct, best_parent, current_node)
            else
                break
            end
        end

    end
    return graph_struct, bayesian_score(graph_vars, graph_struct, dataset, n_vars)
end

function particle_eval(pso::PSO)

    # increment 
    pso.gbest_i += 1

    for particle in pso.P
        _, particle.score = pso.search_fcn(particle.pos)
        if particle.score > particle.score_pbest
            particle.score_pbest = particle.score
            particle.pos_pbest = particle.pos_pbest
        end
        if particle.score_pbest > pso.score_gbest
            pso.score_gbest = particle.score_pbest
            pso.pos_gbest = particle.pos_pbest
            pso.gbest_i = 0
        end
        
    end

end

function particle_update(pso::PSO)
    #C = zeros(3) # Matrix of weighting coefficients
    C = [0.1, 0.45, 0.45]
    for particle in pso.P
        
        # Weight the personal best score based on percent deviation 
        # normalized by number of components shared with pbest position
        #C[2] = abs(particle.score_pbest - particle.score) / 
        #       (sum([particle.pos_pbest[i] == particle.pos[i] for i in 1:pso.n_vars]) + 1)

        # Weight the global best score based on percent deviation 
        # normalized by number of components shared with gbest position
        #C[3] = abs(pso.score_gbest - particle.score) / 
        #        (sum([pso.pos_gbest[i] == particle.pos[i] for i in 1:pso.n_vars]) + 1)

        # Weight the inertia term using a random value less than or equal
        # to the max of the pbest and gbest weighting terms
        #C[1] = rand(0:1e-5:max(C...))

        # Compute the particle velocity update as the weighted combination
        # of current velocity,

        #@printf("\tC_%i: %f, %f, %f\n", particle.i, C[1], C[2], C[3])

        #normalize!(C)

        particle.vel = C[1]*(particle.vel) +
                       C[2]*(particle.pos_pbest - particle.pos) +
                       C[3]*(pso.pos_gbest      - particle.pos) 

        # The position update produces float values. Compute the new
        # particle position by sorting by the most negative values 
        # computed in the x_i + v_i position update.
        particle.pos = particle.pos[sortperm(particle.pos + particle.vel)]
    end
end

function particle_mutate(pso::PSO)

    # generate random indices to mutate
    mut_idx = rand(1:pso.n_vars, Int64(ceil(pso.mut_frac*pso.n_p)), pso.mut_n)

    for particle in pso.P
        for i in 1:Int64(pso.mut_n/2)
            # swap each pair of random indices
            temp = particle.pos[mut_idx[i]]
            particle.pos[mut_idx[i]] = particle.pos[mut_idx[i*2-1]]
            particle.pos[mut_idx[i*2-1]] = temp
        end
    end
end

function particle_swarm_optimization(pso::PSO)
    i = 0
    while pso.gbest_i < pso.gbest_i_max
        i += 1

        @printf("PSO Iteration: %i, gbest: %f, gbest_i: %i\n", i, pso.score_gbest, pso.gbest_i)
        for particle in pso.P
            fmt = "%i "
            @printf("\tParticle %i Ordering: ", particle.i)
            for i in 1:pso.n_vars
                @printf("% i", particle.pos[i])
            end
            @printf(" and Velocity: ")
            for i in 1:pso.n_vars
                @printf("%f ", particle.vel[i])
            end
            @printf("\n")
        end

        particle_eval(pso)
        particle_update(pso)
        particle_mutate(pso)
    end
end

function compute(file_dataset::String, file_output::String)

    dataset_df = DataFrame(CSV.File(file_dataset))

    dataset = Matrix(dataset_df)
    var_names = names(dataset_df)

    n_vars = size(dataset,2)
    graph_struct = SimpleDiGraph(n_vars)
    graph_vars = [Variable(Symbol(var_names[i]),findmax(dataset[:,i])[1]) for i in 1:n_vars]

    for i in 1:n_vars
        @printf("Variable: %s, # Values: %i\n", graph_vars[i].name, graph_vars[i].m)
    end

    search_fcn(ordering::Vector{Int}) = k2(graph_vars, SimpleDiGraph(n_vars), dataset, n_vars, ordering)
    pso = PSO(2, n_vars, search_fcn)

    particle_swarm_optimization(pso)

    graph_struct, best_score = pso.search_fcn(pso.pos_gbest)

    @printf("\tFinal Results: \n\t Score: %f \n\t Ordering: ", pso.score_gbest)
    for i in 1:pso.n_vars
        @printf("% i", pso.pos_gbest[i])
    end

    output(graph_struct, best_score, var_names, file_output)

end

#code execution
runcases = ["small","medium","large"]
which_run = [2]

for i in which_run
    file_dataset = joinpath(@__DIR__,"..","data",runcases[i]*".csv")
    file_output = joinpath(@__DIR__,"..","output",runcases[i],runcases[i])

    @time compute(file_dataset, file_output)
end
