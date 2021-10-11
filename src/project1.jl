# Julia Pkgs
using LightGraphs
using Printf
using SpecialFunctions

# User Pkgs
using aa228_project1

"""
    write_gph(dag::DiGraph, idx2names, filename)

Takes a DiGraph, a Dict of index to names and a output filename to write
the graph in `gph` format.
"""
function write_gph(dag::DiGraph, idx2names, filename)
    open(filename, "w") do io
        for edge in edges(dag)
            @printf(io, "%s,%s\n", idx2names[src(edge)], idx2names[dst(edge)])
        end
    end
end

function get_n_var_values(
    graph_vars::Vector{Variable}, 
    graph_struct::DiGraph,
    n_vars::Int
)
    # Number of possible values (instantiations) for each variable.
    n_var_values = [graph_vars[i].m for i in 1:n_vars]

    return n_var_values, n_pa_values
end

function get_n_pa_values(
    graph_vars::Vector{Variable}, 
    graph_struct::DiGraph,
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
    return dot(k, x.-1) + 1
end

"""
    bayes_net_counts(
        graph_vars::Vector{Variable}, 
        graph_struct::DiGraph, 
        dataset::Matrix{Int}, 
        n_vars::Int
    )

Given a set of variables, a graph structure, and discrete data set, 
compute the count matrices for each variable.
"""
function bayes_net_counts(
    graph_vars::Vector{Variable}, 
    graph_struct::DiGraph, 
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
    for obs_col in eachcol(dataset)
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
            M[i_var][j, obs_value] += 1.0
        end
    end
    return count_matrix
end

"""
    uniform_prior(
        graph_vars::Vector{Variable}, 
        graph_struct::DiGraph, 
        n_vars::Int
    )
Returns alpha vector for uniform Dirchlet prior
"""
function uniform_prior(
    graph_vars::Vector{Variable}, 
    graph_struct::DiGraph, 
    n_vars::Int
)
    # Number of possible values (instantiations) for variable and their parents
    n_var_values = get_n_var_values(graph_vars, graph_struct, n_vars)
    n_pa_values = get_n_pa_values(graph_vars, graph_struct, n_vars, n_var_values)
    return [ones(n_pa_values[i], n_var_values[i]) for i in 1:n_vars]
end

function bayesian_score_component(
    count_matrix::Array{Matrix{Float64}},
    alpha::Array{Matrix{Float64}}
)
    val =  sum(loggamma.(alpha + count_matrix))
    val -= sum(loggamma.(alpha))
    val += sum(loggamma.(sum(alpha, dims=2)))
    val -= sum(loggamma.(sum(alpha, dims=2) + sim(M, dims=2)))
    return val
end

function bayesian_score(
    graph_vars::Vector{Variable}, 
    graph_struct::DiGraph, 
    dataset::Matrix{Int}, 
    n_vars::Int
)
    count_matrix = statistics(graph_vars, graph_struct, dataset, n_vars)
    alpha = uniform_prior(graph_vars, graph_struct, n_vars)
    # alpha = [ones(size(count_matrix[i]))for i in 1:5] # should do the same thing

    # Bayesian score calculated according to P. 96, Eq. 5.5
    return sum(bayesian_score_component(count_matrix[i], alpha[i]) for i in 1:n_vars)
end

function k2(
    graph_vars::Vector{Variable}, 
    graph_struct::DiGraph,
    dataset::Matrix{Int},
    n_vars::Int,
    ordering::Vector{Int}
)

    for i in 1:n_vars
        parents = []
        current_node = ordering[i]
        prev_best_score = bayesian_score(graph_vars, graph_struct, dataset)
        while true
            best_score = -inf
            best_parent = 0
            for j in 1:i-1
                current_parent = ordering[j]
                if !hasedge(graph_struct, current_parent, current_node)
                    add_edge!(graph_struct, current_parent, current_node)
                    current_score = bayesian_score(graph_vars, graph_struct, dataset)
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
    return graph_struct
end

function compute(infile, outfile)

    # WRITE YOUR CODE HERE
    # FEEL FREE TO CHANGE ANYTHING ANYWHERE IN THE CODE
    # THIS INCLUDES CHANGING THE FUNCTION NAMES, MAKING THE CODE MODULAR, BASICALLY ANYTHING

end

if length(ARGS) != 2
    error("usage: julia project1.jl <infile>.csv <outfile>.gph")
end

inputfilename = ARGS[1]
outputfilename = ARGS[2]

compute(inputfilename, outputfilename)
