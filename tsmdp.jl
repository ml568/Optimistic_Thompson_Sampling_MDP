using Distributions
mutable struct TSMDP<:FiniteHorizonAgent
    policy::Matrix{UInt64}
    ns::Array{UInt64, 4}
    phat::Array{Float64, 4}
    n::Array{UInt64, 3}
    rewards::Array{Float64, 3}
    Vopt::Matrix{Float64}
    Qopt::Array{Float64, 3}
    δ::Float64
    rewards_known::Bool
    maxRet::Float64
    maxR::Float64
    explore_bonus::Float64
    # phi::Float64
end

function TSMDP(S, A, H, rewards::Array, δ, explore_bonus=1.)
    policy = zeros(UInt64, S, H)
    ns = zeros(UInt64, S, S, A, H)
    phat = zeros(S, S, A, H)
    phat[1, :, :, :] .= 1.
    n = zeros(UInt64, S, A, H)
    maxRet = sum(max.(reshape(rewards, S*A, H), 1))
    m = TSMDP(policy, ns, phat, n, rewards, zeros(S, H+1), zeros(S, A, H+1), δ, true, maxRet, maximum(rewards[:]), explore_bonus)
    rand!(m.policy, 1:A) # Initialize with uniformly random policy
    m
end

function TSMDP(S, A, H, δ::Real, explore_bonus=1.)
    policy = zeros(UInt64, S, H)
    ns = zeros(UInt64, S, S, A, H)
    phat = zeros(S, S, A, H)
    phat[1, :, :, :] = 1.
    n = zeros(UInt64, S, A, H)
    maxr = 1.
    maxRet = H * maxr
    rewards = zeros(S, A, H)
    m = TSMDP(policy, ns, phat, n, rewards, zeros(S, H+1), zeros(S, A, H+1), δ, false, maxRet, maxr, explore_bonus)
    rand!(m.policy, 1:A) # Initialize with uniformly random policy
    m
end
maxV(m::TSMDP) = m.maxRet
maxR(m::TSMDP) = m.maxR

function update_policy!(m::TSMDP)
    S = nS(m)
    H = horizon(m)
    A = nA(m)
    phi = 1 # number of posteriror samples
    Q = zeros(A)
    # m.phat .= m.ns ./ reshape(m.n, 1, S, A, H) 
    mean = zeros(S,A,H,phi)
    P =  zeros(S,S,A,H,phi)
    # Sampling
    for t ∈ 1:H
        for s ∈ 1:S
            for a ∈ 1:A
                # Draw phi samples of mu[s,a,t] from Beta
                beta_distribution = Beta(m.rewards[s,a,t]* m.n[s,a,t]+1 , (1-m.rewards[s,a,t]) * m.n[s,a,t]+1)
                mean[s,a,t,:] = rand(beta_distribution,phi)
                # Draw phi samples of P[s,a,t] from Dirichlet
                dirchlet_distribution = Dirichlet(m.ns[:,s,a,t].+1)
                P[:,s,a,t,:] = rand(dirchlet_distribution,phi)
            end
        end
    end

    # Planning
    for t=H:-1:1
        V = m.Vopt[:, t+1]
        for s ∈ 1:S
            Q = zeros(A)
            for a ∈ 1:A
                # Want: max over i and j, mu(s,a,t,j) + <V_(t+1), P(s,a,t,j)>
                for i ∈ phi ,j ∈ phi
                    temp  = mean[s,a,t,i] + dot(V, P[:,s,a,t,j])
                    if temp > Q[a]
                        Q[a] = temp
                    end
                end
            end
            # update policy[s,t] and Vopt[s,t]
            bestA = argmax(Q)
            m.policy[s, t] = bestA
            m.Vopt[s, t] = Q[bestA]
        end
    end
end
