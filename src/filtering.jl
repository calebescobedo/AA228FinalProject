# specification of particle filters for the bumper and lidar Roomba environments
# maintained by {jmorton2,kmenda}@stanford.edu

"""
Definition of the particle filter for the Roomba environment
Fields:
- `v_noise_coeff::Float64` coefficient to scale particle-propagation noise in velocity
- `om_noise_coeff::Float64`coefficient to scale particle-propagation noise in turn-rate
"""
mutable struct RoombaParticleFilter{M<:RoombaModel,RM,RNG<:AbstractRNG,PMEM} <: Updater
    model::M
    resampler::RM
    n_init::Int
    v_noise_coeff::Float64
    om_noise_coeff::Float64
    rng::RNG
    _particle_memory::PMEM
    _weight_memory::Vector{Float64}
end

function RoombaParticleFilter(model, n::Integer, v_noise_coeff, om_noise_coeff, rng::AbstractRNG=Random.GLOBAL_RNG)
    return RoombaParticleFilter(model,
                            #    ImportanceResampler(n),
                               LowVarianceResampler(n),
                               n,
                               v_noise_coeff,
                               om_noise_coeff,
                               rng,
                               sizehint!(particle_memory(model), n),
                               sizehint!(Float64[], n)
                              )
end

# Modified Update function adds noise to the actions that propagate particles
function POMDPs.update(up::RoombaParticleFilter, b::ParticleCollection{<:FullRoombaState}, a, o)
    pm = up._particle_memory
    wm = up._weight_memory
    empty!(pm)
    empty!(wm)
    all_terminal = true
    # Need to update everything else other than the roomba itself
    for s in particles(b)
        if !isterminal(up.model, s)
            all_terminal = false
            # noise added here:
            a_pert = a + SVec2(up.v_noise_coeff*(rand(up.rng)-0.5), up.om_noise_coeff*(rand(up.rng)-0.5))
            sp = @gen(:sp)(up.model, s, a_pert, up.rng)
            sp.human.x += (rand()-0.5) * 0.5
            sp.human.y += (rand()-0.5) * 0.5
            #random resample robot state
            if rand() < 0.01
                sp.roomba.x = rand() * 34.0 - 23.0
                if sp.roomba.x > -16
                    sp.roomba.y = rand() * (-6.0) + 3
                else
                    sp.roomba.y = rand() * (-23.0) + 4
                end
            end

            # random resample human state
            if rand() < 0.01
                if sp.human.theta == 3.14
                    sp.human.theta = 0
                else
                    sp.human.theta = 3.14
                end
                sp.human.x = rand() * 34.0 - 23.0
                if sp.human.x > -16
                    sp.human.y = rand() * (-6.0) + 3
                else
                    sp.human.y = rand() * (-23.0) + 4
                end
            end


            push!(pm, sp)
            push!(wm, obs_weight(up.model, s, a_pert, sp, o))
        end
    end
    # if all particles are terminal, issue an error
    if all_terminal
        error("Particle filter update error: all states in the particle collection were terminal.")
    end

    return ParticleFilters.resample(up.resampler,
                    WeightedParticleBelief(pm, wm, sum(wm), nothing),
                    up.model,
                    up.model,
                    b, a, o,
                    up.rng)
end

# initialize belief state
ParticleFilters.initialize_belief(up::RoombaParticleFilter, d) = ParticleCollection([rand(up.rng, d) for i in 1:up.n_init])