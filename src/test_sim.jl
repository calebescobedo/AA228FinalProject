# using AA228FinalProject
using POMDPs
using POMCPOW
using ARDESPOT
using BasicPOMCP
using POMDPPolicies
using BeliefUpdaters
using ParticleFilters
using StaticArrays
using Distributions
using Parameters
using POMDPModelTools
using Statistics
using Graphics
using LinearAlgebra
using Base64
using NearestNeighbors
using POMDPSimulators: RolloutSimulator


using POMDPSimulators
using Cairo
using Gtk
using Random
using Printf
using QMDP
using ProfileView


include("line_segment_utils.jl")
include("env_room.jl")
include("roomba_env.jl")
include("filtering.jl")

s = Lidar()
config = 3 # 1,2,3 or 4
speed = 2.0
aspace = vec([RoombaAct(v, om) for v in (0.0, speed), om in (-1.0, 0, 1.0)])
m = RoombaPOMDP(sensor=s, mdp=RoombaMDP(config=config, aspace=aspace));

num_particles = 5000
v_noise_coefficient = 2.0
om_noise_coefficient = 0.5

belief_updater = RoombaParticleFilter(m, num_particles, v_noise_coefficient, om_noise_coefficient);


# Define the policy to test
struct RandomWalk <: Policy
end
# define a new function that takes in the policy struct and current belief,
# and returns the desired action
function POMDPs.action(p::RandomWalk, b::ParticleCollection{FullRoombaState})
    return rand(actions(m))
end

struct Heuristic <: Policy
    ts::Int64 # to track the current time-step
end

function POMDPs.action(p::Heuristic, b::ParticleCollection{FullRoombaState})
    p.ts += 1
    if (p.ts % 5) == 0
        return RoombaAct(1, 0.8) # 30 degrees
    else
        return RoombaAct(1, 0)
    end
end


lower = -2
upper = 2
ardespot = solve(DESPOTSolver(bounds=IndependentBounds(lower, upper, check_terminal=true), T_max=2.0, K=50), m)

# first seed the environment
Random.seed!(20)


# reset the policy
random_walk = RandomWalk()

heuristic = Heuristic(0)

# run the simulation
c = @GtkCanvas()
win = GtkWindow(c, "Roomba Environment", 600, 600)
for (t, step) in enumerate(stepthrough(m, pomcp, belief_updater, max_steps=300))
# for (t, step) in enumerate(stepthrough(m, heuristic, belief_updater, max_steps=300))
# for (t, step) in enumerate(stepthrough(m, pomcpow, belief_updater, max_steps=300))
# for (t, step) in enumerate(stepthrough(m, ardespot, belief_updater, max_steps=300))
    @guarded draw(c) do widget
        # the following lines render the room, the particles, and the roomba
        ctx = getgc(c)
        set_source_rgb(ctx,1,1,1)
        paint(ctx)
        render(ctx, m, step)
        # render some information that can help with debugging
        # here, we render the time-step, the state, and the observation
        move_to(ctx,300,400)
        show_text(ctx, @sprintf("t=%d, state=%s, o=%.3f",t,string(step.s),step.o))
        # show_text(ctx, @sprintf("t=%d, state=%s, o=%.3f",t,string(step.s),step.o))
    end
    show(c)
    sleep(0.1) # to slow down the simulation
end


# @profview simulate(RolloutSimulator(max_steps=1), m, pomcp, belief_updater)
# @time @show simulate(RolloutSimulator(max_steps=100), m, ardespot, belief_updater)
# @time @show simulate(RolloutSimulator(max_steps=100), m, pomcpow, belief_updater)
# @time @show simulate(RolloutSimulator(max_steps=100), m, heuristic, belief_updater)
# @show mean(simulate(RolloutSimulator(max_steps=500), m, heuristic, belief_updater) for _ in 1:3)
# @show mean(simulate(RolloutSimulator(max_steps=500), m, pomcp, belief_updater) for _ in 1:3)pomcp_data = [simulate(RolloutSimulator(max_steps=100), m, pomcp, belief_updater) for _ in 1:10]
# @show pomcp_data
# @show mean(pomcp_data)
# @show std(pomcp_data)
# pomcpow_data = [simulate(RolloutSimulator(max_steps=100), m, pomcpow, belief_updater) for _ in 1:10]
# @show pomcpow_data
# @show mean(pomcpow_data)
# @show std(pomcpow_data)
# despot_data = [simulate(RolloutSimulator(max_steps=100), m, ardespot, belief_updater) for _ in 1:10]
# @show despot_data
# @show mean(despot_data)
# @show std(despot_data)
# heuristic_data = [simulate(RolloutSimulator(max_steps=100), m, heuristic, belief_updater) for _ in 1:10]
# @show heuristic_data
# @show mean(heuristic_data)
# @show std(heuristic_data)
