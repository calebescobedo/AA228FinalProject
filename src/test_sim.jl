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
mutable struct ToEnd <: Policy
    ts::Int64 # to track the current time-step.
end

# extract goal for heuristic controller
goal_xy = get_goal_xy(m)
pomcp = solve(POMCPSolver(), m)
pomcpow = solve(POMCPOWSolver(criterion=MaxUCB(2.0)), m)
lower = -2 # close proximity to a wall
upper = 2
ardespot = solve(DESPOTSolver(bounds=IndependentBounds(lower, upper, check_terminal=true)), m)

# define a new function that takes in the policy struct and current belief,
# and returns the desired action
function POMDPs.action(p::ToEnd, b::ParticleCollection{FullRoombaState})
    p.ts += 1
    if (p.ts % 5 == 0)
        return RoombaAct(1, 0.8) #30 degrees

    else
        return RoombaAct(1, 0)
    end


    # # after 50 time-steps, we follow a proportional controller to navigate
    # # directly to the goal, using the mean belief state
    # s = mean(b) # TODO: right now we can't take the mean of our system, how can we add all parts togeather
    # # compute the difference between our current heading and one that would
    # # point to the goal
    # goal_x, goal_y = goal_xy
    # x,y,th = s[1:3]
    # ang_to_goal = atan(goal_y - y, goal_x - x)
    # del_angle = wrap_to_pi(ang_to_goal - th)

    # # apply proportional control to compute the turn-rate
    # Kprop = 1.0
    # om = Kprop * del_angle

    # # always travel at some fixed velocity
    # v = 5.0

end


# first seed the environment
Random.seed!(20)

# reset the policy
heuristic = ToEnd(0) # here, the argument sets the time-steps elapsed to 0



# run the simulation
# c = @GtkCanvas()
# win = GtkWindow(c, "Roomba Environment", 600, 600)
# # for (t, step) in enumerate(stepthrough(m, pomcp, belief_updater, max_steps=300))
# for (t, step) in enumerate(stepthrough(m, heuristic, belief_updater, max_steps=300))
# # for (t, step) in enumerate(stepthrough(m, pomcpow, belief_updater, max_steps=300))
# # for (t, step) in enumerate(stepthrough(m, ardespot, belief_updater, max_steps=300))
#     @guarded draw(c) do widget
#         # the following lines render the room, the particles, and the roomba
#         ctx = getgc(c)
#         set_source_rgb(ctx,1,1,1)
#         paint(ctx)
#         render(ctx, m, step)
#         # render some information that can help with debugging
#         # here, we render the time-step, the state, and the observation
#         move_to(ctx,300,400)
#         show_text(ctx, @sprintf("t=%d, state=%s, o=%.3f",t,string(step.s),step.o))
#         # show_text(ctx, @sprintf("t=%d, state=%s, o=%.3f",t,string(step.s),step.o))
#     end
#     show(c)
#     sleep(0.1) # to slow down the simulation
# end


@profview simulate(RolloutSimulator(max_steps=100), m, pomcp, belief_updater)
# @time @show simulate(RolloutSimulator(max_steps=100), m, ardespot, belief_updater)
# @time @show simulate(RolloutSimulator(max_steps=100), m, pomcpow, belief_updater)
# @time @show simulate(RolloutSimulator(max_steps=100), m, heuristic, belief_updater)
# @show mean(simulate(RolloutSimulator(max_steps=500), m, heuristic, belief_updater) for _ in 1:3)
# @show mean(simulate(RolloutSimulator(max_steps=500), m, pomcp, belief_updater) for _ in 1:3)pomcp_data = [simulate(RolloutSimulator(max_steps=100), m, pomcp, belief_updater) for _ in 1:10]
@show pomcp_data
@show mean(pomcp_data)
@show std(pomcp_data)
pomcpow_data = [simulate(RolloutSimulator(max_steps=100), m, pomcpow, belief_updater) for _ in 1:10]
@show pomcpow_data
@show mean(pomcpow_data)
@show std(pomcpow_data)
despot_data = [simulate(RolloutSimulator(max_steps=100), m, ardespot, belief_updater) for _ in 1:10]
@show despot_data
@show mean(despot_data)
@show std(despot_data)
heuristic_data = [simulate(RolloutSimulator(max_steps=100), m, heuristic, belief_updater) for _ in 1:10]
@show heuristic_data
@show mean(heuristic_data)
@show std(heuristic_data)
