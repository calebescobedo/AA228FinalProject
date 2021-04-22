# using AA228FinalProject
using POMDPs
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

import POMDPModelTools: render
using POMDPSimulators
using Cairo
using Gtk
using Random
using Printf

include("line_segment_utils.jl")
include("env_room.jl")
include("roomba_env.jl")
include("filtering.jl")

s = Lidar()
config = 3 # 1,2, or 3
speed = 2.0
aspace = vec([RoombaAct(v, om) for v in (0.0, speed), om in (-1.0, 0, 1.0)])
m = RoombaPOMDP(sensor=s, mdp=RoombaMDP(config=config, aspace=aspace));

num_particles = 1000
v_noise_coefficient = 2.0
om_noise_coefficient = 0.5

belief_updater = RoombaParticleFilter(m, num_particles, v_noise_coefficient, om_noise_coefficient);


# Define the policy to test
mutable struct ToEnd <: Policy
    ts::Int64 # to track the current time-step.
end

# extract goal for heuristic controller
goal_xy = get_goal_xy(m)

# define a new function that takes in the policy struct and current belief,
# and returns the desired action
function POMDPs.action(p::ToEnd, b::ParticleCollection{FullRoombaState})

    # spin around to localize for the first 25 time-steps
    if p.ts < 25
        p.ts += 1
        return RoombaAct(0.,1.0) # all actions are of type RoombaAct(v,om)
    end
    p.ts += 1

    # after 25 time-steps, we follow a proportional controller to navigate
    # directly to the goal, using the mean belief state
    @show typeof(b)
    @show sizeof(b)
    s = mean(b)

    # compute the difference between our current heading and one that would
    # point to the goal
    goal_x, goal_y = goal_xy
    x,y,th = s[1:3]
    ang_to_goal = atan(goal_y - y, goal_x - x)
    del_angle = wrap_to_pi(ang_to_goal - th)

    # apply proportional control to compute the turn-rate
    Kprop = 1.0
    om = Kprop * del_angle

    # always travel at some fixed velocity
    v = 5.0

    return RoombaAct(v, om)
end


# first seed the environment
Random.seed!(2)

# reset the policy
p = ToEnd(0) # here, the argument sets the time-steps elapsed to 0

# run the simulation
c = @GtkCanvas()
win = GtkWindow(c, "Roomba Environment", 600, 600)
for (t, step) in enumerate(stepthrough(m, p, belief_updater, max_steps=100))
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
    end
    show(c)
    sleep(0.1) # to slow down the simulation
end