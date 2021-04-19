# Defines the environment as a POMDPs.jl MDP and POMDP
# maintained by {jmorton2,kmenda}@stanford.edu

# Wraps ang to be in (-pi, pi]
function wrap_to_pi(ang::Float64)
    if ang > pi
		ang -= 2*pi
	elseif ang <= -pi
		ang += 2*pi
    end
	ang
end

"""
State of a Roomba.

# Fields
- `x::Float64` x location in meters
- `y::Float64` y location in meters
- `theta::Float64` orientation in radians
- `status::Bool` indicator whether robot has reached goal state or stairs
"""

num_obs = 5
struct ObstacleState <: FieldVector{2, Float64}
    x::Float64
    y::Float64
end

struct HumanState <: FieldVector{3, Float64}
    x::Float64
    y::Float64
    theta::Float64
end

struct RoombaState <: FieldVector{3, Float64}
    x::Float64
    y::Float64
    theta::Float64
end

struct FullRoombaState
    roomba::RoombaState
    human::HumanState
    obstacles
    visited
end


# Struct for a Roomba action
struct RoombaAct <: FieldVector{2, Float64}
    v::Float64     # meters per second
    omega::Float64 # theta dot (rad/s)
end

# action spaces
struct RoombaActions end

function gen_amap(aspace::RoombaActions)
    return nothing
end

function gen_amap(aspace::AbstractVector{RoombaAct})
    return Dict(aspace[i]=>i for i in 1:length(aspace))
end

"""
Specify a DiscreteRoombaStateSpace
- `x_step::Float64` distance between discretized points in x
- `y_step::Float64` distance between discretized points in y
- `XLIMS::Vector` boundaries of room (x-dimension)
- `YLIMS::Vector` boundaries of room (y-dimension)

"""
struct DiscreteRoombaStateSpace
    x_step::Float64
    y_step::Float64
    XLIMS::SVector{2, Float64}
    YLIMS::SVector{2, Float64}
    indices::SVector{2, Int}
end

# function to construct DiscreteRoombaStateSpace:
# `num_x_pts::Int` number of points to discretize x range to
# `num_y_pts::Int` number of points to discretize yrange to
function DiscreteRoombaStateSpace(num_x_pts::Int, num_y_pts::Int)

    # hardcoded room-limits
    # watch for consistency with env_room
    XLIMS = SVec2(-30.0, 20.0)
    YLIMS = SVec2(-30.0, 20.0)

    x_step = (XLIMS[2]-XLIMS[1])/(num_x_pts-1)
    y_step = (YLIMS[2]-YLIMS[1])/(num_y_pts-1)

    x_step == y_step ? nothing : throw(AssertionError("x_step must equal y_step."))

    # project ROBOT_W.val/2 to nearest multiple of discrete_step
    ROBOT_W.val = 2 * max(1, round(DEFAULT_ROBOT_W/2 / x_step)) * x_step

    return DiscreteRoombaStateSpace(x_step,
                                    y_step,
                                    XLIMS,YLIMS,
                                    cumprod([num_x_pts, num_y_pts]))
end

# state-space definitions
struct ContinuousRoombaStateSpace end


"""
Define the Roomba MDP.

# Fields
- `v_max::Float64` maximum velocity of Roomba [m/s]
- `om_max::Float64` maximum turn-rate of Roombda [rad/s]
- `dt::Float64` simulation time-step [s]
- `contact_pen::Float64` penalty for wall-contact
- `time_pen::Float64` penalty per time-step
- `goal_reward::Float64` reward for reaching goal
- `stairs_penalty::Float64` penalty for reaching stairs
- `config::Int` specifies room configuration (location of stairs/goal) {1,2,3}
- `room::Room` environment room struct
- `sspace::SS` environment state-space (ContinuousRoombaStateSpace or DiscreteRoombaStateSpace)
- `aspace::AS` environment action-space struct
"""
@with_kw mutable struct RoombaMDP{SS,AS} <: MDP{RoombaState, RoombaAct}
    v_max::Float64  = 10.0  # m/s
    om_max::Float64 = 1.0   # rad/s
    dt::Float64     = 0.5   # s
    contact_pen::Float64 = -1.0
    time_pen::Float64 = -0.1
    goal_reward::Float64 = 10
    stairs_penalty::Float64 = -10
    discount::Float64 = 0.95
    config::Int = 1
    sspace::SS = ContinuousRoombaStateSpace()
    dsspace::DiscreteRoombaStateSpace = DiscreteRoombaStateSpace(100, 100)
    room::Room  = Room(sspace,configuration=config)
    aspace::AS = RoombaActions()
    _amap::Union{Nothing, Dict{RoombaAct, Int}} = gen_amap(aspace)
end

"""
Define the Roomba POMDP

Fields:
- `sensor::T` struct specifying the sensor used (Lidar or Bump)
- `mdp::T` underlying RoombaMDP
"""
struct RoombaPOMDP{SS, AS, T, O} <: POMDP{RoombaState, RoombaAct, O}
    sensor::T
    mdp::RoombaMDP{SS, AS}
end

sensor(m::RoombaPOMDP) = m.sensor

# observation models
struct Bumper end
POMDPs.obstype(::Type{Bumper}) = Bool
POMDPs.obstype(::Bumper) = Bool

struct Lidar
    ray_stdev::Float64 # measurement noise: see POMDPs.observation definition
                       # below for usage
end
Lidar() = Lidar(0.1)

POMDPs.obstype(::Type{Lidar}) = Float64
POMDPs.obstype(::Lidar) = Float64 #float64(x)

struct DiscreteLidar
    ray_stdev::Float64
    disc_points::Vector{Float64} # cutpoints: endpoints of (0, Inf) assumed
    _d_disc::Vector{Float64}
end

POMDPs.obstype(::Type{DiscreteLidar}) = Int
POMDPs.obstype(::DiscreteLidar) = Int
DiscreteLidar(disc_points) = DiscreteLidar(Lidar().ray_stdev, disc_points, Vector{Float64}(length(disc_points)+1))

# Shorthands
const RoombaModel{SS, AS} = Union{RoombaMDP{SS, AS}, RoombaPOMDP{SS, AS}}
const BumperPOMDP{SS, AS} = RoombaPOMDP{SS, AS, Bumper, Bool}
const LidarPOMDP{SS, AS} = RoombaPOMDP{SS, AS, Lidar, Float64}
const DiscreteLidarPOMDP{SS, AS} = RoombaPOMDP{SS, AS, DiscreteLidar, Int}

# access the mdp of a RoombaModel
mdp(e::RoombaMDP) = e
mdp(e::RoombaPOMDP) = e.mdp

# access the room of a RoombaModel
room(m::RoombaMDP) = m.room
room(m::RoombaPOMDP) = room(m.mdp)

# access the state space of a RoombaModel
sspace(m::RoombaMDP{SS}) where SS = m.sspace
sspace(m::RoombaPOMDP{SS}) where SS = sspace(m.mdp)

dsspace(m::RoombaMDP{SS}) where SS = m.dsspace
dsspace(m::RoombaPOMDP{SS}) where SS = dsspace(m.mdp)

# RoombaPOMDP Constructor
function RoombaPOMDP(sensor, mdp::RoombaMDP{SS,AS}) where {SS, AS}
    RoombaPOMDP{SS, AS, typeof(sensor), obstype(sensor)}(sensor, mdp)
end

RoombaPOMDP(;sensor=Bumper(), mdp=RoombaMDP()) = RoombaPOMDP(sensor,mdp)

# function to determine if there is contact with a wall
wall_contact(e::RoombaModel, state) = wall_contact(mdp(e).room, SVec2(state.x, state.y))

POMDPs.actions(m::RoombaModel) = mdp(m).aspace
n_actions(m::RoombaModel) = length(mdp(m).aspace)

# maps a RoombaAct to an index in a RoombaModel with discrete actions
function POMDPs.actionindex(m::RoombaModel, a::RoombaAct)
    if mdp(m)._amap !== nothing
        return mdp(m)._amap[a]
    else
        error("Action index not defined for continuous actions.")
    end
end

# function to get goal xy location for heuristic controllers
function get_goal_xy(m::RoombaModel)
    grn = mdp(m).room.goal_rect
    gwn = mdp(m).room.goal_wall
    gr = mdp(m).room.rectangles[grn]
    corners = gr.corners
    if gwn == 4
        return (corners[1,:] + corners[4,:]) / 2.
    else
        return (corners[gwn,:] + corners[gwn+1,:]) / 2.
    end
end

# transition Roomba state given curent state and action
POMDPs.transition(m::RoombaPOMDP, s::RoombaState, a::RoombaAct) = transition(m.mdp, s, a)
POMDPs.transition(m::RoombaMDP{SS}, s::RoombaState, a::RoombaAct) where SS <: ContinuousRoombaStateSpace = Deterministic(get_next_state(m, s, a))

function POMDPs.transition(m::RoombaMDP{SS}, s::RoombaState, a::RoombaAct) where SS <: DiscreteRoombaStateSpace
    # round the states to nearest grid point
    si = stateindex(m, get_next_state(m, s, a))
    return Deterministic(index_to_state(m, si))
end

function get_next_x_y_th(m::RoombaMDP, x::Float64, y::Float64, th::Float64, a::RoombaAct)
    v, om = a
    v = clamp(v, 0.0, m.v_max)
    om = clamp(om, -m.om_max, m.om_max)

    # propagate dynamics without wall considerations
    dt = m.dt

    # dynamics assume robot rotates and then translates
    next_th = wrap_to_pi(th + om*dt)

    # make sure we arent going through a wall
    p0 = SVec2(x, y)
    heading = SVec2(cos(next_th), sin(next_th))
    des_step = v*dt
    next_x, next_y = legal_translate(m.room, p0, heading, des_step)
    return next_x, next_y, next_th
end

function get_next_state(m::RoombaMDP, s::FullRoombaState, a::RoombaAct)
    roomba = s.roomba
    next_x, next_y, next_th = get_next_x_y_th(m, roomba.x, roomba.y, roomba.theta, a)
    human = s.human
    next_h_x, next_h_y, next_h_th = get_next_x_y_th(m, human.x, human.y, human.theta, rand(actions(m)))

    # define next state
    rs = RoombaState(next_x, next_y, next_th)
    hs = HumanState(next_h_x, next_h_y, next_h_th)

    visited = deepcopy(s.visited)
    visited[position_to_index(m, roomba.x, roomba.y)] = 1.0
    
    return FullRoombaState(rs, hs, s.obstacles, visited)
end

# enumerate all possible states in a FullRoombaState with DiscreteRoombaStateSpace
function POMDPs.states(m::RoombaModel)
    ss = dsspace(m)
    x_states = range(ss.XLIMS[1], stop=ss.XLIMS[2], step=ss.x_step)
    y_states = range(ss.YLIMS[1], stop=ss.YLIMS[2], step=ss.y_step)
    th_states = range(-pi, stop=pi, step=ss.th_step)
    roomba_states = vec(collect(RoombaState(x,y,th) for x in x_states, y in y_states, th in th_states))
    human_states = vec(collect(Human(x,y,th) for x in x_states, y in y_states, th in th_states))
    obstacle_states = [ObstacleState(2, 3), ObstacleState(5, 3), ObstacleState(4, 7), ObstacleState(3, 7), ObstacleState(8, 9)]
    visited_states = get_possible_visited_states([], length(x_states)*length(y_states))
    return vec(collect(FullRoombaState(rs, hs, obstacle_states, vs) for rb in roomba_states, hs in human_states, vs in visited_states))
end

function get_possible_visited_states(visited_states, n::Int64)
    if n == 0
        return visited_states
    elseif length(visited_states) == 0
        return get_possible_visited_states([[0], [1]], n-1)
    else
        new_visited_states = []
        for state in visited_states
            state1 = deepcopy(state)
            push!(state1, 0)
            push!(new_visited_states, state1)
            state2 = deepcopy(state)
            push!(state2, 1)
            push!(new_visited_states, state2)
        end
        return get_possible_visited_states(new_visited_states, n -1)
    end
end

function position_to_index(m::RoombaModel, x::Float64, y::Float64)
    ss = sspace(m)
    return round(x, RoundToZero) * ss.XLIMS[2] + round(y, RoundToZero)
end

function index_to_position(m::RoombaModel, index::Int64)
    ss = sspace(m)
    x = round(index/ss.XLIMS[2], RoundToZero)
    y = mod(index, ss.XLIMS[2])
    return x, y
end

POMDPs.states(m::RoombaModel{SS}) where SS <: ContinuousRoombaStateSpace = sspace(m)

# return the number of states in a DiscreteRoombaStateSpace
function n_states(m::RoombaModel)
    ss = dsspace(m)
    return prod((convert(Int, diff(ss.XLIMS)[1]/ss.x_step)+1,
                        convert(Int, diff(ss.YLIMS)[1]/ss.y_step)+1,
                        3))
end

# map a RoombaState to an index in a DiscreteRoombaStateSpace
function POMDPs.stateindex(m::RoombaModel, s::RoombaState)
    ss = dsspace(m)
    xind = floor(Int, (s[1] - ss.XLIMS[1]) / ss.x_step + 0.5) + 1
    yind = floor(Int, (s[2] - ss.YLIMS[1]) / ss.y_step + 0.5)
    xind + ss.indices[1] * yind
end

# map an index in a DiscreteRoombaStateSpace to the corresponding x, y position
function index_to_state(m::RoombaModel, si::Int)
    ss = dsspace(m)
    yi, xi = divrem(si, ss.indices[1])

    x = ss.XLIMS[1] + (xi-1) * ss.x_step
    y = ss.YLIMS[1] + yi * ss.y_step

    return x, y
end

# defines reward function R(s,a,s')
function POMDPs.reward(m::RoombaModel,
                s::FullRoombaState,
                a::RoombaAct,
                sp::FullRoombaState)

    # penalty for each timestep elapsed
    cum_reward = mdp(m).time_pen

    # penalty for bumping into wall (not incurred for consecutive contacts)
    previous_wall_contact = wall_contact(m,s.roomba)
    current_wall_contact = wall_contact(m,sp.roomba)
    if(!previous_wall_contact && current_wall_contact)
        cum_reward += mdp(m).contact_pen
    end

    # penalty for bumping into obstacles
    for obs in sp.obstacles
        if sp.roomba.x == obs.x && sp.roomba.y == obs.y
            cum_reward += mdp(m).contact_pen
        end
    end

    # penalty for bumping into human
    if sp.roomba.x == sp.human.x && sp.roomba.y == sp.human.y
        cum_reward += mdp(m).contact_pen * 5
    end

    # reward for covering new area
    cum_reward += 2* (sum(sp.visited) - sum(s.visited))

    return cum_reward
end

# determine if a terminal state has been reached
# POMDPs.isterminal(m::RoombaModel, s::RoombaState) = (sum(s.visited) == (length(states(m))-num_obs) || (s.x == s.human.x && s.y == s.human.y))
POMDPs.isterminal(m::RoombaModel, s::RoombaState) = abs(s.status) > 0.0

# Bumper POMDP observation
function POMDPs.observation(m::BumperPOMDP,
                            a::RoombaAct,
                            sp::RoombaState)
    return Deterministic(wall_contact(m, sp)) # in {0.0,1.0}
end

n_observations(m::BumperPOMDP) = 2
POMDPs.observations(m::BumperPOMDP) = [false, true]

# Lidar POMDP observation
function lidar_obs_distribution(m::RoombaMDP, ray_stdev::Float64, sp::FullRoombaState)
    x, y, th = sp.x, sp.y, sp.theta
    # determine uncorrupted observation
    rl = ray_length(m.room, SVec2(x, y), SVec2(cos(th), sin(th)))
    # compute observation noise
    sigma = ray_stdev * max(rl, 0.01)
    # disallow negative measurements
    return Truncated(Normal(rl, sigma), 0.0, Inf)
end

POMDPs.observation(m::LidarPOMDP, a::RoombaAct, sp::FullRoombaState) = lidar_obs_distribution(mdp(m), sensor(m).ray_stdev, sp)

function n_observations(m::LidarPOMDP)
    error("n_observations not defined for continuous observations.")
end

function POMDPs.observations(m::LidarPOMDP)
    error("LidarPOMDP has continuous observations. Use DiscreteLidarPOMDP for discrete observation spaces.")
end

# DiscreteLidar POMDP observation
function POMDPs.observation(m::DiscreteLidarPOMDP{SS, AS},
                            a::AbstractVector{Float64},
                            sp::AbstractVector{Float64}) where {SS, AS}

    s = sensor(m)
    d = lidar_obs_distribution(mdp(m), s.ray_stdev, sp)

    # discretize observations
    interval_start = 0.0
    d_disc = s._d_disc
    for i in 1:length(s.disc_points)
        interval_end = cdf(d, s.disc_points[i])
        d_disc[i] = interval_end - interval_start
        interval_start = interval_end
    end
    d_disc[end] = 1.0 - interval_start

    return SparseCat(1:length(d_disc), d_disc)
end

n_observations(m::DiscreteLidarPOMDP) = length(sensor(m).disc_points) + 1
POMDPs.observations(m::DiscreteLidarPOMDP) = vec(1:n_observations(m))

# define discount factor
POMDPs.discount(m::RoombaModel) = mdp(m).discount

# struct to define an initial distribution over Roomba states
struct RoombaInitialDistribution{M<:RoombaModel}
    m::M
end

# definition of initialstate for Roomba environment
POMDPs.initialstate(m::RoombaModel) = RoombaInitialDistribution(m)

function get_a_random_state(m::RoombaMDP, rng::AbstractRNG)
    x, y = init_pos(m.room, rng)
    th = rand(rng) * 2*pi - pi
    roomba = RoombaState(x, y, th)

    h_x, h_y = init_pos(m.room, rng)
    h_th = rand(rng) * 2*pi - pi
    human = HumanState(h_x, h_y, h_th)
    obstacles = [ObstacleState(2, 3), ObstacleState(5, 3), ObstacleState(4, 7), ObstacleState(3, 7), ObstacleState(8, 9)]
    visited = zeros(n_states(m))
    return FullRoombaState(roomba, human, obstacles, visited)
end

function Base.rand(rng::AbstractRNG, d::RoombaInitialDistribution{<:RoombaModel{SS}}) where SS <: DiscreteRoombaStateSpace
    return get_a_random_state(mdp(d.m), rng)
end

Base.rand(rng::AbstractRNG, d::RoombaInitialDistribution{<:RoombaModel{SS}}) where SS <: ContinuousRoombaStateSpace = get_a_random_state(mdp(d.m), rng)

# Render a room and show robot
function render(ctx::CairoContext, m::RoombaModel, step)
    env = mdp(m)
    state = step[:sp]

    radius = ROBOT_W.val*6

    # render particle filter belief
    if haskey(step, :bp)
        bp = step[:bp]
        if bp isa AbstractParticleBelief
            for p in particles(bp)
                x, y = transform_coords(SVec2(p[1],p[2]))
                arc(ctx, x, y, radius, 0, 2*pi)
                set_source_rgba(ctx, 0.6, 0.6, 1, 0.3)
                fill(ctx)
            end
        end
    end

    # Render room
    render(env.room, ctx)

    # Find center of robot in frame and draw circle
    x, y = transform_coords(SVec2(state[1],state[2]))
    arc(ctx, x, y, radius, 0, 2*pi)
    set_source_rgb(ctx, 1, 0.6, 0.6)
    fill(ctx)

    # Draw line indicating orientation
    move_to(ctx, x, y)
    end_point = SVec2(state[1] + ROBOT_W.val*cos(state[3])/2, state[2] + ROBOT_W.val*sin(state[3])/2)
    end_x, end_y = transform_coords(end_point)
    line_to(ctx, end_x, end_y)
    set_source_rgb(ctx, 0, 0, 0)
    stroke(ctx)
    return ctx
end

# this object should have show methods for a variety of mimes
# in particular, for now it has both png and html-like
# it would also give us a ton of hacker cred to make an ascii rendering
struct RoombaVis
    m::RoombaModel
    step::Any
    text::String
end

render(m::RoombaModel, step; text::String="") = RoombaVis(m, step, text)

function Base.show(io::IO, mime::Union{MIME"text/html", MIME"image/svg+xml"}, v::RoombaVis)
    c = CairoSVGSurface(io, 800, 600)
    ctx = CairoContext(c)
    render(ctx, v.m, v.step)
    finish(c)
end

function Base.show(io::IO, mime::MIME"image/png", v::RoombaVis)
    c = CairoRGBSurface(800, 600)
    ctx = CairoContext(c)
    render(ctx, v.m, v.step)
    # finish(c) # doesn't work with this; I wonder why
    write_to_png(c, io)
end
