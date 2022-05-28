param map = localPath('../maps/CARLA/Town02.xodr')
param carla_map = 'Town02'
param render = '0'
model scenic.simulators.carla.model

EGO_SPEED = 10

V2_MIN_SPEED = 1
THRESHOLD = 15

monitor StopAfterTimeInIntersection:
    totalTime = 0
    while totalTime < 1000:
        totalTime += 1
        wait
    terminate

behavior Vehicle2Behavior(min_speed=1, threshold=10):
    while (ego.speed <= 0.1):
        wait
    do FollowLaneBehavior(EGO_SPEED)

lane = Uniform(*network.lanes)

spot = OrientedPoint on lane.centerline

ego = Car following roadDirection from spot for -DISTF,
    with blueprint V1_MODEL,
    with rolename "hero"
vehicle2 = Car left of spot by 3,
    with blueprint V2_MODEL,
    with heading -180 deg relative to spot.heading,
    with regionContainedIn None,
    with behavior Vehicle2Behavior(V2_MIN_SPEED, THRESHOLD)

require (distance to intersection) > 50
terminate when (distance to spot) > 100 or (distance to vehicle2) > 100
