param map = localPath('../maps/CARLA/TOWN.xodr')
param carla_map = 'TOWN'
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
    try:
        do CrossingBehavior(ego, min_speed, threshold)
    interrupt when withinDistanceToAnyCars(self, 10) or ego.speed == 0:
        do FollowLaneBehavior(EGO_SPEED)

lane = Uniform(*network.lanes)

spot = OrientedPoint on lane.centerline

ego = Car following roadDirection from spot for -DISTF-Range(0, 10),
    with blueprint V1_MODEL,
    with rolename "hero"
vehicle2 = Car V2_DIRE of spot by DISTLR,
    with blueprint V2_MODEL,
    with heading V2_DEG deg relative to spot.heading,
    with regionContainedIn None,
    with behavior Vehicle2Behavior(V2_MIN_SPEED, THRESHOLD)

require (distance to intersection) > 30
terminate when (distance to spot) > 100 or (distance to vehicle2) > 50
