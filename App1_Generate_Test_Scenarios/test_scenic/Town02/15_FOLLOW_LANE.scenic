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

ego = Car following roadDirection from spot for -20,
    with blueprint "vehicle.lincoln.mkz2017",
    with rolename "hero"
vehicle2 = Car left of spot by 3,
    with blueprint "vehicle.nissan.patrol",
    with heading -180 deg relative to spot.heading,
    with regionContainedIn None,
    with behavior Vehicle2Behavior(V2_MIN_SPEED, THRESHOLD)

require (distance to intersection) > 50
terminate when (distance to spot) > 100 or (distance to vehicle2) > 100
# Vehicle one (V1 - case vehicle), a 2002 Subaru Forester, 4-door utility vehicle was traveling west in the westbound lane of a two-lane, two-way roadway and was negotiating a right curve. Vehicle two (V2), a 2002 Buick Rendezvous, 4-door utility vehicle was traveling east in the eastbound lane of the same roadway and was negotiating a left curve. It was daylight, snowing, and the bituminous road was icy. The driver of V1 lost control on the icy road and began to rotate counterclockwise. V1 rotated approximately 90 degrees, crossed the centerline and entered the eastbound lane. The driver of V2 could not avoid V1 and the front of V2 struck the right side of V1 in a T-type configuration. Both vehicles were towed due to disabling vehicle damage. The 36-year-old male driver of V1 (case occupant) was using the available three-point seat belt but no airbags deployed in the driver's seating position. He was transported via ground ambulance to a regional level-one trauma center.