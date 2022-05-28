param map = localPath('../maps/CARLA/Town07.xodr')
param carla_map = 'Town07'
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

ego = Car following roadDirection from spot for -10-Range(0, 10),
    with blueprint "vehicle.tesla.cybertruck",
    with rolename "hero"
vehicle2 = Car right of spot by 2,
    with blueprint "vehicle.toyota.prius",
    with heading 30 deg relative to spot.heading,
    with regionContainedIn None,
    with behavior Vehicle2Behavior(V2_MIN_SPEED, THRESHOLD)

require (distance to intersection) > 30
terminate when (distance to spot) > 100 or (distance to vehicle2) > 50
# This crash occurred on a two-lane asphalt roadway. It was clear, dry weekend morning. The speed limit was 48 kmph (30mph) and there was a 2.8% uphill grade. Vehicle 1 was a 2004 Nissan Frontier driven by an 18 year-old female. Vehicle 1 was traveling east along a slight left curve that had a radius of 1125m. Vehicle 2 was an unoccupied 2005 Ford Crown Victoria that was legally parked at the right (south) curb heading east. The front of Vehicle 1 contacted the back of Vehicle 2. Both vehicles traveled 25m east before coming to rest at the right curb. The Critical Precrash Event for Vehicle 1 was this vehicle traveling over the lane line on the right side of travel lane. The Critical Reason for the Critical Precrash Event was an internal distraction. The driver of Vehicle 1 told police that she was distracted by an insect inside the vehicle.