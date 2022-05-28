param map = localPath('../maps/CARLA/Town06.xodr')
param carla_map = 'Town06'
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
    with blueprint "vehicle.charger2020.charger2020",
    with rolename "hero"
vehicle2 = Car right of spot by 2,
    with blueprint "vehicle.lincoln2020.mkz2020",
    with heading 30 deg relative to spot.heading,
    with regionContainedIn None,
    with behavior Vehicle2Behavior(V2_MIN_SPEED, THRESHOLD)

require (distance to intersection) > 30
terminate when (distance to spot) > 100 or (distance to vehicle2) > 50
# The crash occurred on a two-way, two-lane straight, level, bituminous residential street with a speed limit of 40 kmph (25 mph). Conditions at the time of the weeknight crash were cloudy, dark, and dry. V1, a 2001 Kia Sephia, driven by a 28 year-old female, was southbound on the road when it left the travel lane and the front of V1 struck the back of a legally parked, unoccupied vehicle on the right side of the road. V1 pushed the parked vehicle 25 meters where V1 came to rest against it. Both vehicles were towed from the scene due to damage. The driver of v1 was transported to the Police Department to undergo a Breathalyzer, where she recorded a Blood Alcohol level of .21. The Critical Precrash Event coded to v1 was 'this vehicle traveling over the lane line on the right side of travel lane.' The Critical Reason for the Critical Event coded was 'internal distraction.' The driver stated during the interview that she heard something fall and looked down and reached with her right hand to get it, and ran into the parked car. A contributing factor was coded for alcohol consumption. The driver stated that she had been drinking at home two miles away and was driving to the store when she had the crash.