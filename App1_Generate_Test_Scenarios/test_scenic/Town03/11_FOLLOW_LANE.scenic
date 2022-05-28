param map = localPath('../maps/CARLA/Town03.xodr')
param carla_map = 'Town03'
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
    with blueprint "vehicle.chevrolet.impala",
    with rolename "hero"
vehicle2 = Car left of spot by 2,
    with blueprint "vehicle.lincoln.mkz2017",
    with heading -30 deg relative to spot.heading,
    with regionContainedIn None,
    with behavior Vehicle2Behavior(V2_MIN_SPEED, THRESHOLD)

require (distance to intersection) > 30
terminate when (distance to spot) > 100 or (distance to vehicle2) > 50
# This crash occurred in the evening hours on the bituminous northbound lanes of a divided urban highway. The two northbound were dry, level, curved to the right, and featured shoulders on both sides as well as a posted speed limit of 80kmph (50mph). There were no adverse atmospheric conditions and the roadway was dark but lighted with normal traffic flow. V1, a 2000 Mazda 626, was traveling north in lane one when its left side was contacted by V2, an unknown type van, as it re-entered the roadway from the shoulder. The initial impact caused V1 to lose control of her vehicle and began to swerve off the road to the left. She drove a few feet up the embankment on the side of the road digging her left side tires into the grass/dirt causing them to blow out and damage her undercarriage. V1 abruptly steered right descended down the hill and began to rollover. V1 re-entered the roadway still turning clockwise and rolled over two quarter turns onto its top coming to final rest facing south. V2 continued traveling and didn't stop. V1 was driven a 25-year old female in good health on her way home. She stated she didn't expect a vehicle (V2) to be traveling on the shoulder. When she finally saw V2 encroaching from the left shoulder into her lane. Both driver and passenger were transported to the hospital with moderate injuries and V1 was towed due to damages. The Critical Pre-Crash Event for V1 was coded other motor vehicle encroachment from adjacent lane (same direction) - over left lane line. The only associated factor coded to this driver was fatigue; she said that she'd worked 12hrs that day (but claimed she wasn't tired). V1 was not thought to have contributed to this crash. V2 left the scene without stopped therefore no driver interview or vehicle inspection could be obtained. The Critical Precrash Event for V2 was coded as this vehicle traveling, over the lane line on the right side of the travel lane. The Critical Reason for the Critical Precrash Event was coded as an unknown decision error. There are no known associated factors coded to this vehicle, as the driver wasn't interviewed.