param map = localPath('../maps/CARLA/Town04.xodr')
param carla_map = 'Town04'
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
    with blueprint "vehicle.jeep.wrangler_rubicon",
    with rolename "hero"
vehicle2 = Car right of spot by 2,
    with blueprint "vehicle.audi.etron",
    with heading 30 deg relative to spot.heading,
    with regionContainedIn None,
    with behavior Vehicle2Behavior(V2_MIN_SPEED, THRESHOLD)

require (distance to intersection) > 30
terminate when (distance to spot) > 100 or (distance to vehicle2) > 50
# This crash occurred on a straight one-way asphalt street with an uphill grade of 3% and a posted speed limit was 32 Kmph (20 MPH.) Each side of the street was lined with parked cars. It was late evening on a weeknight and there were no adverse weather conditions. The street was not lighted. Vehicle 1, a 1989 Jeep Wagoneer, was traveling East with the driver and one passenger inside. V2, an unoccupied 2003 Chevrolet Tahoe, was legally parked at the left road edge. As V2 entered the roadway, the left front of Vehicle 1 contacted the right side of V2. The Jeep pushed the Tahoe 1.7m east before both vehicles came to rest. Vehicle 1 was driven by a 25-year-old male. The driver and front right passenger were transported to a medical facility for treatment. The driver stated that he was looking at his wife as he was speaking to her. He stated that he was distracted and hit the other vehicle. The driver indicated that, he had recently had his brakes worked on and they weren't quite right. While he was not comfortable with the condition of his vehicle's brakes, he was uncertain if this contributed to the crash. The Critical Precrash Event for Vehicle 1 occurred when the vehicle traveled left, over the left edge of the travel lane and into the parking lane. The Critical Reason for the Precrash Event was an internal distraction.