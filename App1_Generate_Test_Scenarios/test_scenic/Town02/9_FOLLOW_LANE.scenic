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
    try:
        do CrossingBehavior(ego, min_speed, threshold)
    interrupt when withinDistanceToAnyCars(self, 10) or ego.speed == 0:
        do FollowLaneBehavior(EGO_SPEED)

lane = Uniform(*network.lanes)

spot = OrientedPoint on lane.centerline

ego = Car following roadDirection from spot for -10-Range(0, 10),
    with blueprint "vehicle.audi.etron",
    with rolename "hero"
vehicle2 = Car left of spot by 2,
    with blueprint "vehicle.tesla.cybertruck",
    with heading -30 deg relative to spot.heading,
    with regionContainedIn None,
    with behavior Vehicle2Behavior(V2_MIN_SPEED, THRESHOLD)

require (distance to intersection) > 30
terminate when (distance to spot) > 100 or (distance to vehicle2) > 50
# This crash occurred on a two lane undivided straight roadway with no grade and a posted speed limit of 89 KMPH (55 MPH). The weather was clear and the roadway was dry. This crash occurred on late morning on a weekday. Vehicle #1,2000 Ford Expedition was traveling south on the two lane undivided roadway when it impacted the rear of Vehicle #2, a 2004 Dodge Ram 1500, with its front. Vehicle # 1 came to rest on the southbound shoulder. Vehicle #2 came to rest in a north side parking lot. Vehicle #2 had been stopped in the northbound traveling lane waitng for traffic to pass and intending to make a left turn into the parking lot. The Ford Expedition, (Vehicle #1) was driven by a 34 year old belted male who was examined at the scene and refused further medical attention. He stated he was looking at an exterior site, to the right of his vehicle, and did not realize the vehicle in front of him was stopped. The Critical Precrash Event coded to Vehicle #1 was coded as Other Motor Vehicle in Lane, Other vehicle stopped. The Critical Reason for the Critical Precrash Event was coded as Driver Related Factor, Recognition Error, External Distraction. An associated factor coded to this driver was the use of prescription medications. He was taking two medications: (a) a depression medication with the possible side effects of drowsiness and (b) a pain medication also with the possible side effects of drowsiness. The Dodge Ram (Vehicle #2) was driven by a 38 year old belted female with a belted 16 year old female passenger in the right front seat. The driver stated she was stopped in the roadway waiting for several vehicles to pass before proceeding to make a left turn into a parking lot. She stated she did not see Vehicle #2 approaching and had no time to react. No Critical Precrash Event was coded to this vehicle as the driver was not thought to have contributed to the crash. Both Vehicles were towed from the scene. Police did not consider illegal drugs or alcohol to be involved and ordered no tests for either driver.