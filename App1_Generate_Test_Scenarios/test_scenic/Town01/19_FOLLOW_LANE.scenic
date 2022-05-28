param map = localPath('../maps/CARLA/Town01.xodr')
param carla_map = 'Town01'
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
vehicle2 = Car right of spot by 2,
    with blueprint "vehicle.audi.etron",
    with heading 30 deg relative to spot.heading,
    with regionContainedIn None,
    with behavior Vehicle2Behavior(V2_MIN_SPEED, THRESHOLD)

require (distance to intersection) > 30
terminate when (distance to spot) > 100 or (distance to vehicle2) > 50
# Vehicle one (V1-case vehicle), a 2007 Mercedes-Benz ML-350, 4-door SUV was traveling east in the eastbound lane of a two-lane, two-way residential street. Vehicle two (V2), a 2002 Mitsubishi Montero, 4-door SUV was traveling west in the westbound lane of the same street. It was daylight, and the bituminous road surface was dry except for an icy patch in the westbound lane of travel. There were no adverse weather conditions at the time of the crash. As V2 was exiting a left curve in the westbound roadway, it lost control on an icy patch and crossed over into the eastbound lane, impacting the front of V1 with its front right . After initial impact, V1 rotated clockwise and slightly rearward before traveling to its final resting position, partially off the west road-edge facing southeast. V2 also rotated clockwise and traveled to its final resting position, off the roadway in a driveway facing northwest. Both vehicles were towed from the scene due to disabling vehicle damage. There were two occupants in V1, a 33 year-old female driver and a 21 month-old male (case occupant). The driver of V1 was properly restrained by the available lap and shoulder belt (pretensioner fired) and the deployed steering wheel mounted air bag. Our case occupant was seated in the 2nd row, right seating position in a Britax Marathon forward facing child safety seat with internal 5-point harness. The child safety seat was properly restrained by the LATCH and Tether system. Our case occupant was transported from the scene to a local Pediatric trauma center where he was diagnosed with a left tibia fracture.