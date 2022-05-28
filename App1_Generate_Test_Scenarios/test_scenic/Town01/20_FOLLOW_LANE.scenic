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
    with blueprint "vehicle.bmw.isetta",
    with rolename "hero"
vehicle2 = Car left of spot by 2,
    with blueprint "vehicle.audi.tt",
    with heading -30 deg relative to spot.heading,
    with regionContainedIn None,
    with behavior Vehicle2Behavior(V2_MIN_SPEED, THRESHOLD)

require (distance to intersection) > 30
terminate when (distance to spot) > 100 or (distance to vehicle2) > 50
# The focus of this case a 17-year old female who was the restrained driver (case occupant) of a 2008 Chevrolet Cobalt 2-door coupe involved in a near-side impact with a utility pole. The single vehicle collision occurred during the early evening hours on a four-lane, two-way roadway (two northbound lanes, two southbound lanes). It was daylight, but the weather was cloudy. The roadway was dry and free of defects. The posted speed limit for this roadway is 80 km/h (50 mph). Vehicle one (V1-case vehicle) was traveling southbound in the left lane when an unknown vehicle pulled out of a driveway, and proceeded to travel in the lane in front of V1 at a slower speed. The driver of V1 avoided contacting the unknown vehicle by steering to the left, crossing into the northbound travel lane. She then steered sharply to the right to reenter the southbound lane, causing the vehicle to yaw to the left, crossing both southbound lanes and departing the west road edge before impacting a utility pole with its left side (pole has been replaced in scene images). V1 rotated slightly counter clockwise around the pole before coming to final rest facing west. V1 left approximately 18 meters of yaw mark on the southbound travel lanes from its left front and left rear tires. V1 was towed from the scene due to disabling damage. The 17-year old female driver (case occupant) of V1 was using her available lap and shoulder belt (pretensioner fired) and the side curtain air bag did deploy in the crash. The driver was taken from the scene to a to a local hospital where she stayed for 6 hours in the ER before being transferred to a trauma center where she was admitted for three days due to her injuries. There was also a 13-year old female (case occupant's sister) seated in the right front of V1. She was wearing her available lap and shoulder belt (pretensioner fired) and no air bags deployed for her seating position. She was transported from the scene to a local hospital and released the same day. Of note, Event Data Recorder data collected from the case vehicle shows that the speed of travel immediately preceding the crash was much higher than the posted speed limit. EDR also confirms that both the case occupant and the right front passenger were utilizing their belt restraints.