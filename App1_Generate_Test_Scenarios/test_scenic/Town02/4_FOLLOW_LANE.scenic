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
    with blueprint "vehicle.toyota.prius",
    with rolename "hero"
vehicle2 = Car left of spot by 3,
    with blueprint "vehicle.lincoln.mkz2017",
    with heading -180 deg relative to spot.heading,
    with regionContainedIn None,
    with behavior Vehicle2Behavior(V2_MIN_SPEED, THRESHOLD)

require (distance to intersection) > 50
terminate when (distance to spot) > 100 or (distance to vehicle2) > 100
# This crash occurred at an unknown time early in the morning, sometime after midnight. The driver was unable to give a time, and the time the crash was reported was the time a passerby saw the vehicle in the light of dawn. Vehicle one (V1) was a 1984 white 4 door Oldsmobile Cutlass that struck the side of a building at a high rate of speed. V1 came from a one-lane roadway inside of a trailer court. This vehicle crossed a 2 lane bituminous roadway before entering a parking lot and crashing into the side of a building. The vehicle was towed and the driver was transported and admitted for a dislocated right hip. The driver is a 53-year-old male. Medical records states the driver admitted to smoking methamphetamine's as well as drinking alcohol earlier. Tests were positive for amphetamines and negative for alcohol. The driver stated he got into his vehicle to go see his brother about some problem. The driver had known about an accelerator problem before the crash. The vehicle would not remain running unless the driver held their foot on the gas and then putting the vehicle into gear. At the time of the crash the driver did just this, but according to the driver the accelerator stuck at full throttle. The driver, due to his altered conscious state, failed to control the vehicle as it sped down the street into the side of the building. The critical pre-crash event for V1 was coded: this vehicle loss of control due to - non-disabling vehicle problem. The critical reason was coded as a vehicle related factor: other vehicle failure (specified as: accelerator stuck).