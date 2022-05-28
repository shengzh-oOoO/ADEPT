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
# This case focuses on a restrained 43 year old female driver in a 2010 Honda Accord (V1). V1 was equipped with dual frontal airbags, dual side curtains and dual first row side airbags. The driver's steering mounted airbag and both side curtains deployed during the crash. V2, a 2011 Hyundai Genesis was also involved in the crash. The crash occurred during the nighttime hours on a wet asphalt, two-way, two-lane roadway. There were no street lights near the roadway and it was dark. The posted speed limit for both travel lanes was 64 kph (40 mph). V1 was traveling south in the southbound lane of the roadway. V2 was traveling north in the northbound lane while negotiating a right curve. V2 drifted over the left lane line and then its left tires departed the roadway onto the left roadside. The driver of V2 noticed that his vehicle was partially off the roadway. He steered right and braked in an attempt to slow and maneuver his vehicle back onto the road. The driver of V1 noticed V2 in its lane and applied her brakes. The front of V2 contacted the front of V1 while in the southbound lane. V1 rotated counterclockwise onto the adjacent roadside. V1 traveled a short distance down an embankment where it contacted a tree with its right rear. V1 came to rest on the roadside facing east. V2 rotated counterclockwise after impact and traveled across the roadway onto the roadside that was adjacent to the northbound lane. V2 faced north at final rest. Both vehicles were towed from the crash scene due to disabling damage. The restrained driver of V1 was transported to a local hospital by ground ambulance. She was later transferred to a level one trauma center due to serious injuries sustained in the crash.