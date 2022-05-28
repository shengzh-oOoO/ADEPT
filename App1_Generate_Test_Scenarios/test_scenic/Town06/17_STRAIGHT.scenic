param map = localPath('../maps/CARLA/Town06.xodr')
param carla_map = 'Town06'
param render = '0'
model scenic.simulators.carla.model

EGO_SPEED = 10
SAFETY_DISTANCE = 20
BRAKE_INTENSITY = 1.0

monitor StopAfterTimeInIntersection:
    totalTime = 0
    while totalTime < 1000:
        totalTime += 1
        wait
    terminate

monitor TrafficLights:
    freezeTrafficLights()
    while True:
        if withinDistanceToTrafficLight(ego, 100):
            setClosestTrafficLightStatus(ego, "green")
        if withinDistanceToTrafficLight(adversary, 100):
            setClosestTrafficLightStatus(adversary, "red")
        wait

behavior AdversaryBehavior(trajectory):
    while (ego.speed < 1):
        wait
    do FollowTrajectoryBehavior(trajectory=trajectory)

fourWayIntersection = filter(lambda i: i.is4Way and i.isSignalized, network.intersections)

intersec = Uniform(*fourWayIntersection)
ego_startLane = Uniform(*intersec.incomingLanes)

ego_maneuvers = filter(lambda i: i.type == ManeuverType.STRAIGHT, ego_startLane.maneuvers)
ego_maneuver = Uniform(*ego_maneuvers)
# ego_trajectory = [ego_maneuver.startLane, ego_maneuver.connectingLane, ego_maneuver.endLane]

adv_maneuvers = filter(lambda i: i.type == ManeuverType.STRAIGHT, ego_maneuver.conflictingManeuvers)
adv_maneuver = Uniform(*adv_maneuvers)
adv_trajectory = [adv_maneuver.startLane, adv_maneuver.connectingLane, adv_maneuver.endLane]

ego_spawn_pt = OrientedPoint in ego_maneuver.startLane.centerline
adv_spawn_pt = OrientedPoint in adv_maneuver.startLane.centerline

ego = Car at ego_spawn_pt,
    with blueprint "vehicle.lincoln2020.mkz2020",
    with rolename "hero"

adversary = Car at adv_spawn_pt,
    with blueprint "vehicle.bmw.grandtourer",
    with behavior AdversaryBehavior(adv_trajectory)

require 15 <= (distance to intersec) <= 20
require 10 <= (distance from adversary to intersec) <= 15
terminate when (distance to ego_spawn_pt) > 70# This case focuses on a restrained 70 year old female right front passenger in a 2012 Toyota Camry (V1). There was also a restrained 72 year old male driver in V1. V1 was equipped with dual frontal airbags. It was also equipped dual side curtains, dual first and second row side airbags, and dual frontal knee bolster airbags. All left side airbags deployed during the crash. V2, a 2001 Chevrolet Malibu was also involved in the crash. The crash occurred during the daylight hours at the intersection of a two-way, two-lane, and dry asphalt roadway and a four-lane, two-way, dry, asphalt roadway. The two-lane roadway (east and westbound lanes) was controlled by stop signs and the four-lane roadway (north and southbound lanes) was not controlled by signs or signals. The posted speed limit for two-lane roadway was 56 kph (35 mph) and the four-lane roadway posted speed limit was 89 kph (55mph). V1 was traveling east in the eastbound lane attempting to crossover the intersection onto the east leg of the intersection. V2 was traveling south in the southbound lane attempting to crossover the intersection onto the south leg of the intersection. While both vehicles were in the intersection, the front of V2 contacted the left side of V1. After impact V1 continued southeast across the intersection while rotating counterclockwise. It then departed the roadway onto the southeast corner of the intersection and came to rest in the parking lot of a gas station. V1 faced southeast at final rest. V2 rotated counterclockwise after impact and continued across the intersection. V2 came to rest facing northeast while partially in its original travel lane and partially off the roadway. Both vehicles were towed from the crash scene due to disabling damage. The right front passenger in V1 was taken to a local hospital by ambulance; however, she was later transported to a level one trauma center where she received additional treatment for serious injury. The driver of V1 was transported to a local hospital where he was hospitalized and treated for unknown injuries.