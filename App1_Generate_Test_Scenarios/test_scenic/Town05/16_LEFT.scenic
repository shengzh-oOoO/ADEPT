param map = localPath('../maps/CARLA/Town05.xodr')
param carla_map = 'Town05'
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

ego_maneuvers = filter(lambda i: i.type == ManeuverType.LEFT_TURN, ego_startLane.maneuvers)
ego_maneuver = Uniform(*ego_maneuvers)
# ego_trajectory = [ego_maneuver.startLane, ego_maneuver.connectingLane, ego_maneuver.endLane]

adv_maneuvers = filter(lambda i: i.type == ManeuverType.STRAIGHT, ego_maneuver.conflictingManeuvers)
adv_maneuver = Uniform(*adv_maneuvers)
adv_trajectory = [adv_maneuver.startLane, adv_maneuver.connectingLane, adv_maneuver.endLane]

ego_spawn_pt = OrientedPoint in ego_maneuver.startLane.centerline
adv_spawn_pt = OrientedPoint in adv_maneuver.startLane.centerline

ego = Car at ego_spawn_pt,
    with blueprint "vehicle.mini.cooperst",
    with rolename "hero"

adversary = Car at adv_spawn_pt,
    with blueprint "vehicle.tesla.model3",
    with behavior AdversaryBehavior(adv_trajectory)

require 15 <= (distance to intersec) <= 20
require 10 <= (distance from adversary to intersec) <= 15
terminate when (distance to ego_spawn_pt) > 70# This case involves an oblique, side-impact to the corner of the left fender of a passenger car and the front of another passenger car. The case occupant is the restrained driver of the passenger car sustaining the oblique-angle, side-impact. The case vehicle (V1), a 2006 four-door, Ford Focus, was northbound on dry, level, two-lane, concrete road approaching a four-leg intersection. The roadway narrows on the approach to the intersection as the legal parking lanes end. Vehicle two (V2), a 2003 four-door Hyundai XG350, was in the outside eastbound lane of the intersecting five-lane road approaching the same intersection. It was daylight and clear. As the case vehicle was making a left-turn through the intersection, it entered the path of V2. The front of V2 struck the left fender of the case vehicle, at the front corner. The impact resulted in the case vehicle rotating clockwise until it came to rest on the east side of the intersection, near the raised median for east-west traffic, facing northeast. V2 yawed counterclockwise, crossed into the inside eastbound lane, exited the intersection, rode over a raised median and came to rest in the westbound left-turn lane facing northeast. The sole occupant of the case vehicle was the restrained 46-year-old male driver. He had the benefit of a deployed frontal-impact air bag. The driver sustained serious injuries and was transported to a level-one trauma center and enrolled as a case occupant.