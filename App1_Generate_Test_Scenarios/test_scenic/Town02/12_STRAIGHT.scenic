param map = localPath('../maps/CARLA/Town02.xodr')
param carla_map = 'Town02'
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

adv_maneuvers = filter(lambda i: i.type == ManeuverType.LEFT_TURN, ego_maneuver.conflictingManeuvers)
adv_maneuver = Uniform(*adv_maneuvers)
adv_trajectory = [adv_maneuver.startLane, adv_maneuver.connectingLane, adv_maneuver.endLane]

ego_spawn_pt = OrientedPoint in ego_maneuver.startLane.centerline
adv_spawn_pt = OrientedPoint in adv_maneuver.startLane.centerline

ego = Car at ego_spawn_pt,
    with blueprint "vehicle.charger2020.charger2020",
    with rolename "hero"

adversary = Car at adv_spawn_pt,
    with blueprint "vehicle.lincoln.mkz2017",
    with behavior AdversaryBehavior(adv_trajectory)

require 15 <= (distance to intersec) <= 20
require 10 <= (distance from adversary to intersec) <= 15
terminate when (distance to ego_spawn_pt) > 70# The crash occurred in the southwestbound lanes of a three-lane roadway with a T-intersection. Conditions were dark with lights, dry with cloudy skies in the evening on a weekday. The southwestbound lanes had a posted speed of 80 kmph (50 mph), the southbound lane had a speed limit of 56 kmph (35 mph). V1 was a 1998 Ford Escort sedan traveling southbound in the right lane approaching a T-intersection with a stop sign. V2, a 1993 Subaru Legacy sedan, was traveling southwestbound in the right lane approaching the T-intersection. V1 made a left turn at the stop sign and the front of V2 struck the left side of V1 in the southwestbound lane. Both vehicles came to final rest facing westbound. V1 was driven by a 19-year-old male who was not injured. He stated that he was lost and trying to find a church that was close by. He approached the stop sign and did not see V2 until it was too late. He turned left directly into V2 traveling between 21-30 MPH. V1 was towed due to damage. The Critical Precrash Event for V1 was when this vehicle was turning left at the intersection. The Critical Reason for the Critical Event was inadequate surveillance; he looked but did not see. Associated factors coded to this driver was the driver being fatigued due to a busy work and exercise schedule, the driver not familiar with the area (he was lost), and traveling too fast (he was in a hurry). The driver has myopic (near-sighted) vision but does not need to wear glasses to drive, this was not thought to have contributed to the crash. V2 was driven by a 63-year-old female who was transported to a local hospital with minor injuries. The driver refused to be interviewed at the hospital and refused us when we visited her home. V2 remained at the scene. The Critical Precrash Event for V2 was when V1 encroached from the crossing street, turning into the opposite direction. No Critical Reason was coded to the driver of V2.