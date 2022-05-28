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
    with blueprint "vehicle.lincoln.mkz2017",
    with rolename "hero"

adversary = Car at adv_spawn_pt,
    with blueprint "vehicle.mini.cooperst",
    with behavior AdversaryBehavior(adv_trajectory)

require 15 <= (distance to intersec) <= 20
require 10 <= (distance from adversary to intersec) <= 15
terminate when (distance to ego_spawn_pt) > 70# This near-side impact crash case involves a 31 year old female driver, wearing the available manual lap/shoulder belt with deployed frontal, side impact and side curtain air bags who sustained moderate injuries. This crash occurred during evening hours in a four legged intersection with stop signs present for north/southbound traffic. These intersecting streets are both two lane, two way asphalt roadways with no curbs. In the area where the crash occurred, there is an uphill grade for southbound vehicles approaching the intersection, then the travel lane levels off. The intersecting roadway is comprised of one eastbound lane, one westbound lane and left turn only lanes present at the intersection. The case vehicle one (V1), a 2013 Subaru Forester compact utility vehicle, was traveling south in the southbound lane and Vehicle 2 (V2), a 2012 Honda CR-V compact utility vehicle, was traveling west on the intersecting street. Both vehicles entered the intersection and the front of V2 impacted the left side of V1, resulting in the actuation of the driver's seat belt retractor pretensioner and the deployment of V1's frontal, left side impact and both side curtain air bags. V1 rotated clockwise as V2 rotated counterclockwise. The vehicles traveled in a southwesterly direction. V1 departed the southwest corner of the intersection and came to final rest facing southeast. V2 came to final rest near the same corner, facing southwest. Both vehicles were towed due to damage. The case occupant is the 31 year old female driver of V1 who was restrained with the available manual lap/shoulder belt. This case occupant's steering column mounted frontal, left seat back mounted and left and right roof side rail mounted curtain air bags all deployed during the crash. This case occupant sustained moderate injuries and was transported to a local hospital where she was hospitalized overnight. She was transferred to the trauma center the following day and was hospitalized for an additional two days. There were no other occupants in the vehicle. The driver and sole occupant of V2 was reported to have sustained visible injuries, but her treatment status is unknown.