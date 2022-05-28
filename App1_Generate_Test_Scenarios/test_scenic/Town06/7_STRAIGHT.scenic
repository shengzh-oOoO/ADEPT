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
    with blueprint "vehicle.mini.cooperst",
    with rolename "hero"

adversary = Car at adv_spawn_pt,
    with blueprint "vehicle.mini.cooperst",
    with behavior AdversaryBehavior(adv_trajectory)

require 15 <= (distance to intersec) <= 20
require 10 <= (distance from adversary to intersec) <= 15
terminate when (distance to ego_spawn_pt) > 70# This crash occurred on an intersection between two lane undivided roadways. The Precrash roadway of Vehicle #1 had a elevation of +5.7% and the Precrash roadway of Vehicle #2 had an elevation of + 18.8% and posted speed limits of 48KMPH (30MPH). The weather was clear and the roadway was dry. This crash occurred on a weekday evening. Vehicle #1 a 1998 Mercury Sable was traveling in a westerly direction on the two lane undivided roadway crossing an intersection and intending to go straight. Vehicle #2 a 1995 Nissan Altima was traveling northwest intending to cross the intersection and go in a northern direction. Vehicle #2 impacted the front left of Vehicle #1 with its front left. Vehicle #1 came to rest facing southwest partially off the roadway with its front. Vehicle #2 came to rest on the roadway facing north after having gone through the intersection. The Mercury Sable was driven by belted 28 year old female who was transported to an area hospital, treated and released. She stated that the event happened so quickly she did not have time to react. She did, however, cover her face with her hands prior to impact. The Critical Precrash Event coded to Vehicle #1 was Other Vehicle Encroachment From Crossing street, Across Path. The Critical Reason for the Critical Precrash Event was not coded to this vehicle. The driver of Vehicle #1 was not thought to have contributed to the crash. There is no evidence of any illegal drug use, however the driver stated she takes several types of antihistamines for allergies. The 1995 Nissan Altima (Vehicle #2) was driven by a belted 30 year old male who stated he was not injured. He stated that Vehicle #1 did not stop at the intersection and that he has stopped for 2 seconds before continuing through the intersection. This driver was ticketed by police for driving with a suspended registration. The Critical Precrash Event coded to this Vehicle was This Vehicle Traveling Crossing over (passing through) an intersection. The Critical Reason for the Critical Precrash Event was coded as a Driver Related Factor, Inadequate surveillance. This Critical Reason was coded due to the driver stating he had driven on this roadway many times and knew there were no stop signs. It should be noted at this time that there were no stops signs or traffic control signals on either road of this intersection. It should further be noted that a utility pole situated extremely close to the roadway, on Driver #2's right side, with a large bright green sign may have partially blocked this drivers sight line for approaching traffic. It should also be noted that several parked vehicles to the right of this driver might also have limited his sight line. The police did not consider illegal drugs to be involved and ordered no tests for either driver.