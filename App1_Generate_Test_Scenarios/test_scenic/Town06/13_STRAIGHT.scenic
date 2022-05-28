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
    with blueprint "vehicle.tesla.model3",
    with rolename "hero"

adversary = Car at adv_spawn_pt,
    with blueprint "vehicle.mini.cooperst",
    with behavior AdversaryBehavior(adv_trajectory)

require 15 <= (distance to intersec) <= 20
require 10 <= (distance from adversary to intersec) <= 15
terminate when (distance to ego_spawn_pt) > 70# The two-vehicle crash occurred in the middle of a T-intersection. The roadways for both vehicles were asphalt, dry, and level. The posted speed limit was 56 kmph (35 mph) for both directions. Vehicle one (V1) had a stop sign controlling its direction; there was no control device for Vehicle two (V2). The crash occurred in the early afternoon on a weekday. The traffic flow was light and the weather was clear. Vehicle one, a 2005 Ford Expedition, was traveling westbound and stopped at the corner, adhering to the passive stop sign located on both sides of the corner. She said she looked both ways before making her intended right turn; however, her view was obstructed by several vehicles parked on the southeast corner. She accelerated into the intersection and the front of her vehicle contacted the right front of V2, a 1992 Pontiac Bonneville. The driver of V1, a 21-year old female, was aware of the posted stop sign, and said that she felt comfortable with the rented vehicle. She gave no other explanation for the crash besides her inability to see the traffic to her left. After the initial impact, V1 continued traveling north for a few feet and spun around 45 degrees clockwise before striking a narrow pole on the northeast corner of the intersection, where it came to final rest. The driver of V2, a 26-year old female, stated that she had the right of way and didn't see V1 until the last second and she then applied her brakes. She was issued a citation for driving without proof of insurance. Although V2 was towed, police logged the crash as a non-reportable accident. The Critical Pre-Crash Event for V1 was when it began to turn right at the intersection. The Critical Reason was coded as inadequate surveillance (looked but did not see). Associated factors coded to this driver include a sightline restriction caused by the parked vehicles on the side of the roadway and unfamiliarity with both her vehicle and the roadway. The Critical Pre-Crash Event for V1 was coded as other vehicle encroachment from crossing street, turning into same direction. The Critical Reason was not coded to this vehicle. No associated factors were coded to this driver.