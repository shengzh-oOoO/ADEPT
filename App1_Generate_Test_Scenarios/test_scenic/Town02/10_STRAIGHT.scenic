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

adv_maneuvers = filter(lambda i: i.type == ManeuverType.RIGHT_TURN, ego_maneuver.conflictingManeuvers)
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
terminate when (distance to ego_spawn_pt) > 70# The crash occurred in the right lane of a three-lane roadway near a T-intersection controlled by a traffic control signal. Conditions were daylight and clear in the afternoon on a weekday. The posted speed limit was 80 kmph (50 mph). V1 was a 1990 Chevy C25 Cheyenne pickup truck traveling eastbound in the right lane approaching a T-intersection with a traffic control signal. V2, a 2005 Cadillac STS sedan, was traveling eastbound in the right lane directly in front of V1 and decelerating in traffic. V3, a 1996 Ford Explorer SUV, was traveling eastbound in the right lane directly 3-4 car lengths in front of V2. The driver of V1 was traveling between 21-30 MPH when he looked down for a second then looked back up and struck the back of V2. The driver of V1 braked and swerved right prior to the first impact. The front of V2 was pushed into the rear of V3. All three vehicles came to final rest facing eastbound in the right lane. A 39-year old male drove V1 and was not injured. He stated that he was traveling eastbound in the right lane at about 21-30 MPH when he looked down at his lap for a second. When he looked up, he saw that V2 was decelerating abruptly and he slammed on his brakes and swerved right in response but could not avoid contacting V2. He stated that there was stop and go traffic but thought he had enough room in between both vehicles. V1 was towed due to damage. The Critical Precrash Event for V1 was when V2 was decelerating while in the same lane in front of V1. The Critical Reason for the Critical Precrash Event was a recognition error - an internal distraction when the driver looked down while driving. The driver had about 100 lbs. of cargo in the bed of his pickup but there was no cargo shift or cargo spillage. The driver was familiar with the area and the surrounding traffic. The driver was keeping up with traffic but did not think he was traveling too closely. A 57-year old male drove V2 and was transported due to moderate injuries. He stated that he was decelerating in traffic when he was struck in the back by V1. The driver of V2 did not take any avoidance actions. V2 was towed due to damage. The Critical Precrash Event for V2 was the other vehicle in his lane, traveling in the same direction but at a higher speed. The Critical Reason for the Critical Precrash Event was not coded to this vehicle. The driver of V2 had three pre-existing medical conditions including sleep apnea, but he does not receive any treatment for his sleep apnea. The driver reported being drowsy prior to the crash. A 23-year old male drove V3 and was not injured. He stated that he was stopped in traffic when he saw V1 in his rearview mirror driving very fast towards the vehicle in back of him. The driver of V3 turned his wheel to the right but was struck in the back by V2. V3 was driven away from the scene. No Critical Precrash Event and no Critical Reason were coded to V3 because it was not involved in the first harmful event. The driver of V3 was talking with his passenger at the time of the crash but had watched the crash unfold in his rear view mirror. It was not thought that the driver of V3 contributed to the crash.