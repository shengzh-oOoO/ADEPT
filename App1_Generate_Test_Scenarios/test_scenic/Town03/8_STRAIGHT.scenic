param map = localPath('../maps/CARLA/Town03.xodr')
param carla_map = 'Town03'
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
    with blueprint "vehicle.chevrolet.impala",
    with rolename "hero"

adversary = Car at adv_spawn_pt,
    with blueprint "vehicle.yamaha.yzf",
    with behavior AdversaryBehavior(adv_trajectory)

require 15 <= (distance to intersec) <= 20
require 10 <= (distance from adversary to intersec) <= 15
terminate when (distance to ego_spawn_pt) > 70# This crash occurred on a weekday afternoon in an intersection between two, two lane undivided, bituminous roadways. The East/West roadway was dry and straight with a posted speed limit of 89 kmph (55 mph) and a -2 % grade. This roadway had a black on yellow intersection warning with flashing yellow beacon with a suggested speed of 72 kmph (45 mph) prior to the intersection. The North/South roadway was a dry, level roadway with a posted speed limit of 72 kmph (45 mph) and was controlled by a stop sign. There was also a flashing signal, yellow for the main roadway and red for the cross street. V1, a 1993 Chevrolet Prizm, was stopped in the northbound lane for the stop sign and flashing red beacon. V2, a 2001 Jeep Wrangler, was westbound approaching the same intersection. V1 entered the intersection and was struck in the right side by the front of V2. V1 rotated counterclockwise and the right side struck the left side of V2. V2 continued forward a short distance, struck a light pole with its front and came to rest facing west. V1 continued off the right side of the roadway and swiped an 80 cm diameter tree with its right side. V1 than struck a 120 cm diameter tree with its front and came to final rest facing west. Both vehicles were towed due to damage. V1 was driven by a belted 20year-old female who was transported, treated and released for minor injuries. The driver could not be located or contacted for additional information. She did state to the police that she did not see V2 coming when she proceeded into the intersection. The Critical Precrash Event for Vehicle 1 was coded as this vehicle traveling, crossing over (passing through) intersection. The Critical Reason for the Critical Event was coded as a recognition error, inadequate surveillance, looked but did not see. This was chosen because the driver stated that she did not see V2 coming when she began to enter the intersection. V2 was driven by a belted 19-year-old male who was in good health and on his way to the dentist. The driver stated that he was traveling between 66-80kmph (41-50mph) prior to the crash. He also stated that his sight line to the other vehicle was clear, that she went through the stop sign and he attempted to avoid the collision by braking and steering right but it was too late. The driver was uninjured but his passenger was transported to a treatment facility with non-incapacitating injuries. The Critical Precrash Event for V2 was coded as other vehicle encroachment from crossing street, across path. The Critical Reason for the Critical Event was not coded to this vehicle. There were no known associated factors coded to this driver and he was not thought to have contributed to the crash.