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

adv_maneuvers = filter(lambda i: i.type == ManeuverType.STRAIGHT, ego_maneuver.conflictingManeuvers)
adv_maneuver = Uniform(*adv_maneuvers)
adv_trajectory = [adv_maneuver.startLane, adv_maneuver.connectingLane, adv_maneuver.endLane]

ego_spawn_pt = OrientedPoint in ego_maneuver.startLane.centerline
adv_spawn_pt = OrientedPoint in adv_maneuver.startLane.centerline

ego = Car at ego_spawn_pt,
    with blueprint "vehicle.lincoln2020.mkz2020",
    with rolename "hero"

adversary = Car at adv_spawn_pt,
    with blueprint "vehicle.tesla.model3",
    with behavior AdversaryBehavior(adv_trajectory)

require 15 <= (distance to intersec) <= 20
require 10 <= (distance from adversary to intersec) <= 15
terminate when (distance to ego_spawn_pt) > 70# The crash occurred on a level two lane undivided roadway with a posted speed limit of 30 MPH (48 KPH). It was raining, the roadway was wet and it was dark, but lighted at the time of this weekday evening crash. Vehicle #1, a 1990 Buick Century, was traveling east on a one-way street with a posted speed limit of 25 MPH (40 KPH) going the wrong way. Vehicle #1 had failed to comply with police and attempted to elude them. Vehicle #1 approached a “T” intersection with no controls for this direction. Vehicle #1 entered the intersection, crossed both north and southbound lanes and struck a parked vehicle in the left side with its front. Vehicle #1 came to rest facing east in the northbound lane. The parked vehicle, a 1994 Chrysler LHS, which was legally parked in a designated parking lane, was rotated counterclockwise and struck the curb with its right side. The parked vehicle continued to rotate, struck a light pole with its right side and came to rest facing in a northwesterly direction. Both vehicles were towed due to damage. The Buick Century (Vehicle #1) was driven by an unbelted 47 year-old male, who was transported, treated and released at a local hospital for minor injuries. He stated that he was traveling down a one way street and hit a parked car. He stated that he was attempting to turn right at the intersection and braked in an attempt to avoid the crash. He stated he did not wear corrective lenses. No additional information was obtained due to driver being incarcerated. The Chrysler LHS was occupied by an unbelted 39-year-old male, who was transported, treated and released at a local hospital. This person could not be located and no further information was obtained. The Critical Precrash Event for Vehicle #1 was this vehicle traveling, end departure. The Critical Reason for the Critical Precrash Event was too fast for conditions. This was chosen because the driver was eluding police, going down a one-way street the wrong way at a high rate of speed and it was raining. Police considered alcohol to be a factor and issued a citation.