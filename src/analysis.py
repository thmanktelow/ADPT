# ------------------------------------------------------------------------------
# (A) Import Libraries
# ------------------------------------------------------------------------------
import os

# Set working directory
if os.getcwd().endswith("ADPT"):
    os.chdir(os.getcwd() + "/src")
import numpy as np
import scipy as sp
import copy
import matplotlib.pyplot as plt
import util
from subject import Subject
from pose import Pose
from filter import filt_acf, filt_butter, finite_diff


# ------------------------------------------------------------------------------
# (B) Load Subject
# ------------------------------------------------------------------------------
# Load Subject
# set sidx to be the id of the subject of interest from the subject table
sidx = 3
sub = Subject(sidx)

# Load a trial
# Set trial" to be either flat or uphill (the trial of interest).
# Note that this needs to match the namining convention for the MVNX files
trial = "flat"
sub.load_trial(trial)
print(sub)

# ------------------------------------------------------------------------------
# (C) Process Poses
# ------------------------------------------------------------------------------
# Process the pose for each frame in the trial
poses = []
for frame in range(sub.mvnx.frame_count):
    poses.append(sub.get_pose(frame))
    # Add mid hand and mid feet point to pose
    l_toe = poses[-1].points["LeftToe"]
    r_toe = poses[-1].points["RightToe"]
    mid_point_feet = (l_toe + r_toe) / 2
    poses[-1].add_point("MidFeet", mid_point_feet)
    l_hand = poses[-1].points["LeftHand"]
    r_hand = poses[-1].points["RightHand"]
    mid_point_hands = (l_hand + r_hand) / 2
    poses[-1].add_point("MidHands", mid_point_hands)

# Get some useful constants and define a time vector
N = len(poses)  # number of frames
dt = 1 / sub.mvnx.frame_rate  # time step
t = np.arange(0, (N * dt), dt)  # time vector
t = t[:N]  # remove last element to match the length of the pose array

# ------------------------------------------------------------------------------
# (D) Trim to relevant frames (Optional)
# ------------------------------------------------------------------------------
# Optional: Trim The time series to a specific range of frames...
# I suggest you do this by examining the time series of left/right hands in the
# Vectical direction to identify the start/end of polling motion

# Set the start and end frame for the polling motion
frame_start = 0  # Use 2nd bout...
frame_end = len(poses)

lhand = np.array([poses[i].points["LeftHand"][2] for i in range(N)])
rhand = np.array([poses[i].points["RightHand"][2] for i in range(N)])
plt.plot(np.array(lhand))
plt.plot(np.array(rhand))
plt.vlines(
    x=[frame_start, frame_end],
    ymin=np.min(np.hstack((lhand, rhand))) - 0.1,
    ymax=np.max(np.hstack((lhand, rhand))) + 0.1,
    color="r",
)
plt.vlines(
    x=frame_end,
    ymin=np.min(np.hstack((lhand, rhand))) - 0.1,
    ymax=np.max(np.hstack((lhand, rhand))) + 0.1,
    color="r",
)
plt.legend(["Left Hand", "Right Hand"])
plt.xlabel("Frame")
plt.title("Trial Segmentation")
plt.show()


poses = poses[frame_start:frame_end]
t = t[frame_start:frame_end] - t[frame_start]
N = len(poses)

# ------------------------------------------------------------------------------
# (E) Segment polling cycles
# ------------------------------------------------------------------------------
# Q How are we defining pole plant and recovery
# - Force sensor in pole (need sync with MVNX data)
# - Vertical displacement of hands
# - Joint angles
# - other?

# Option 1 use peak vertical displacement of hands to segment polling cycles
# Other options include peak acceleration of hands or force sensor data...
# Compute norm of acceleration at hand midpoint...
min_distance = 0.7  # Need to set this parameter based on length of polling cycle,
# needs to be less than the length of a polling cycle but greater than the time
# between pole plant and pole lift...
# This is a very simple approach which will likely need to be refined
mid_hand_z = np.array([poses[i].points["MidHands"][2] for i in range(N)])
fc = 25  # Cutoff frequency for filtering
mid_hand_z = filt_butter(mid_hand_z, fc, 1 / dt)
pk = sp.signal.find_peaks(mid_hand_z, distance=min_distance / dt)
fig, ax1 = plt.subplots()
ax1.plot(t, mid_hand_z)
ax1.plot(t[pk[0]], mid_hand_z[pk[0]], "x")
ax1.set_ylabel("Vert. Hand Pos (m)")
ax1.set_xlabel("Time (s)")
ax1.set_title("Polling Cycle Segmentation")
plt.show()


# trim to full polling cycles
# ignore first and last cycle as they may be incomplete
pk = pk[0][1:-1]
# Trim to complete polling cycles
t = t[pk[0] : pk[-1]]
poses = poses[pk[0] : pk[-1]]
N = len(poses)  # number of frames
pk = pk - pk[0]  # reset peak indices to start from 0


class PollingCycle:
    # Define a class for a polling cycle
    # Polling cycle is a class which contains signals for a single polling cycle
    def __init__(self, start, end, t, poses_in):
        self.start = start
        self.end = end
        self.t = t[start:end] - t[start]
        self.poses = poses_in[start:end]

    def get_timeseries(self, key, axis=None):
        if key not in self.poses[0].points:
            raise ValueError(f"Key {key} not in pose points")
        if axis is None:
            return np.array([pose.points[key] for pose in self.poses])
        else:
            if axis not in [0, 1, 2]:
                raise ValueError("Axis must be 0, 1 or 2")
            return np.array([pose.points[key][axis] for pose in self.poses])

    def get_joint_timeseries(self, key, axis=None):
        if key not in self.poses[0].joint_angles:
            raise ValueError(f"Key {key} not in pose joints")
        if axis is None:
            return np.array([pose.joint_angles[key] for pose in self.poses])
        else:
            if axis not in [0, 1, 2]:
                raise ValueError("Axis must be 0, 1 or 2")
            return np.array([pose.joint_angles[key][axis] for pose in self.poses])

    def filter_point(self, point, axis, fc=None):
        if axis not in [0, 1, 2]:
            raise ValueError("Axis must be 0, 1 or 2")
        y = self.get_timeseries(point, axis)
        if fc is None:
            # Determine cutoff frequency via minimization of residual autocorrelation
            # See Challis 1999 for details.
            _, fc = filt_acf(y, 1 / dt)

        y_filt = filt_butter(y, fc, 1 / dt)
        for i, pose in enumerate(self.poses):
            pose.points[point][axis] = y_filt[i]

        return y_filt

    # TODO Get pole lift


# Segment into polling cycles
cycles = []  # a list of polling cycles
for i in range(len(pk) - 1):
    cycles.append(PollingCycle(pk[i], pk[i + 1], t, poses))

nCycles = len(cycles)  # number of polling cycles examined.
print(f"Number of polling cycles to analyze: {nCycles}")


# ------------------------------------------------------------------------------
# (E) Visualize polling cycles
# ------------------------------------------------------------------------------
# Plot of relationship between hand and pelvis vertical position
fig, ax = plt.subplots()
ax2 = ax.twinx()
for i, cycle in enumerate(cycles):
    rhand = cycle.get_timeseries("RightHand", 2)  # - cycle.get_timeseries("MidFeet", 2)
    lhand = cycle.get_timeseries("LeftHand", 2)  # - cycle.get_timeseries("MidFeet", 2)
    pelvis = cycle.get_timeseries("Pelvis", 2)  # - cycle.get_timeseries("MidFeet", 2)
    com = cycle.get_timeseries("CoM", 2)
    ax.plot(cycle.t, rhand, color=util.CMAP(0), alpha=0.2)
    ax.plot(cycle.t, lhand, color=util.CMAP(1), alpha=0.2)
    ax2.plot(cycle.t, pelvis, color=util.CMAP(2), alpha=0.2)
    ax2.plot(cycle.t, com, color="black", alpha=0.2)
ax.set_xlabel("Time (s)")
ax.set_ylabel("Hand Z (m)")
ax2.set_ylabel("Pelvi/CoM Z (m)")
ax.set_title("Vertical Hand and Pelvis Position")
ax.legend(
    ["Right Hand", "Left Hand", "Pelvis", "CoM"],
    loc="lower right",
)
leg = ax.get_legend()
leg.legendHandles[0].set_color(util.CMAP(0)),
leg.legendHandles[1].set_color(util.CMAP(1)),
leg.legendHandles[2].set_color(util.CMAP(2)),
leg.legendHandles[3].set_color("black"),
plt.show()

# ------------------------------------------------------------------------------
# (F) Plot joint angles
# ------------------------------------------------------------------------------
fig, ax = plt.subplots(1, 3)
for i, cycle in enumerate(cycles):
    rhip = cycle.get_joint_timeseries("jRightHip", 0)
    rknee = cycle.get_joint_timeseries("jRightKnee", 0)
    rankle = cycle.get_joint_timeseries("jRightAnkle", 0)
    ax[0].plot(cycle.t, rhip, color=util.CMAP(0), alpha=0.2)
    ax[0].plot(cycle.t, rknee, color=util.CMAP(1), alpha=0.2)
    ax[0].plot(cycle.t, rankle, color=util.CMAP(2), alpha=0.2)

    rhip = cycle.get_joint_timeseries("jRightHip", 1)
    rknee = cycle.get_joint_timeseries("jRightKnee", 1)
    rankle = cycle.get_joint_timeseries("jRightAnkle", 1)
    ax[1].plot(cycle.t, rhip, color=util.CMAP(0), alpha=0.2)
    ax[1].plot(cycle.t, rknee, color=util.CMAP(1), alpha=0.2)
    ax[1].plot(cycle.t, rankle, color=util.CMAP(2), alpha=0.2)

    rhip = cycle.get_joint_timeseries("jRightHip", 2)
    rknee = cycle.get_joint_timeseries("jRightKnee", 2)
    rankle = cycle.get_joint_timeseries("jRightAnkle", 2)
    ax[2].plot(cycle.t, rhip, color=util.CMAP(0), alpha=0.2)
    ax[2].plot(cycle.t, rknee, color=util.CMAP(1), alpha=0.2)
    ax[2].plot(cycle.t, rankle, color=util.CMAP(2), alpha=0.2)
fig.suptitle("Right Leg Joint Angles")
ax[0].set_xlabel("Time (s)")
ax[0].set_ylabel("Transverse Joint Angle (rad)")
ax[1].set_xlabel("Time (s)")
ax[1].set_ylabel("Frontal Joint Angle (rad)")
ax[2].set_xlabel("Time (s)")
ax[2].set_ylabel("Sagittal Joint Angle (rad)")

# ax.set_xlabel("Time (s)")
# ax.set_ylabel("Joint Angle (rad)")
# ax.set_title("Right Leg Joint Angles")
ax[2].legend(
    ["Hip", "Knee", "Ankle"],
    loc="lower right",
)
plt.show()


# Things to consider:
# 1) How are you going to define a polling cycle and identify it reliably in the
# time series?
#   a) Is there any existing literature on this which could be used?
#   b) what phases of the polling cycle are you interested in?
#       Force application + recover or more complex than this?
# Currently I have used a simple peak finding algorithm to segment polling cycles
# based on vertical displacement of hands but this is likely biases (occurs before
# pole plant) and this bias may influence your outcomes.  You may want to consider
# using force sensored poles and working on how to synchronise these with the Xsens data.

# 2) The time delta you are interested in would that be the time between minimum
# Pelvis height and the next pole plant?
#   You should think through and do some reading on how you could find these in
# the time series. - My example for identifying pole plant (via scipy find peaks function)
# is probably a good starting point but by no means robust....

# 3) Keep in mind reference frames are always important.  Right now Z is consistently
# Orientate upwards and if you assume the mid point between the feet is ground level
# you can compute vertical displacement of the hands and pelvis relative to the ground
# as I have demonstrated in the code above.  However the forward direction
# is not consistent with either the X or Y axis in the MVNX file.  We can reorientate
# reference frames using an optimization procedure to algin the forward direction
# with the X axis if you need to look at AP displacement of hands/pelvis etc.

# 3) You mention you are interested in examining how the inertial of the upper
# body is used to generate force.  Keep in the back of your mind that if necessary
# we can estimate the COM and rotational moment of inertial of say the upper body
# (arms and trunk) and lower body separately which may be of use but it would take
# more work to do this...
#   - Rotation to consistent forward direction
#   - Estimation of segmental CoM and moment of inertia from DeLava's equations
#    - Comparison of CoM motion and pelvis/leg motion may be useful.

# 5) Joint angles vs global positions.  Right now this analysis focuses on global positions
# of the relevant landmarks (hands, pelvis, feet).  However, joint angles may be
# a better way to look at this problem.  I.e. Timing of Flexion/extension of ankle knee, hip
# in relation to pole plant.
