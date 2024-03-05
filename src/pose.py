import copy
import util
import mvn
import numpy as np
import quaternion as quat
import matplotlib.pyplot as plt


class Pose:
    """Defines a subjects pose at a given frame:
    :sub: <Subject> subject.
    :frame: <int> frame number.
    """

    def __init__(self, sub, frame):

        self.frame = frame
        self.points = {}
        self.vectors = {}
        self.coordinate_system = "Xsens"  # default coordiante system = XSense
        self.__get_points(sub)
        # self.__get_xsens_ori()
        # self.__to_ground()
        # self.__to_lcs()

    def __str__(self):
        # com = self.get_com()
        # vel = sel.get_com_vel()
        # acc = self.get_com_acc()
        # origin = self.__get_origin()
        # ori = self.__get_ori()
        #
        return """
        Pose at frame{self.frame}:
        \t Coordiante System: {self.coordiante_system}
        \t CoM: {com} m
        \t CoM Velocity: {vel} m/s
        \t CoM Acceleration: {acc} m/s^2
        \t Global Origin: {origin} m
        \t Direction: {ori}
        """.format(
            self=self,
            com=self.get_com(),
            vel=self.get_com_vel(),
            acc=self.get_com_acc(),
            origin=self.__get_point("Origin"),
            ori=self.__get_vector("Ori"),
        )

    def __get_points(self, sub):
        for seg in mvn.SEGMENTS.items():
            if seg[1] == "com":
                # Get CoM (from XSens) ignores pole/ski mass
                self.points["CoM"] = sub.mvnx.get_segment_pos(seg[0], self.frame)
            else:
                self.points[seg[1]] = sub.mvnx.get_segment_pos(seg[0], self.frame)

        self.points["Origin"] = sub.mvnx.get_segment_pos(
            mvn.SEGMENT_CENTER_OF_MASS, self.frame
        )

        self.vectors["XsensCoMVel"] = sub.mvnx.get_center_of_mass_vel(self.frame)
        self.vectors["XsensCoMAcc"] = sub.mvnx.get_center_of_mass_acc(self.frame)
        self.vectors["XsensOri"] = np.quaternion(
            *sub.mvnx.get_segment_ori(mvn.SEGMENT_PELVIS, self.frame)
        )

    def __get_point(self, point):
        return copy.deepcopy(self.points[point])

    def __get_vector(self, vector):
        return copy.deepcopy(self.vectors[vector])

    def get_com(self):
        return copy.deepcopy(self.points["CoM"])

    def __get_origin(self):
        return copy.deepcopy(self.points["Origin"])

    def add_point(self, name, data):
        self.points[name] = copy.deepcopy(data)

    def get_com_vel(self):
        com_vel = None
        if self.coordinate_system == "GCS":
            com_vel = copy.deepcopy(self.vectors["XsensCoMVel"]) + copy.deepcopy(
                self.vectors["GCSVel"]
            )
        elif self.coordinate_system == "LCS":
            com_vel = copy.deepcopy(self.vectors["XsenseCoMVel"])
        return com_vel

    def get_com_acc(self):
        com_acc = None
        if self.coordinate_system == "GCS":
            com_acc = copy.deepcopy(self.vectors["XsensCoMAcc"]) + copy.deepcopy(
                self.vectors["GCScc"]
            )
        elif self.coordiante_system == "LCS":
            com_acc = copy.deepcopy(self.vectors["XsenseCoMAcc"])
        return com_acc + [0, 0, -util.GRAVITY]

    def __get_ori(self):
        return copy.deepcopy(self.vectors["Ori"])

    def __to_lcs(self):
        # for i, k in enumerate(self.points.keys()):
        #     self.points[k] = self.points[k] - self.get_com()
        #
        for i, k in enumerate(self.points.keys()):
            self.points[k] = quat.rotate_vectors(
                self.vectors["LCSOri"], self.__get_point(k)
            )
        self.coordinate_system = "LCS"

    def __to_gcs(self):
        if self.coordinate_system == "GCS":
            print("WARNING: pose already in GCS - doing nothing")

        elif self.coordinate_system != "LCS":
            self.__to_lcs()
            ori = self.vectors["Ori"]
            for i, k in enumerate(self.points.keys()):
                if k != "GCSOrigin":
                    self.points[k] = (
                        quat.rotate_vectors(self.vectors["Ori"], self.__get_point(k))
                        + self.points["GCSOrigin"]
                    )
            for i, k in enumerate(self.vectors.keys()):
                if k in ["XsensCoMVel", "XsensCoMAcc"]:
                    self.vectors[k] = quat.rotate_vectors(ori, self.__get_vector(k))
        else:
            ori = self.vectors["Ori"]
            for i, k in enumerate(self.points.keys()):
                if k != "GCSOrigin":
                    self.points[k] = (
                        quat.rotate_vectors(ori, self.__get_point(k))
                        + self.points["GCSOrigin"]
                    )

            for i, k in enumerate(self.vectors.keys()):
                if k in ["XsensCoMVel", "XsensCoMAcc"]:
                    self.vectors[k] = quat.rotate_vectors(ori, self.__get_vector(k))

        self.coordinate_system = "GCS"

    def __to_ground(self):
        min_height = 1e12
        # pts = copy.deepcopy(self.points)
        for i, k in enumerate(self.points.keys()):
            if k not in ["Origin", "CoM"]:
                if self.points[k][2] < min_height:
                    min_height = self.points[k][2]

        if min_height != 0:
            for i, k in enumerate(self.points.keys()):
                if k == "CoM":
                    com = self.get_com()
                    com[2] = com[2] - min_height
                    self.points["CoM"] = copy.deepcopy(com)
                elif k != "CoM":
                    self.points[k][2] = self.points[k][2] - min_height

    def __parse_coordinate_systems(self, cs):
        if cs not in ["GCS", "LCS", "Xsens"]:
            raise ValueError(
                '{} is not a valid coordinate system.  \nValid coordinate systems are ["GCS", "LCS", "Xsens"]'
            )

    def __get_xsens_ori(self):
        pelvis = self.points["Pelvis"]
        r_hip = self.points["RightUpperLeg"]

        # rotate and project given orientation of pelvis.
        v1 = r_hip - pelvis
        v3 = np.array([0, 0, 1])
        v2 = np.cross(v3, v1)
        v1 = np.cross(v2, v3)
        v1 = v1 / np.linalg.norm(v1)
        v2 = v2 / np.linalg.norm(v2)
        v3 = v3 / np.linalg.norm(v3)

        self.vectors["LCSOri"] = util.rel_quat(
            v1, [1, 0, 0]
        )  # make v1 the new x axis...

    def __get_gcs_kin(self, sub):
        ori = sub.gnss.vel[self.frame, :]

        self.vectors["Ori"] = util.rel_quat(ori, [1, 0, 0])
        self.points["GCSOrigin"] = sub.gnss.pos[self.frame, :]

        self.vectors["GCSVel"] = sub.gnss.vel[self.frame, :]
        self.vectors["GCSAcc"] = sub.gnss.acc[self.frame, :]

    def plot_pose(self, title=""):
        """
        Plot a subjects pose
        """

        if self.coordinate_system == "GCS":
            xlabel = "X"
            ylabel = "Y"
        else:
            xlabel = "X [m]"
            ylabel = "Y [m]"

        zlabel = "Up [m]"

        pts = self.points
        fig = plt.figure(0)
        ax = plt.axes(projection="3d")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_zlabel(zlabel)

        for i, k in enumerate(pts.keys()):
            pt = pts[k]
            if k == "RightShoulder":
                ax.scatter(pt[1], pt[0], pt[2], color=util.CMAP(0))
            elif k == "LeftShoulder":
                ax.scatter(pt[1], pt[0], pt[2], color=util.CMAP(0))
            elif k[:4] == "Righ":
                ax.scatter(pt[1], pt[0], pt[2], color=util.CMAP(2))
            elif k[:4] == "Left":
                ax.scatter(pt[1], pt[0], pt[2], color=util.CMAP(3))
            # elif k[:4] == "CoM":
            #     ax.scatter(pt[1], pt[0], pt[2], color="black")
            # elif k[:4] == "Orig":
            # break
            else:
                ax.scatter(pt[1], pt[0], pt[2], color=util.CMAP(0))

            print("Point{p}: {pp}".format(p=k, pp=pt))
        #
        ax.add_artist(
            util.Arrow3D(
                [pts["Pelvis"][1], pts["L5"][1]],
                [pts["Pelvis"][0], pts["L5"][0]],
                [pts["Pelvis"][2], pts["L5"][2]],
                **dict(mutation_scale=3, arrowstyle="-", color=util.CMAP(0)),
            )
        )
        ax.add_artist(
            util.Arrow3D(
                [pts["L5"][1], pts["L3"][1]],
                [pts["L5"][0], pts["L3"][0]],
                [pts["L5"][2], pts["L3"][2]],
                **dict(mutation_scale=3, arrowstyle="-", color=util.CMAP(0)),
            )
        )
        ax.add_artist(
            util.Arrow3D(
                [pts["L3"][1], pts["T12"][1]],
                [pts["L3"][0], pts["T12"][0]],
                [pts["L3"][2], pts["T12"][2]],
                **dict(mutation_scale=3, arrowstyle="-", color=util.CMAP(0)),
            )
        )
        ax.add_artist(
            util.Arrow3D(
                [pts["T12"][1], pts["Neck"][1]],
                [pts["T12"][0], pts["Neck"][0]],
                [pts["T12"][2], pts["Neck"][2]],
                **dict(mutation_scale=3, arrowstyle="-", color=util.CMAP(0)),
            )
        )
        ax.add_artist(
            util.Arrow3D(
                [pts["Neck"][1], pts["Head"][1]],
                [pts["Neck"][0], pts["Head"][0]],
                [pts["Neck"][2], pts["Head"][2]],
                **dict(mutation_scale=3, arrowstyle="-", color=util.CMAP(0)),
            )
        )
        ax.add_artist(
            util.Arrow3D(
                [pts["T8"][1], pts["RightShoulder"][1]],
                [pts["T8"][0], pts["RightShoulder"][0]],
                [pts["T8"][2], pts["RightShoulder"][2]],
                **dict(mutation_scale=3, arrowstyle="-", color=util.CMAP(0)),
            )
        )
        ax.add_artist(
            util.Arrow3D(
                [pts["T8"][1], pts["LeftShoulder"][1]],
                [pts["T8"][0], pts["LeftShoulder"][0]],
                [pts["T8"][2], pts["LeftShoulder"][2]],
                **dict(mutation_scale=3, arrowstyle="-", color=util.CMAP(0)),
            )
        )
        ax.add_artist(
            util.Arrow3D(
                [pts["Neck"][1], pts["RightShoulder"][1]],
                [pts["Neck"][0], pts["RightShoulder"][0]],
                [pts["Neck"][2], pts["RightShoulder"][2]],
                **dict(mutation_scale=3, arrowstyle="-", color=util.CMAP(0)),
            )
        )
        ax.add_artist(
            util.Arrow3D(
                [pts["Neck"][1], pts["LeftShoulder"][1]],
                [pts["Neck"][0], pts["LeftShoulder"][0]],
                [pts["Neck"][2], pts["LeftShoulder"][2]],
                **dict(mutation_scale=3, arrowstyle="-", color=util.CMAP(0)),
            )
        )

        ax.add_artist(
            util.Arrow3D(
                [pts["LeftShoulder"][1], pts["LeftUpperArm"][1]],
                [pts["LeftShoulder"][0], pts["LeftUpperArm"][0]],
                [pts["LeftShoulder"][2], pts["LeftUpperArm"][2]],
                **dict(mutation_scale=3, arrowstyle="-", color=util.CMAP(3)),
            )
        )
        ax.add_artist(
            util.Arrow3D(
                [pts["LeftUpperArm"][1], pts["LeftForeArm"][1]],
                [pts["LeftUpperArm"][0], pts["LeftForeArm"][0]],
                [pts["LeftUpperArm"][2], pts["LeftForeArm"][2]],
                **dict(mutation_scale=3, arrowstyle="-", color=util.CMAP(3)),
            )
        )
        ax.add_artist(
            util.Arrow3D(
                [pts["LeftForeArm"][1], pts["LeftHand"][1]],
                [pts["LeftForeArm"][0], pts["LeftHand"][0]],
                [pts["LeftForeArm"][2], pts["LeftHand"][2]],
                **dict(mutation_scale=3, arrowstyle="-", color=util.CMAP(3)),
            )
        )

        ax.add_artist(
            util.Arrow3D(
                [pts["RightShoulder"][1], pts["RightUpperArm"][1]],
                [pts["RightShoulder"][0], pts["RightUpperArm"][0]],
                [pts["RightShoulder"][2], pts["RightUpperArm"][2]],
                **dict(mutation_scale=3, arrowstyle="-", color=util.CMAP(2)),
            )
        )
        ax.add_artist(
            util.Arrow3D(
                [pts["RightUpperArm"][1], pts["RightForeArm"][1]],
                [pts["RightUpperArm"][0], pts["RightForeArm"][0]],
                [pts["RightUpperArm"][2], pts["RightForeArm"][2]],
                **dict(mutation_scale=3, arrowstyle="-", color=util.CMAP(2)),
            )
        )
        ax.add_artist(
            util.Arrow3D(
                [pts["RightForeArm"][1], pts["RightHand"][1]],
                [pts["RightForeArm"][0], pts["RightHand"][0]],
                [pts["RightForeArm"][2], pts["RightHand"][2]],
                **dict(mutation_scale=3, arrowstyle="-", color=util.CMAP(2)),
            )
        )

        ax.add_artist(
            util.Arrow3D(
                [pts["Pelvis"][1], pts["LeftUpperLeg"][1]],
                [pts["Pelvis"][0], pts["LeftUpperLeg"][0]],
                [pts["Pelvis"][2], pts["LeftUpperLeg"][2]],
                **dict(mutation_scale=3, arrowstyle="-", color=util.CMAP(3)),
            )
        )
        ax.add_artist(
            util.Arrow3D(
                [pts["LeftUpperLeg"][1], pts["LeftLowerLeg"][1]],
                [pts["LeftUpperLeg"][0], pts["LeftLowerLeg"][0]],
                [pts["LeftUpperLeg"][2], pts["LeftLowerLeg"][2]],
                **dict(mutation_scale=3, arrowstyle="-", color=util.CMAP(3)),
            )
        )
        ax.add_artist(
            util.Arrow3D(
                [pts["LeftLowerLeg"][1], pts["LeftFoot"][1]],
                [pts["LeftLowerLeg"][0], pts["LeftFoot"][0]],
                [pts["LeftLowerLeg"][2], pts["LeftFoot"][2]],
                **dict(mutation_scale=3, arrowstyle="-", color=util.CMAP(3)),
            )
        )
        ax.add_artist(
            util.Arrow3D(
                [pts["LeftFoot"][1], pts["LeftToe"][1]],
                [pts["LeftFoot"][0], pts["LeftToe"][0]],
                [pts["LeftFoot"][2], pts["LeftToe"][2]],
                **dict(mutation_scale=3, arrowstyle="-", color=util.CMAP(3)),
            )
        )

        ax.add_artist(
            util.Arrow3D(
                [pts["Pelvis"][1], pts["RightUpperLeg"][1]],
                [pts["Pelvis"][0], pts["RightUpperLeg"][0]],
                [pts["Pelvis"][2], pts["RightUpperLeg"][2]],
                **dict(mutation_scale=3, arrowstyle="-", color=util.CMAP(2)),
            )
        )
        ax.add_artist(
            util.Arrow3D(
                [pts["RightUpperLeg"][1], pts["RightLowerLeg"][1]],
                [pts["RightUpperLeg"][0], pts["RightLowerLeg"][0]],
                [pts["RightUpperLeg"][2], pts["RightLowerLeg"][2]],
                **dict(mutation_scale=3, arrowstyle="-", color=util.CMAP(2)),
            )
        )
        ax.add_artist(
            util.Arrow3D(
                [pts["RightLowerLeg"][1], pts["RightFoot"][1]],
                [pts["RightLowerLeg"][0], pts["RightFoot"][0]],
                [pts["RightLowerLeg"][2], pts["RightFoot"][2]],
                **dict(mutation_scale=3, arrowstyle="-", color=util.CMAP(2)),
            )
        )
        ax.add_artist(
            util.Arrow3D(
                [pts["RightFoot"][1], pts["RightToe"][1]],
                [pts["RightFoot"][0], pts["RightToe"][0]],
                [pts["RightFoot"][2], pts["RightToe"][2]],
                **dict(mutation_scale=3, arrowstyle="-", color=util.CMAP(2)),
            )
        )

        limits = np.array([getattr(ax, f"get_{axis}lim")() for axis in "xyz"])
        ax.set_box_aspect(np.ptp(limits, axis=1))
        ax.set_title(title)
        plt.show(block=True)  # show plot
        return fig, ax


if __name__ == "__main__":
    from subject import Subject

    sub = Subject(1)
    sub.load_trial("flat")
    print(sub)

    frame = 5000  # Pick a random frame
    p = Pose(sub, frame)
    p.plot_pose()
