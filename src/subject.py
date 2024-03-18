# Subject Class
import os
import util
import csv
import datetime
from tqdm import tqdm
import numpy as np
from pose import Pose
from load_mvnx import load_mvnx


class Subject:
    """
    Defines a subject and associated attributes
    """

    def __init__(self, subject_id):
        self.id = subject_id
        self.trial = None
        self.trial_date = None
        self.age = None  # Defined at first of Jan 2024
        self.gender = None
        self.weight = None
        self.height = None
        self.pole_length = None
        self.mvnx_filename = None
        self.mvnx = None
        self.segment_mdl = {}
        self.poses = []

        self.__load_subject()

    def __str__(self):
        return """Subject {self.id}:  
        \tGender = {self.gender} 
        \tAge = {self.age} 
        \tWeight[kg] = {self.weight} 
        \tHeight[m] = {self.height}
        \t"Model = {self.segment_mdl}

        \tTrial: {self.trial}
        \t\tDate = {self.trial_date}
        \tEquipment: 
        \t\t\tPole Length[m] = {self.pole_length}

        \tData:
        \t\tMVNX = {self.mvnx}
        """.format(
            self=self
        )

    def __calc_age(self, dob, date=datetime.date(2024, 1, 1)):
        if isinstance(date, str):
            date = datetime.datetime.strptime(date, "%Y%m%d")

        dob = datetime.datetime.strptime(dob, "%Y%m%d").date()
        delta = date - dob
        return delta.days

    def __load_subject(self):
        with open(util.SUBJECT_TABLE, "r") as sfile:
            reader = csv.reader(sfile)
            header = next(reader)
            header[0] = util.strip_utf8(header[0])
            header[-1] = util.strip_endline(header[-1])
            found = False
            for r, row in enumerate(reader):
                if len(row) > 0:
                    found = True
                    if row[header.index("sub_id")] == str(self.id):
                        self.age = self.__calc_age(row[header.index("sub_dob")])
                        self.gender = np.int16(row[header.index("sub_gender")])
                        self.weight = np.float64(row[header.index("sub_weight")])
                        self.height = np.float64(row[header.index("sub_height")])
                        self.pole_length = np.float64(
                            row[header.index("sub_pole_length")]
                        )

            if not found:
                raise ValueError("{} not found in subject table.".format(self.id))

    def __load_data(self, trial="flat", dataset="mvnx", **kwargs):
        """
                Loads a specific dataset
                    :trial: <String> "climb", "flat"
        ="""
        self.trial = trial
        # check arguments
        if self.trial not in ["flat", "climb"]:
            raise ValueError("{trial} type not valid".format(trial=self.trial))

        if dataset not in ["mvnx"]:
            raise ValueError("{dataset} not valid".format(dataset=dataset))

        print("STATUS: Loading mvnx data for subject {self.id}".format(self=self))
        files = os.listdir(util.DATA_DIR)
        for f in files:
            if f.startswith("100{self.id}_{self.trial}_".format(self=self)):
                self.mvnx_filename = f
                self.mvnx = load_mvnx(util.DATA_DIR + self.mvnx_filename)
                break
        else:
            raise ValueError("{dataset} type not valid.".format(dataset=dataset))

    def load_trial(self, trial_name):
        if trial_name not in ["flat", "climb"]:
            raise ValueError("{self.trial} type not valid".format(self=self))

        self.trial = trial_name

        self.__load_data(trial=self.trial, dataset="mvnx")

        print("STATUS: Estimating pose")
        for frame in tqdm(range(self.mvnx.frame_count)):
            self.poses.append(self.get_pose(frame))

    def get_total_mass(self):
        return self.weight + 2 * self.get_ski_mass() + 2 * self.get_pole_mass()

    def get_ski_mass(self):
        return util.SKI_MASS  # Weight of typical rollerski in kg

    def get_pole_mass(self):
        return self.pole_length * util.POLE_DENSITY

    def get_pose(self, frame):
        return Pose(self, frame)


if __name__ == "__main__":
    sub = Subject(1)
    print(sub)
    sub.load_trial("flat")
    print(sub)
