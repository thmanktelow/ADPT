# Subject Class
import util
import csv
import datetime
import numpy as np


class subject:
    """
    Defines a subject and associated attributes
    """
    "Andys change"
    def __init__(self, subject_id):
        self.id = subject_id
        self.trial = None
        self.trial_date = None
        self.age = None # Defined at first of Jan 2024
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

        dob = datetime.datetime.strptime(dob, '%Y%m%d').date()
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
                        self.pole_length = np.float64(row[header.index("sub_pole_length")])


                        # self.mvnx_filename = str(row[header.index("mvnx_file")])
                        # for s in util.SEGMENT_NAMES:
                        #     self.segment_mdl[s] = Segment(self, s)
            if not found:
                raise ValueError("{} not found in subject table.".format(self.id))
            
        


if __name__ == "__main__":
    sub = subject(1)
    print(sub)

