

from roboflow import Roboflow
rf = Roboflow(api_key="KpOU7hm9Q8bT4rnxZC20")
project = rf.workspace("nomuunaa").project("car_type_detector-yadjv")
version = project.version(1)
dataset = version.download("yolov8")