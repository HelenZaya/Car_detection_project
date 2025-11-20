

from roboflow import Roboflow
rf = Roboflow(api_key="sQvXMKzuRhz6wr4C2TpQ")
project = rf.workspace("carbrandcartype-detection").project("my-first-project-vkkov")
version = project.version(1)
dataset = version.download("yolov8")