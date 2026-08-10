from pydantic import Field
from typing import List, Optional, Union, Literal
from sdks.novavision.src.base.model import Package, Detection, Input, Output, Config, Inputs, Configs, Outputs, Request, Response, KeyPoints


class KeyPoints(KeyPoints):
    confidence: Optional[float] = None


class Detection(Detection):
    keyPoints: Optional[List[KeyPoints]] = None
    classPosition: Optional[str] = None


class InputDetections(Input):
    name: Literal["inputDetections"] = "inputDetections"
    value: List[Detection]
    type: Literal["list"] = "list"

    class Config:
        title = "Detections"


class OutputDetections(Output):
    name: Literal["outputDetections"] = "outputDetections"
    value: List[Detection]
    type: Literal["list"] = "list"

    class Config:
        title = "Detections"


class KneeAngleThreshold(Config):
    name: Literal["kneeAngleThreshold"] = "kneeAngleThreshold"
    value: float = 130.0
    type: Literal["number"] = "number"
    field: Literal["textInput"] = "textInput"

    class Config:
        title = "Knee Angle Threshold"
        json_schema_extra = {
            "shortDescription": "Knee angle limit (60-180)"
        }
        schema_extra = {
            "shortDescription": "Knee angle limit (60-180)"
        }


class BackTiltThreshold(Config):
    name: Literal["backTiltThreshold"] = "backTiltThreshold"
    value: float = 20.0
    type: Literal["number"] = "number"
    field: Literal["textInput"] = "textInput"

    class Config:
        title = "Back Tilt Threshold"
        json_schema_extra = {
            "shortDescription": "Back flexion angle threshold (10-45)"
        }
        schema_extra = {
            "shortDescription": "Back flexion angle threshold (10-45)"
        }


# Dropdown Options for PoseClassifier
class PoseClassMode(Config):
    kneeAngleThreshold: KneeAngleThreshold
    name: Literal["Standard Pose Classification"] = "Standard Pose Classification"
    value: Literal["Standard Pose Classification"] = "Standard Pose Classification"
    type: Literal["string"] = "string"
    field: Literal["option"] = "option"

    class Config:
        title = "Standard Pose Classification"


class ErgonomicMode(Config):
    backTiltThreshold: BackTiltThreshold
    name: Literal["Ergonomic Safety Assessment"] = "Ergonomic Safety Assessment"
    value: Literal["Ergonomic Safety Assessment"] = "Ergonomic Safety Assessment"
    type: Literal["string"] = "string"
    field: Literal["option"] = "option"

    class Config:
        title = "Ergonomic Safety Assessment"


class PoseGeometryMode(Config):
    name: Literal["poseGeometryMode"] = "poseGeometryMode"
    value: Union[PoseClassMode, ErgonomicMode]
    type: Literal["object"] = "object"
    field: Literal["dependentDropdownlist"] = "dependentDropdownlist"

    class Config:
        title = "Geometry Mode"
        json_schema_extra = {
            "shortDescription": "Select Mode of Geometry Analysis"
        }
        schema_extra = {
            "shortDescription": "Select Mode of Geometry Analysis"
        }


# Inputs, Configs and Requests for PoseClassifier
class PoseClassifierInputs(Inputs):
    inputDetections: InputDetections


class PoseClassifierConfigs(Configs):
    poseGeometryMode: PoseGeometryMode


class PoseClassifierRequest(Request):
    inputs: Optional[PoseClassifierInputs] = None
    configs: PoseClassifierConfigs

    class Config:
        json_schema_extra = {
            "target": "configs"
        }
        schema_extra = {
            "target": "configs"
        }


class PoseClassifierOutputs(Outputs):
    outputDetections: OutputDetections


class PoseClassifierResponse(Response):
    outputs: PoseClassifierOutputs


# Veritabanında kayıtlı olan orijinal "PoseClassifier" ismine sadık kalındı
class PoseClassifierExecutor(Config):
    name: Literal["PoseClassifier"] = "PoseClassifier"
    value: Union[PoseClassifierRequest, PoseClassifierResponse]
    type: Literal["object"] = "object"
    field: Literal["option"] = "option"

    class Config:
        title = "Geometry Pose Classifier"
        json_schema_extra = {
            "target": {
                "value": 0
            }
        }
        schema_extra = {
            "target": {
                "value": 0
            }
        }


# Global Capsule Configurations
class ConfigExecutor(Config):
    name: Literal["ConfigExecutor"] = "ConfigExecutor"
    value: Union[PoseClassifierExecutor]
    type: Literal["executor"] = "executor"
    field: Literal["dependentDropdownlist"] = "dependentDropdownlist"
    restart: Literal[True] = True

    class Config:
        title = "Task"
        json_schema_extra = {
            "target": "value"
        }
        schema_extra = {
            "target": "value"
        }


class PackageConfigs(Configs):
    executor: ConfigExecutor


class PackageModel(Package):
    configs: PackageConfigs
    type: Literal["capsule"] = "capsule"
    name: Literal["PoseClassification"] = "PoseClassification"
