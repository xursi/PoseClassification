from pydantic import Field
from typing import List, Optional, Union, Literal
from sdks.novavision.src.base.model import Package, Detection, Input, Output, Config, Inputs, Configs, Outputs, Request, Response


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


# Inputs, Configs and Requests for PoseClassifier
class PoseClassifierInputs(Inputs):
    inputDetections: InputDetections


class PoseClassifierConfigs(Configs):
    kneeAngleThreshold: KneeAngleThreshold = KneeAngleThreshold()


class PoseClassifierRequest(Request):
    inputs: PoseClassifierInputs
    configs: PoseClassifierConfigs

    class Config:
        json_schema_extra = {
            "target": "configs"
        }


class PoseClassifierOutputs(Outputs):
    outputDetections: OutputDetections


class PoseClassifierResponse(Response):
    outputs: PoseClassifierOutputs


class PoseClassifierExecutor(Config):
    name: Literal["PoseClassifier"] = "PoseClassifier"
    value: Union[PoseClassifierRequest, PoseClassifierResponse]
    type: Literal["object"] = "object"
    field: Literal["option"] = "option"

    class Config:
        title = "Static Pose Classification"


# Dummy Executor to force Pydantic v2 to generate an 'anyOf' schema
class DummyExecutor(Config):
    name: Literal["DummyExecutor"] = "DummyExecutor"
    value: Literal["Dummy"] = "Dummy"
    type: Literal["string"] = "string"
    field: Literal["option"] = "option"

    class Config:
        title = "Dummy"


# Global Capsule Configurations
class ConfigExecutor(Config):
    name: Literal["ConfigExecutor"] = "ConfigExecutor"
    value: Union[PoseClassifierExecutor, DummyExecutor]
    type: Literal["executor"] = "executor"
    field: Literal["dependentDropdownlist"] = "dependentDropdownlist"
    restart: Literal[True] = True

    class Config:
        title = "Execution Mode"
        json_schema_extra = {
            "shortDescription": "Classify static poses",
            "target": "value"
        }


class PackageConfigs(Configs):
    executor: ConfigExecutor


class PackageModel(Package):
    configs: PackageConfigs
    type: Literal["capsule"] = "capsule"  # capsule tipine dönüştürüldü
    name: Literal["PoseClassification"] = "PoseClassification"
