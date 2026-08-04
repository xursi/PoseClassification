from pydantic import Field
from typing import List, Optional, Union, Literal
from sdks.novavision.src.base.model import Package, Detection, Input, Output, Config, Inputs, Configs, Outputs, Request, Response


class InputDetections(Input):
    name: Literal["inputDetections"] = "inputDetections"
    value: List[Detection]
    type: Literal["list"] = "list"

    class Config:
        title = "Detections2"


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


# Inputs, Configs and Requests for PoseClassifier
class PoseClassifierInputs(Inputs):
    inputDetections: InputDetections


class PoseClassifierConfigs(Configs):
    kneeAngleThreshold: KneeAngleThreshold = KneeAngleThreshold()


class PoseClassifierRequest(Request):
    inputs: Optional[PoseClassifierInputs] = None  # Docker başlatma (bootstrap) sırasında girdi olmadığı için Optional
    configs: PoseClassifierConfigs = PoseClassifierConfigs()  # Arayüz formunun (Yii2) çözebilmesi için doğrudan referans (Optional/anyOf kaldırıldı)

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
    outputs: PoseClassifierOutputs  # Arayüz çıkış portunun (outputDetections) çizilebilmesi için doğrudan referans (Optional/anyOf kaldırıldı)


class PoseClassifierExecutor(Config):
    name: Literal["PoseClassifier"] = "PoseClassifier"
    value: Union[PoseClassifierRequest, PoseClassifierResponse]
    type: Literal["object"] = "object"
    field: Literal["option"] = "option"

    class Config:
        title = "Static Pose Classification"
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
