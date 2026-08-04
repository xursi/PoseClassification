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


# Custom Weight Filepicker configs (matching Yolo)
class CustomFieldStorageID(Config):
    name: Literal["Id"] = "Id"
    value: int = 0
    type: Literal["number"] = "number"
    field: Literal["filePicker"] = "filePicker"
    restart: Literal[True] = True

    class Config:
        title = "Storage Source"
        json_schema_extra = {
            "shortDescription": "File Selector",
            "class": "portalium\\storage\\widgets\\FilePicker",
            "options": {
                "multiple": 0,
                "returnAttribute": ["name"],
                "name": "app::logo_wide"
            }
        }
        schema_extra = {
            "shortDescription": "File Selector",
            "class": "portalium\\storage\\widgets\\FilePicker",
            "options": {
                "multiple": 0,
                "returnAttribute": ["name"],
                "name": "app::logo_wide"
            }
        }


class CustomFieldStorage(Config):
    name: Literal["storageID"] = "storageID"
    storageID: CustomFieldStorageID  # Instantiation kaldırıldı, Yii2 PHP parser uyumluluğu için
    value: Literal["storageID"] = "storageID"
    type: Literal["object"] = "object"
    field: Literal["option"] = "option"

    class Config:
        title = "Storage ID"


# Dropdown Hierarchies
class ConfigGeometryBased(Config):
    kneeAngleThreshold: KneeAngleThreshold  # Instantiation kaldırıldı, Yii2 PHP parser uyumluluğu için
    name: Literal["Geometry-Based"] = "Geometry-Based"
    value: Literal["Geometry-Based"] = "Geometry-Based"
    type: Literal["string"] = "string"
    field: Literal["option"] = "option"

    class Config:
        title = "Geometry-Based"


class DefaultModelName(Config):
    name: Literal["DefaultModelName"] = "DefaultModelName"
    value: str = "pose_mlp_v1.pth"
    type: Literal["string"] = "string"
    field: Literal["textInput"] = "textInput"

    class Config:
        title = "Default Model Name"
        json_schema_extra = {
            "shortDescription": "Default Pre-trained Model Weights"
        }
        schema_extra = {
            "shortDescription": "Default Pre-trained Model Weights"
        }


class PoseModelPreTrained(Config):
    defaultModelName: DefaultModelName  # Instantiation kaldırıldı, Yii2 PHP parser uyumluluğu için
    name: Literal["PreTrained"] = "PreTrained"
    value: Literal["PreTrained"] = "PreTrained"
    type: Literal["string"] = "string"
    field: Literal["option"] = "option"

    class Config:
        title = "Pre-trained Model"


class PoseModelCustomWeight(Config):
    customFieldStorage: CustomFieldStorage  # Instantiation kaldırıldı, Yii2 PHP parser uyumluluğu için
    name: Literal["CustomWeight"] = "CustomWeight"
    value: Literal["CustomWeight"] = "CustomWeight"
    type: Literal["string"] = "string"
    field: Literal["option"] = "option"

    class Config:
        title = "Custom Weight"


class PoseModelSelection(Config):
    name: Literal["PoseModelSelection"] = "PoseModelSelection"
    value: Union[PoseModelPreTrained, PoseModelCustomWeight]
    type: Literal["object"] = "object"
    field: Literal["dependentDropdownlist"] = "dependentDropdownlist"

    # dependentDropdownlist'in kendisi seçilebilir olduğu için target: value kaldırıldı (Yolo tarzı)
    class Config:
        title = "Model Source"
        json_schema_extra = {
            "shortDescription": "Select where to load model weights"
        }
        schema_extra = {
            "shortDescription": "Select where to load model weights"
        }


class ConfigModelBased(Config):
    poseModelSelection: PoseModelSelection
    name: Literal["Model-Based"] = "Model-Based"
    value: Literal["Model-Based"] = "Model-Based"
    type: Literal["string"] = "string"
    field: Literal["option"] = "option"

    class Config:
        title = "Model-Based"


class PoseMethod(Config):
    name: Literal["PoseMethod"] = "PoseMethod"
    value: Union[ConfigGeometryBased, ConfigModelBased]
    type: Literal["object"] = "object"
    field: Literal["dependentDropdownlist"] = "dependentDropdownlist"

    # dependentDropdownlist'in kendisi seçilebilir olduğu için target: value kaldırıldı (Yolo tarzı)
    class Config:
        title = "Classification Method"
        json_schema_extra = {
            "shortDescription": "Select Pose Classification Method"
        }
        schema_extra = {
            "shortDescription": "Select Pose Classification Method"
        }


# Inputs, Configs and Requests for PoseClassifier
class PoseClassifierInputs(Inputs):
    inputDetections: InputDetections


class PoseClassifierConfigs(Configs):
    poseMethod: PoseMethod


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
