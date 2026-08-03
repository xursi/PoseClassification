from pydantic import Field
from typing import List, Optional, Union, Literal
from sdks.novavision.src.base.model import Package, Detection, Input, Output, Config, Inputs, Configs, Outputs, Request, Response


class InputDetections(Input):
    name: Literal["inputDetections"] = "inputDetections"
    value: List[Detection]
    type: str = "list"

    class Config:
        title = "Detections"


class OutputDetections(Output):
    name: Literal["outputDetections"] = "outputDetections"
    value: List[Detection]
    type: str = "list"

    class Config:
        title = "Detections"


# Configs for PoseClassifier (Static Image)
class PoseMethod(Config):
    name: Literal["poseMethod"] = "poseMethod"
    value: Literal["Geometry-Based", "Model-Based"] = "Geometry-Based"
    type: Literal["string"] = "string"
    field: Literal["option"] = "option"

    class Config:
        title = "Pose Classification Method"


class PoseModelPath(Config):
    name: Literal["poseModelPath"] = "poseModelPath"
    value: str = ""
    type: Literal["string"] = "string"
    field: Literal["textInput"] = "textInput"
    placeHolder: Literal["path/to/pose_model.pkl"] = "path/to/pose_model.pkl"

    class Config:
        title = "Pose Model Path (for Model-Based)"


class KneeAngleThreshold(Config):
    name: Literal["kneeAngleThreshold"] = "kneeAngleThreshold"
    value: float = Field(default=130.0, ge=60.0, le=180.0)
    type: Literal["number"] = "number"
    field: Literal["textInput"] = "textInput"

    class Config:
        title = "Knee Angle Threshold"


# Configs for ActionClassifier (Video Buffer)
class ActionMethod(Config):
    name: Literal["actionMethod"] = "actionMethod"
    value: Literal["Geometry-Based", "Model-Based"] = "Geometry-Based"
    type: Literal["string"] = "string"
    field: Literal["option"] = "option"

    class Config:
        title = "Action Classification Method"


class ActionModelPath(Config):
    name: Literal["actionModelPath"] = "actionModelPath"
    value: str = ""
    type: Literal["string"] = "string"
    field: Literal["textInput"] = "textInput"
    placeHolder: Literal["path/to/action_model.pkl"] = "path/to/action_model.pkl"

    class Config:
        title = "Action Model Path (for Model-Based)"


class VelocityThreshold(Config):
    name: Literal["velocityThreshold"] = "velocityThreshold"
    value: float = Field(default=3.0, ge=0.5, le=50.0)
    type: Literal["number"] = "number"
    field: Literal["textInput"] = "textInput"

    class Config:
        title = "Walking/Running Speed Threshold"


# Inputs, Configs and Requests for PoseClassifier
class PoseClassifierInputs(Inputs):
    inputDetections: InputDetections


class PoseClassifierConfigs(Configs):
    poseMethod: PoseMethod = PoseMethod()
    poseModelPath: PoseModelPath = PoseModelPath()
    kneeAngleThreshold: KneeAngleThreshold = KneeAngleThreshold()


class PoseClassifierRequest(Request):
    inputs: Optional[PoseClassifierInputs]
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


# Inputs, Configs and Requests for ActionClassifier
class ActionClassifierInputs(Inputs):
    inputDetections: InputDetections


class ActionClassifierConfigs(Configs):
    actionMethod: ActionMethod = ActionMethod()
    actionModelPath: ActionModelPath = ActionModelPath()
    velocityThreshold: VelocityThreshold = VelocityThreshold()


class ActionClassifierRequest(Request):
    inputs: Optional[ActionClassifierInputs]
    configs: ActionClassifierConfigs

    class Config:
        json_schema_extra = {
            "target": "configs"
        }


class ActionClassifierOutputs(Outputs):
    outputDetections: OutputDetections


class ActionClassifierResponse(Response):
    outputs: ActionClassifierOutputs


class ActionClassifierExecutor(Config):
    name: Literal["ActionClassifier"] = "ActionClassifier"
    value: Union[ActionClassifierRequest, ActionClassifierResponse]
    type: Literal["object"] = "object"
    field: Literal["option"] = "option"

    class Config:
        title = "Dynamic Action Classification"


# Global Component Configurations
class ConfigExecutor(Config):
    name: Literal["ConfigExecutor"] = "ConfigExecutor"
    value: Union[PoseClassifierExecutor, ActionClassifierExecutor]
    type: Literal["executor"] = "executor"
    field: Literal["dependentDropdownlist"] = "dependentDropdownlist"
    restart: Literal[True] = True

    class Config:
        title = "Execution Mode"
        json_schema_extra = {
            "shortDescription": "Select whether to classify static poses or dynamic actions",
            "target": "value"
        }


class PackageConfigs(Configs):
    executor: ConfigExecutor


class PackageModel(Package):
    configs: PackageConfigs
    type: Literal["component"] = "component"
    name: Literal["PoseActionClassifier"] = "PoseActionClassifier"
