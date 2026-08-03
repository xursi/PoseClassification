from sdks.novavision.src.helper.package import PackageHelper
from components.PoseClassification.src.models.PackageModel import (
    PackageModel,
    PackageConfigs,
    ConfigExecutor,
    PoseClassifierOutputs,
    PoseClassifierResponse,
    PoseClassifierExecutor,
    OutputDetections,
)


def build_response_pose(context):
    outputDetections = OutputDetections(value=context.detections)
    outputs = PoseClassifierOutputs(outputDetections=outputDetections)
    response = PoseClassifierResponse(outputs=outputs)
    executor_val = PoseClassifierExecutor(value=response)
    executor = ConfigExecutor(value=executor_val)
    packageConfigs = PackageConfigs(executor=executor)

    package = PackageHelper(packageModel=PackageModel, packageConfigs=packageConfigs)
    packageModel = package.build_model(context)
    return packageModel
