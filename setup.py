from setuptools import setup


setup(
    name="novavision-pose-classification",
    version="0.1.0",
    description="Simple task-based pose classification capsule for NovaVision",
    install_requires=["sdk"],
    packages=[
        "novavision.pose_classification",
        "novavision.pose_classification.models",
        "novavision.pose_classification.executors",
        "novavision.pose_classification.utils",
    ],
    package_dir={"novavision.pose_classification": "src"},
    python_requires=">=3.8",
)
