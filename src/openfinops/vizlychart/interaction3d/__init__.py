"""Advanced 3D interaction libraries for Vizly."""

from .camera import (
    CameraController,
    OrbitController,
    FlyController,
    FirstPersonController,
)
from .gestures import GestureRecognizer, TouchHandler, MouseHandler
from .selection import SelectionManager, RaycastSelector, BoxSelector
from .manipulation import Transform3D, ManipulatorGizmo, ObjectManipulator
from .navigation import NavigationController, PathPlanner, Waypoint
from .animation import CameraAnimator, ObjectAnimator, KeyFrameSystem
from .vr import VRRenderer, ARRenderer, SpatialController
from .scene import Scene3D, Object3D, Cube, Sphere, create_cube, create_sphere
from .advanced import AdvancedCamera, GestureRecognizer as AdvancedGestureRecognizer, PhysicsEngine, Interactive3DObject, Advanced3DScene
from .scene_manager import (
    Transform3D as SceneTransform3D,
    BoundingBox,
    SceneObject,
    MeshObject,
    PointCloudObject,
    SceneGraph,
    RenderPipeline,
    Scene3DManager,
)

__all__ = [
    # Camera controls
    "CameraController",
    "OrbitController",
    "FlyController",
    "FirstPersonController",
    # Gesture recognition
    "GestureRecognizer",
    "TouchHandler",
    "MouseHandler",
    # Selection systems
    "SelectionManager",
    "RaycastSelector",
    "BoxSelector",
    # Object manipulation
    "Transform3D",
    "ManipulatorGizmo",
    "ObjectManipulator",
    # Navigation
    "NavigationController",
    "PathPlanner",
    "Waypoint",
    # Animation
    "CameraAnimator",
    "ObjectAnimator",
    "KeyFrameSystem",
    # VR/AR support
    "VRRenderer",
    "ARRenderer",
    "SpatialController",
    # Scene management
    "Scene3D",
    "Object3D",
    "Cube",
    "Sphere",
    "create_cube",
    "create_sphere",
    # Advanced 3D features
    "AdvancedCamera",
    "AdvancedGestureRecognizer",
    "PhysicsEngine",
    "Interactive3DObject",
    "Advanced3DScene",
    # Scene graph management
    "SceneTransform3D",
    "BoundingBox",
    "SceneObject",
    "MeshObject",
    "PointCloudObject",
    "SceneGraph",
    "RenderPipeline",
    "Scene3DManager",
]
