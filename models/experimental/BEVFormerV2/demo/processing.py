# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.

# SPDX-License-Identifier: Apache-2.0

import numpy as np
import os
from pyquaternion import Quaternion

EMBEDDED_NUSCENES_DATA = {
    "samples": {
        "3e8750f331d7499e9b5123e9eb70f2e2": {
            "token": "3e8750f331d7499e9b5123e9eb70f2e2",
            "timestamp": 1533151603547590,
            "scene_token": "scene_001",
            "data": {
                "LIDAR_TOP": "lidar_top_sample_001",
                "CAM_FRONT": "cam_front_sample_001",
                "CAM_FRONT_RIGHT": "cam_front_right_sample_001",
                "CAM_FRONT_LEFT": "cam_front_left_sample_001",
                "CAM_BACK": "cam_back_sample_001",
                "CAM_BACK_LEFT": "cam_back_left_sample_001",
                "CAM_BACK_RIGHT": "cam_back_right_sample_001",
            },
            "anns": [],
        }
    },
    "sample_data": {
        "lidar_top_sample_001": {
            "token": "lidar_top_sample_001",
            "sample_token": "3e8750f331d7499e9b5123e9eb70f2e2",
            "ego_pose_token": "ego_pose_001",
            "calibrated_sensor_token": "lidar_calib_001",
            "timestamp": 1533151603547590,
            "filename": "samples/LIDAR_TOP/n008-2018-08-01-15-16-36-0400__LIDAR_TOP__1533151603547590.pcd.bin",
            "channel": "LIDAR_TOP",
        },
        "cam_front_sample_001": {
            "token": "cam_front_sample_001",
            "sample_token": "3e8750f331d7499e9b5123e9eb70f2e2",
            "ego_pose_token": "ego_pose_cam_front_001",
            "calibrated_sensor_token": "cam_front_calib_001",
            "timestamp": 1533151603512404,
            "filename": "samples/CAM_FRONT/n008-2018-08-01-15-16-36-0400__CAM_FRONT__1533151603512404.jpg",
            "channel": "CAM_FRONT",
            "width": 1600,
            "height": 900,
        },
        "cam_front_right_sample_001": {
            "token": "cam_front_right_sample_001",
            "sample_token": "3e8750f331d7499e9b5123e9eb70f2e2",
            "ego_pose_token": "ego_pose_cam_front_right_001",
            "calibrated_sensor_token": "cam_front_right_calib_001",
            "timestamp": 1533151603520482,
            "filename": "samples/CAM_FRONT_RIGHT/n008-2018-08-01-15-16-36-0400__CAM_FRONT_RIGHT__1533151603520482.jpg",
            "channel": "CAM_FRONT_RIGHT",
            "width": 1600,
            "height": 900,
        },
        "cam_front_left_sample_001": {
            "token": "cam_front_left_sample_001",
            "sample_token": "3e8750f331d7499e9b5123e9eb70f2e2",
            "ego_pose_token": "ego_pose_cam_front_left_001",
            "calibrated_sensor_token": "cam_front_left_calib_001",
            "timestamp": 1533151603504799,
            "filename": "samples/CAM_FRONT_LEFT/n008-2018-08-01-15-16-36-0400__CAM_FRONT_LEFT__1533151603504799.jpg",
            "channel": "CAM_FRONT_LEFT",
            "width": 1600,
            "height": 900,
        },
        "cam_back_sample_001": {
            "token": "cam_back_sample_001",
            "sample_token": "3e8750f331d7499e9b5123e9eb70f2e2",
            "ego_pose_token": "ego_pose_cam_back_001",
            "calibrated_sensor_token": "cam_back_calib_001",
            "timestamp": 1533151603537558,
            "filename": "samples/CAM_BACK/n008-2018-08-01-15-16-36-0400__CAM_BACK__1533151603537558.jpg",
            "channel": "CAM_BACK",
            "width": 1600,
            "height": 900,
        },
        "cam_back_left_sample_001": {
            "token": "cam_back_left_sample_001",
            "sample_token": "3e8750f331d7499e9b5123e9eb70f2e2",
            "ego_pose_token": "ego_pose_cam_back_left_001",
            "calibrated_sensor_token": "cam_back_left_calib_001",
            "timestamp": 1533151603547405,
            "filename": "samples/CAM_BACK_LEFT/n008-2018-08-01-15-16-36-0400__CAM_BACK_LEFT__1533151603547405.jpg",
            "channel": "CAM_BACK_LEFT",
            "width": 1600,
            "height": 900,
        },
        "cam_back_right_sample_001": {
            "token": "cam_back_right_sample_001",
            "sample_token": "3e8750f331d7499e9b5123e9eb70f2e2",
            "ego_pose_token": "ego_pose_cam_back_right_001",
            "calibrated_sensor_token": "cam_back_right_calib_001",
            "timestamp": 1533151603528113,
            "filename": "samples/CAM_BACK_RIGHT/n008-2018-08-01-15-16-36-0400__CAM_BACK_RIGHT__1533151603528113.jpg",
            "channel": "CAM_BACK_RIGHT",
            "width": 1600,
            "height": 900,
        },
    },
    "ego_pose": {
        "ego_pose_001": {
            "token": "ego_pose_001",
            "timestamp": 1533151603547590,
            "rotation": [-0.9687876119182126, -0.004506968075376869, -0.00792272203393983, 0.24772460658591755],
            "translation": [600.1202137947669, 1647.490776275174, 0.0],
        },
        "ego_pose_cam_front_001": {
            "token": "ego_pose_cam_front_001",
            "timestamp": 1533151603512404,
            "rotation": [-0.9687876119182126, -0.004506968075376869, -0.00792272203393983, 0.24772460658591755],
            "translation": [599.849775495386, 1647.6411294309523, 0.0],
        },
        "ego_pose_cam_front_right_001": {
            "token": "ego_pose_cam_front_right_001",
            "timestamp": 1533151603520482,
            "rotation": [-0.9687599514054591, -0.004456697153369989, -0.007899682341935369, 0.2478343991908144],
            "translation": [599.9118549287866, 1647.606633933739, 0.0],
        },
        "ego_pose_cam_front_left_001": {
            "token": "ego_pose_cam_front_left_001",
            "timestamp": 1533151603504799,
            "rotation": [-0.9688139605779101, -0.004563341952090969, -0.007978863995668394, 0.24762405442308663],
            "translation": [599.7840556879353, 1647.6742918430988, 0.0],
        },
        "ego_pose_cam_back_001": {
            "token": "ego_pose_cam_back_001",
            "timestamp": 1533151603537558,
            "rotation": [-0.9687345485285538, -0.0043670388304257405, -0.007816404838658813, 0.24793791011951208],
            "translation": [600.0344866024128, 1647.5585532545996, 0.0],
        },
        "ego_pose_cam_back_left_001": {
            "token": "ego_pose_cam_back_left_001",
            "timestamp": 1533151603547405,
            "rotation": [-0.968669701688471, -0.004043399262151301, -0.007666594265959211, 0.24820129589817977],
            "translation": [600.1152100852063, 1647.4951638031797, 0.0],
        },
        "ego_pose_cam_back_right_001": {
            "token": "ego_pose_cam_back_right_001",
            "timestamp": 1533151603528113,
            "rotation": [-0.9687345485285538, -0.0043670388304257405, -0.007816404838658813, 0.24793791011951208],
            "translation": [599.9705034252927, 1647.574034904777, 0.0],
        },
    },
    "calibrated_sensor": {
        "lidar_calib_001": {
            "token": "lidar_calib_001",
            "sensor_token": "lidar_top_sensor",
            "translation": [0.985793, 0.0, 1.84019],
            "rotation": [0.706749235646644, -0.015300993788500868, 0.01739745181256607, -0.7070846669051719],
            "camera_intrinsic": [],
        },
        "cam_front_calib_001": {
            "token": "cam_front_calib_001",
            "sensor_token": "cam_front_sensor",
            "translation": [1.72200568478, 0.00475453292289, 1.49491291905],
            "rotation": [0.5077241387638071, -0.4973392230703816, 0.49837167536166627, -0.4964832014373754],
            "camera_intrinsic": [
                [1252.8131021185304, 0.0, 826.588114781398],
                [0.0, 1252.8131021185304, 469.9846626224581],
                [0.0, 0.0, 1.0],
            ],
        },
        "cam_front_right_calib_001": {
            "token": "cam_front_right_calib_001",
            "sensor_token": "cam_front_right_sensor",
            "translation": [1.58082565783, -0.499078711449, 1.51749368405],
            "rotation": [0.20335173766558642, -0.19146333228946724, 0.6785710044972951, -0.6793609166212989],
            "camera_intrinsic": [
                [1256.7485116440405, 0.0, 817.7887570959712],
                [0.0, 1256.7485116440403, 451.9541780095127],
                [0.0, 0.0, 1.0],
            ],
        },
        "cam_front_left_calib_001": {
            "token": "cam_front_left_calib_001",
            "sensor_token": "cam_front_left_sensor",
            "translation": [1.71055672796, 0.504901370698, 1.51155957763],
            "rotation": [0.6866719672266984, -0.6805822551352499, 0.19082498283193827, -0.1778431068382811],
            "camera_intrinsic": [
                [1257.8625342125129, 0.0, 827.2410631095686],
                [0.0, 1257.8625342125129, 450.915498205774],
                [0.0, 0.0, 1.0],
            ],
        },
        "cam_back_calib_001": {
            "token": "cam_back_calib_001",
            "sensor_token": "cam_back_sensor",
            "translation": [0.05524611077, 0.0107882366898, 1.56794286957],
            "rotation": [0.5067997344989889, -0.4977567019405021, -0.4987849934090844, 0.496594225837321],
            "camera_intrinsic": [
                [796.8910634503094, 0.0, 857.7774326863696],
                [0.0, 796.8910634503094, 476.8848988407415],
                [0.0, 0.0, 1.0],
            ],
        },
        "cam_back_left_calib_001": {
            "token": "cam_back_left_calib_001",
            "sensor_token": "cam_back_left_sensor",
            "translation": [1.03322032896, 0.484795032713, 1.59097015059],
            "rotation": [0.37005309970281335, -0.36156023815986623, -0.607689082777746, 0.6126617854318387],
            "camera_intrinsic": [
                [1254.9860565800168, 0.0, 829.5769333630991],
                [0.0, 1254.9860565800168, 467.1680561863987],
                [0.0, 0.0, 1.0],
            ],
        },
        "cam_back_right_calib_001": {
            "token": "cam_back_right_calib_001",
            "sensor_token": "cam_back_right_sensor",
            "translation": [1.05945173053, -0.46720294852, 1.55050857555],
            "rotation": [0.13819187705364147, -0.13796718183628456, -0.6893329941542625, 0.697630335509333],
            "camera_intrinsic": [
                [1249.9629280788233, 0.0, 825.3768045375984],
                [0.0, 1249.9629280788233, 462.54816385708756],
                [0.0, 0.0, 1.0],
            ],
        },
    },
    "sensor": {
        "lidar_top_sensor": {"token": "lidar_top_sensor", "channel": "LIDAR_TOP"},
        "cam_front_sensor": {"token": "cam_front_sensor", "channel": "CAM_FRONT"},
        "cam_front_right_sensor": {"token": "cam_front_right_sensor", "channel": "CAM_FRONT_RIGHT"},
        "cam_front_left_sensor": {"token": "cam_front_left_sensor", "channel": "CAM_FRONT_LEFT"},
        "cam_back_sensor": {"token": "cam_back_sensor", "channel": "CAM_BACK"},
        "cam_back_left_sensor": {"token": "cam_back_left_sensor", "channel": "CAM_BACK_LEFT"},
        "cam_back_right_sensor": {"token": "cam_back_right_sensor", "channel": "CAM_BACK_RIGHT"},
    },
}


class EmbeddedNuScenesWrapper:
    def __init__(self, dataroot="", version="v1.0-mini"):
        self.dataroot = dataroot
        self.version = version
        self.data = EMBEDDED_NUSCENES_DATA
        self.colormap = {
            "vehicle.bicycle": (255, 61, 99),
            "vehicle.construction": (255, 158, 0),
            "movable_object.trafficcone": (255, 99, 71),
            "vehicle.car": (0, 0, 230),
            "vehicle.truck": (255, 140, 0),
            "vehicle.bus": (255, 127, 80),
            "vehicle.trailer": (255, 20, 147),
            "human.pedestrian": (255, 255, 0),
            "vehicle.motorcycle": (255, 61, 99),
        }
        self.explorer = type("Explorer", (), {"colormap": self.colormap})()

    def get(self, table_name, token):
        table_map = {
            "sample": "samples",
            "sample_data": "sample_data",
            "ego_pose": "ego_pose",
            "calibrated_sensor": "calibrated_sensor",
            "sensor": "sensor",
            "sample_annotation": "sample_annotation",
        }

        actual_table = table_map.get(table_name, table_name)

        if actual_table in self.data and token in self.data[actual_table]:
            return self.data[actual_table][token]
        return {}

    def get_sample_data_path(self, sample_data_token):
        sample_data = self.get("sample_data", sample_data_token)
        return os.path.join(self.dataroot, sample_data.get("filename", ""))

    def get_sample_data(self, sample_data_token, box_vis_level=None, selected_anntokens=None):
        sample_data = self.get("sample_data", sample_data_token)
        cs_record = self.get("calibrated_sensor", sample_data.get("calibrated_sensor_token", ""))
        sensor_record = self.get("sensor", cs_record.get("sensor_token", ""))
        pose_record = self.get("ego_pose", sample_data.get("ego_pose_token", ""))

        data_path = self.get_sample_data_path(sample_data_token)
        boxes = []
        camera_intrinsic = np.array(cs_record.get("camera_intrinsic", [[1, 0, 0], [0, 1, 0], [0, 0, 1]]))

        return data_path, boxes, camera_intrinsic

    def get_boxes(self, sample_data_token):
        return []

    def get_box(self, ann_token):
        return None

    def box_velocity(self, ann_token):
        return [0.0, 0.0]


SAMPLE_0_INFO = {
    "token": "3e8750f331d7499e9b5123e9eb70f2e2",
    "can_bus": [0.0] * 18,
    "ego2global_translation": [600.1202137947669, 1647.490776275174, 0.0],
    "ego2global_rotation": [-0.968669701688471, -0.004043399262151301, -0.007666594265959211, 0.24820129589817977],
    "lidar2ego_translation": [0.985793, 0.0, 1.84019],
    "lidar2ego_rotation": [0.706749235646644, -0.015300993788500868, 0.01739745181256607, -0.7070846669051719],
    "timestamp": 1533151603547590.0,
    "cams": {
        "CAM_FRONT": {
            "data_path": "./data/nuscenes/samples/CAM_FRONT/n008-2018-08-01-15-16-36-0400__CAM_FRONT__1533151603512404.jpg",
            "sensor2lidar_rotation": [
                [0.9998801270116863, -0.01013819478916953, -0.01170250458296505],
                [0.012232575354279787, 0.05390463505646997, 0.9984711585316994],
                [-0.009491875857770705, -0.9984946205793245, 0.054022189578496575],
            ],
            "sensor2lidar_translation": [-0.006275140311572613, 0.4437230287236389, -0.3316126716045531],
            "cam_intrinsic": [
                [1252.8131021185304, 0.0, 826.588114781398],
                [0.0, 1252.8131021185304, 469.9846626224581],
                [0.0, 0.0, 1.0],
            ],
        },
        "CAM_FRONT_RIGHT": {
            "data_path": "./data/nuscenes/samples/CAM_FRONT_RIGHT/n008-2018-08-01-15-16-36-0400__CAM_FRONT_RIGHT__1533151603520482.jpg",
            "sensor2lidar_rotation": [
                [0.5372736788664327, -0.001367748330040076, 0.843406855119068],
                [-0.8417394728983844, 0.0620003137815353, 0.5363120554823875],
                [-0.05302502958114645, -0.9980751927362483, 0.0321598489178931],
            ],
            "sensor2lidar_translation": [0.49830135431261624, 0.3730319110657092, -0.30971646952113474],
            "cam_intrinsic": [
                [1256.7485116440405, 0.0, 817.7887570959712],
                [0.0, 1256.7485116440403, 451.9541780095127],
                [0.0, 0.0, 1.0],
            ],
        },
        "CAM_FRONT_LEFT": {
            "data_path": "./data/nuscenes/samples/CAM_FRONT_LEFT/n008-2018-08-01-15-16-36-0400__CAM_FRONT_LEFT__1533151603504799.jpg",
            "sensor2lidar_rotation": [
                [0.5672581511644471, -0.014333426843678844, -0.8234152918257058],
                [0.8228127923991059, 0.051874023077259974, 0.5659400978850749],
                [0.03460200285939523, -0.9985507691673464, 0.041219689390163995],
            ],
            "sensor2lidar_translation": [-0.5023761049415043, 0.22914751525587462, -0.3316580138974885],
            "cam_intrinsic": [
                [1257.8625342125129, 0.0, 827.2410631095686],
                [0.0, 1257.8625342125129, 450.915498205774],
                [0.0, 0.0, 1.0],
            ],
        },
        "CAM_BACK": {
            "data_path": "./data/nuscenes/samples/CAM_BACK/n008-2018-08-01-15-16-36-0400__CAM_BACK__1533151603537558.jpg",
            "sensor2lidar_rotation": [
                [-0.9999283380727066, -0.008594852305998796, -0.008333500644689665],
                [0.007990712947773468, 0.039174287069874525, -0.9992004422232587],
                [0.008914439171549637, -0.9991954282053173, -0.03910280076735698],
            ],
            "sensor2lidar_translation": [-0.009512203183021484, -1.0046424905119693, -0.3205656017442351],
            "cam_intrinsic": [
                [796.8910634503094, 0.0, 857.7774326863696],
                [0.0, 796.8910634503094, 476.8848988407415],
                [0.0, 0.0, 1.0],
            ],
        },
        "CAM_BACK_LEFT": {
            "data_path": "./data/nuscenes/samples/CAM_BACK_LEFT/n008-2018-08-01-15-16-36-0400__CAM_BACK_LEFT__1533151603547405.jpg",
            "sensor2lidar_rotation": [
                [-0.31910314470327766, -0.015891215448150118, -0.9475867518660553],
                [0.9468607653579226, 0.037220814098564134, -0.31948286655727137],
                [0.04034692139792285, -0.999180704512163, 0.0031694895944351597],
            ],
            "sensor2lidar_translation": [-0.48218189331873873, 0.07357368426630728, -0.27649453910384736],
            "cam_intrinsic": [
                [1254.9860565800168, 0.0, 829.5769333630991],
                [0.0, 1254.9860565800168, 467.1680561863987],
                [0.0, 0.0, 1.0],
            ],
        },
        "CAM_BACK_RIGHT": {
            "data_path": "./data/nuscenes/samples/CAM_BACK_RIGHT/n008-2018-08-01-15-16-36-0400__CAM_BACK_RIGHT__1533151603528113.jpg",
            "sensor2lidar_rotation": [
                [-0.38201342410869077, 0.013854058151105826, 0.9240529253638575],
                [-0.923050639707394, 0.043186672761450676, -0.38224655372098476],
                [-0.04520243728526039, -0.9989709587212956, -0.003709891498541602],
            ],
            "sensor2lidar_translation": [0.4673898584139806, -0.08280982434399675, -0.2960748534933302],
            "cam_intrinsic": [
                [1249.9629280788233, 0.0, 825.3768045375984],
                [0.0, 1249.9629280788233, 462.54816385708756],
                [0.0, 0.0, 1.0],
            ],
        },
    },
}


def convert_to_numpy(info):
    result = {}
    for key, value in info.items():
        if key == "cams":
            result[key] = {}
            for cam_name, cam_data in value.items():
                result[key][cam_name] = {}
                for cam_key, cam_value in cam_data.items():
                    if cam_key in ["sensor2lidar_rotation", "cam_intrinsic"]:
                        result[key][cam_name][cam_key] = np.array(cam_value, dtype=np.float32)
                    elif cam_key == "sensor2lidar_translation":
                        result[key][cam_name][cam_key] = np.array(cam_value, dtype=np.float32)
                    else:
                        result[key][cam_name][cam_key] = cam_value
        elif key in ["can_bus", "ego2global_translation", "lidar2ego_translation"]:
            result[key] = np.array(value, dtype=np.float32)
        elif key in ["ego2global_rotation", "lidar2ego_rotation"]:
            result[key] = np.array(value, dtype=np.float32)
        elif key == "timestamp":
            result[key] = float(value)
        else:
            result[key] = value
    return result


def load_demo_data(sample_idx=0):
    if sample_idx == 0:
        info = convert_to_numpy(SAMPLE_0_INFO.copy())
        return [info]
    else:
        raise ValueError(f"Sample {sample_idx} not available. Only sample 0 is available in demo data.")


def prepare_demo_sample(sample_idx=0, data_root=None):
    from models.experimental.BEVFormerV2.tt.ttnn_utils import prepare_sample_images

    if data_root is None:
        data_root = "models/experimental/BEVFormerV2/demo/demo_data"

    infos = load_demo_data(sample_idx)
    info = infos[0]

    img, img_metas = prepare_sample_images(info, data_root)
    return img, img_metas


def generate_synthetic_lidar_points(num_points=34720, x_range=(-50, 50), y_range=(-50, 50), z_range=(-5, 3)):
    x = np.random.uniform(x_range[0], x_range[1], num_points)
    y = np.random.uniform(y_range[0], y_range[1], num_points)
    z = np.random.uniform(z_range[0], z_range[1], num_points)
    intensity = np.random.uniform(0, 255, num_points)
    points = np.stack([x, y, z, intensity], axis=-1).astype(np.float32)
    return points


def save_synthetic_lidar_bin(output_path, num_points=34720):
    points = generate_synthetic_lidar_points(num_points)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    points.tofile(output_path)
    print(f"Synthetic LIDAR data saved to: {output_path}")


class SyntheticLidarPointCloud:
    @staticmethod
    def from_file(file_path):
        if os.path.exists(file_path):
            points = np.fromfile(file_path, dtype=np.float32).reshape(-1, 4)
        else:
            print(f"LIDAR file not found, generating synthetic data...")
            points = generate_synthetic_lidar_points()
        return SyntheticLidarPointCloud(points)

    def __init__(self, points):
        self.points = points

    def render_height(self, ax, view=np.eye(4)):
        points_2d = self.points[:, :2]
        heights = self.points[:, 2]
        ax.scatter(points_2d[:, 0], points_2d[:, 1], c=heights, s=0.5, cmap="viridis")
        ax.set_xlim(-50, 50)
        ax.set_ylim(-50, 50)
        ax.set_aspect("equal")


class BoxVisibility:
    ALL = 0
    ANY = 1
    NONE = 2


class Box:
    def __init__(self, center, size, orientation, label=0, score=0, velocity=(0, 0, 0), name="", token=""):
        self.center = np.array(center)
        self.wlh = np.array(size)
        self.orientation = Quaternion(orientation)
        self.label = label
        self.score = score
        self.velocity = np.array(velocity)
        self.name = name
        self.token = token

    def translate(self, translation):
        """
        Translate box by given translation vector.
        :param translation: Translation vector <np.ndarray: 3>.
        """
        self.center += translation

    def rotate(self, quaternion):
        """
        Rotate box by given quaternion.
        :param quaternion: Rotation quaternion <Quaternion>.
        """
        self.center = quaternion.rotate(self.center)
        self.orientation = quaternion * self.orientation
        self.velocity = quaternion.rotate(self.velocity)

    def corners(self):
        w, l, h = self.wlh
        x_corners = l / 2 * np.array([1, 1, 1, 1, -1, -1, -1, -1])
        y_corners = w / 2 * np.array([1, -1, -1, 1, 1, -1, -1, 1])
        z_corners = h / 2 * np.array([1, 1, -1, -1, 1, 1, -1, -1])
        corners = np.vstack((x_corners, y_corners, z_corners))
        corners = self.orientation.rotation_matrix @ corners
        corners += self.center.reshape(3, 1)
        return corners

    def render(self, axis, view, normalize=False, colors=("b", "r", "k"), linewidth=2):
        corners = self.corners()

        def draw_rect(selected_corners, color):
            prev = selected_corners[-1]
            for corner in selected_corners:
                if normalize:
                    points = np.column_stack((prev, corner))
                    points_2d = view_points(points, view, normalize=True)
                    axis.plot(points_2d[0, :], points_2d[1, :], color=color, linewidth=linewidth)
                else:
                    axis.plot([prev[0], corner[0]], [prev[1], corner[1]], color=color, linewidth=linewidth)
                prev = corner

        if normalize:
            bottom_corners = [corners[:, i] for i in range(4)]
            top_corners = [corners[:, i + 4] for i in range(4)]
        else:
            corners_2d = corners[:2, :]
            bottom_corners = [corners_2d[:, i] for i in range(4)]
            top_corners = [corners_2d[:, i + 4] for i in range(4)]

        draw_rect(bottom_corners, colors[0])
        draw_rect(top_corners, colors[1])

        for i in range(4):
            if normalize:
                points = view_points(corners[:, [i, i + 4]], view, normalize=True)
                axis.plot(points[0, :], points[1, :], color=colors[2], linewidth=linewidth)
            else:
                axis.plot(
                    [corners[0, i], corners[0, i + 4]],
                    [corners[1, i], corners[1, i + 4]],
                    color=colors[2],
                    linewidth=linewidth,
                )


def view_points(points, view_matrix, normalize=True):
    """
    Project 3D points to 2D using view matrix (camera intrinsic or transformation matrix).
    :param points: Points in 3D (3xN) or homogeneous (4xN).
    :param view_matrix: Either 3x3 camera intrinsic or 4x4 transformation matrix.
    :param normalize: Whether to normalize by depth (for camera projection).
    :return: Projected points (3xN).
    """
    nbr_points = points.shape[1]

    # If view_matrix is 4x4, add homogeneous coordinate
    if view_matrix.shape[0] == 4:
        if points.shape[0] == 3:
            points = np.vstack((points, np.ones((1, nbr_points))))
    # If view_matrix is 3x3 (camera intrinsic), keep points as 3xN
    else:
        if points.shape[0] == 4:
            points = points[:3, :]

    points_transformed = view_matrix @ points

    # Normalize by depth for camera projection
    if normalize and points_transformed.shape[0] > 2:
        points_transformed[:2, :] /= points_transformed[2:3, :]

    return points_transformed[:3, :]


def box_in_image(box, intrinsic, imsize, vis_level=BoxVisibility.ANY):
    corners_3d = box.corners()
    corners_2d = view_points(corners_3d, intrinsic, normalize=True)[:2, :]
    visible = np.logical_and.reduce(
        [
            corners_2d[0, :] > 0,
            corners_2d[0, :] < imsize[0],
            corners_2d[1, :] > 0,
            corners_2d[1, :] < imsize[1],
            corners_3d[2, :] > 1,
        ]
    )
    if vis_level == BoxVisibility.ALL:
        return np.all(visible)
    elif vis_level == BoxVisibility.ANY:
        return np.any(visible)
    else:
        return True


class DetectionBox(Box):
    def __init__(
        self,
        sample_token="",
        translation=(0, 0, 0),
        size=(0, 0, 0),
        rotation=(1, 0, 0, 0),
        velocity=(0, 0),
        ego_translation=(0, 0, 0),
        num_pts=-1,
        detection_name="car",
        detection_score=-1.0,
        attribute_name="",
    ):
        super().__init__(translation, size, rotation, name=detection_name, score=detection_score)
        self.sample_token = sample_token
        self.ego_translation = ego_translation
        self.num_pts = num_pts
        self.detection_name = detection_name
        self.detection_score = detection_score
        self.attribute_name = attribute_name

    def serialize(self):
        return {
            "sample_token": self.sample_token,
            "translation": self.center.tolist(),
            "size": self.wlh.tolist(),
            "rotation": [self.orientation.w, self.orientation.x, self.orientation.y, self.orientation.z],
            "velocity": self.velocity[:2].tolist(),
            "ego_translation": self.ego_translation,
            "num_pts": self.num_pts,
            "detection_name": self.detection_name,
            "detection_score": self.detection_score,
            "attribute_name": self.attribute_name,
        }


class EvalBoxes:
    def __init__(self):
        self.boxes = {}

    def add_boxes(self, sample_token, boxes):
        self.boxes[sample_token] = boxes


def category_to_detection_name(category_name):
    mapping = {
        "car": "car",
        "truck": "truck",
        "bus": "bus",
        "trailer": "trailer",
        "construction_vehicle": "construction_vehicle",
        "pedestrian": "pedestrian",
        "motorcycle": "motorcycle",
        "bicycle": "bicycle",
        "traffic_cone": "traffic_cone",
        "barrier": "barrier",
    }
    for key in mapping:
        if key in category_name:
            return mapping[key]
    return "car"


def visualize_sample(
    nusc, sample_token, gt_boxes=None, pred_boxes=None, savepath=None, conf_th=0.15, eval_range=50, verbose=True
):
    print(f"Skipping BEV visualization (simplified demo mode)")
