import copy
import enum
import os
import sys
import threading
import time

import cv2
import numpy as np
import open3d
from numba import jit
from open3d.cpu.pybind.geometry import AxisAlignedBoundingBox

from Utils.GlobalVariable import get_global_object
from Utils.Math import Vector3
from Utils.Util import print_error


@jit
def back_project_pixel_to_world(pixel_x: float, pixel_y: float, depth_data, depth_data_width, depth_data_height, depth_view_matrix, depth_project_matrix):
    pixel_y = pixel_y
    x = round(pixel_x * depth_data_width)
    y = round(pixel_y * depth_data_height)
    depth = depth_data[y][x]
    temp_position = np.array([0.0, 0.0, depth, 1.0])
    # print(temp_position, self.depth_project_matrix)
    temp_ndc_position = temp_position @ depth_project_matrix
    ndc_position = np.array([(pixel_x * 2.0 - 1.0) * temp_ndc_position[3], (1.0 - pixel_y * 2.0) * temp_ndc_position[3], temp_ndc_position[2], temp_ndc_position[3]])
    view_project_matrix = depth_view_matrix @ depth_project_matrix
    view_project_matrix_inv = np.linalg.inv(view_project_matrix)
    world_position = ndc_position @ view_project_matrix_inv
    # print(self.depth_view_matrix, self.depth_project_matrix)
    return world_position


class CardInformation:
    def __init__(self, color, threaten, card_pic_path, obj_path, obj_scale):
        self.color = color
        self.threaten = threaten
        self.card_pic_path = card_pic_path
        self.obj_path = obj_path
        self.scale = obj_scale


card_information_template = {
    "0": CardInformation([1.0, 0.0, 0.0], 0.7, "./card.png", "Cube.obj", 400.0),
    "1": CardInformation([0.0, 1.0, 0.0], 0.8, "./card.png", "Cube.obj", 400.0),

    "friend_vehicle": CardInformation([0.0, 1.0, 1.0], 0.8, "./card.png", "Cube.obj", 400.0),
    "friend_drone": CardInformation([0.0, 0.6, 0.6], 0.8, "./card.png", "Cube.obj", 400.0),
}


class Color:
    def __init__(self):
        self.red = 128
        self.green = 128
        self.blue = 128


class Point:
    def __init__(self, x, y, z, red=128, green=128, blue=128):
        self.location = Vector3()
        self.location.x = x
        self.location.y = y
        self.location.z = z
        self.color = Color()
        self.color.red = red
        self.color.green = green
        self.color.blue = blue


class PointCloudInstance:
    def __init__(self):
        self.pcd = None
        self.points = []

    def add_point(self, point: Point):
        self.points.append(point)

    def to_numpy_vertex(self):
        vertices = []
        for point in self.points:
            vertices.append(point.location.x)
            vertices.append(point.location.y)
            vertices.append(point.location.z)
        return np.array(vertices).reshape(-1, 3)

    def to_numpy_color(self):
        colors = []
        for point in self.points:
            colors.append(point.color.red / 255.0)
            colors.append(point.color.green / 255.0)
            colors.append(point.color.blue / 255.0)
        return np.array(colors).reshape(-1, 3)

    def complete(self):
        self.pcd = open3d.geometry.PointCloud()
        vert = self.to_numpy_vertex().reshape(-1, 3)
        vert[:, 1] = -vert[:, 1]
        self.pcd.points = open3d.utility.Vector3dVector(vert)
        self.pcd.colors = open3d.utility.Vector3dVector(self.to_numpy_color())

    def generate_pcd_directly(self, vertices, colors):
        self.pcd = open3d.geometry.PointCloud()
        vert = vertices.reshape(-1, 3)
        vert[:, 1] = -vert[:, 1]
        self.pcd.points = open3d.utility.Vector3dVector(vert)
        if colors is not None:
            self.pcd.colors = open3d.utility.Vector3dVector(colors.reshape(-1, 3) / 255.0)

    def get_pcd(self):
        return self.pcd


# @enum.unique
# class VirtualMapViewerState(enum.Enum):
#     IDLE = 0
#     RUNNING = 1
#     STOP = 2


@enum.unique
class VirtualMapDataConvertState(enum.Enum):
    IDLE = 0
    RUNNING = 1
    STOP = 2


class DynamicModel:
    def __init__(self, color, path, scale):
        self.color = color
        self.x = 0.0
        self.y = 0.0
        self.z = 0.0

        self.model: open3d.visualization.rendering.TriangleMeshModel = open3d.io.read_triangle_model(path)

        aabb = None
        for mesh in self.model.meshes:
            m: open3d.geometry.TriangleMesh = mesh.mesh
            m.paint_uniform_color((color[0], color[1], color[2]))
            m.scale(scale, m.get_center())
            if aabb is None:
                aabb = m.get_axis_aligned_bounding_box()
            else:
                aabb += m.get_axis_aligned_bounding_box()

        min_bound = aabb.min_bound - [100.0, 100.0, 100.0]
        max_bound = aabb.max_bound + [100.0, 100.0, 100.0]

        bounding_box_vertices = np.array([
            [min_bound[0], min_bound[1], min_bound[2]],
            [max_bound[0], min_bound[1], min_bound[2]],
            [min_bound[0], max_bound[1], min_bound[2]],
            [min_bound[0], min_bound[1], max_bound[2]],
            [max_bound[0], max_bound[1], max_bound[2]],
            [min_bound[0], max_bound[1], max_bound[2]],
            [max_bound[0], min_bound[1], max_bound[2]],
            [max_bound[0], max_bound[1], min_bound[2]],
        ])

        lines = [[0, 1], [0, 2], [1, 7], [2, 7],
                 [3, 6], [3, 5], [4, 6], [4, 5],
                 [1, 6], [4, 7], [2, 5], [0, 3]]

        self.aabb_line = open3d.geometry.LineSet()
        self.aabb_line.points = open3d.utility.Vector3dVector(bounding_box_vertices)
        self.aabb_line.lines = open3d.utility.Vector2iVector(lines)
        colors = [[color[0], color[1], color[2]] for _ in range(len(lines))]
        self.aabb_line.colors = open3d.utility.Vector3dVector(colors)

    def get_center(self):
        center = [0.0, 0.0, 0.0]
        for mesh in self.model.meshes:
            m: open3d.geometry.TriangleMesh = mesh.mesh
            center += m.get_center()
        center /= float(len(self.model.meshes))
        return center

    def highlight(self, visual):
        visual.add_geometry(self.aabb_line, False)

    def de_highlight(self, visual):
        visual.remove_geometry(self.aabb_line, False)

    def add_mesh_to_visual(self, visual):
        for mesh in self.model.meshes:
            visual.add_geometry(mesh.mesh, False)

    def update_mesh_to_visual(self, visual):
        for mesh in self.model.meshes:
            visual.update_geometry(mesh.mesh)

    def remove_mesh_to_visual(self, visual):
        for mesh in self.model.meshes:
            visual.remove_geometry(mesh.mesh, False)

    def set_location(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z
        for mesh in self.model.meshes:
            mesh.mesh.translate([x, y, z], False)


class VirtualMapViewer:
    def __init__(self):
        self.convert = None
        self.vis = None
        self.window_name = "虚拟沙盘"
        self.running_thread = None
        self.point_clouds_to_add = []
        self.point_clouds_exist = []

        self.history_dynamic_enemies = {}
        self.history_dynamic_friends = {}

        self.card_models = {}
        for card_name, card_info in card_information_template.items():
            self.card_models[card_name] = DynamicModel(card_info.color, card_info.obj_path, card_info.scale)

        # self.state = VirtualMapViewerState.RUNNING
        # self.running_thread = threading.Thread(target=self.run)
        # self.running_thread.daemon = True
        # self.running_thread.start()

        self.convert_state = VirtualMapDataConvertState.RUNNING
        self.convert_running_thread = threading.Thread(target=self.convert_run)
        self.convert_running_thread.daemon = True
        self.convert_running_thread.start()

        self.lock = threading.Lock()

    # def focus_enemy(self, name):
    #     if name in self.dynamic_enemies:
    #         center = self.dynamic_enemies[name].get_center()
    #         ctrl: open3d.visualization.ViewControl = self.vis.get_view_control()
    #         cam_param: open3d.camera.PinholeCameraParameters = ctrl.convert_to_pinhole_camera_parameters()
    #         extrinsic = copy.deepcopy(cam_param.extrinsic)
    #         extrinsic[:, 3] = [center[0], center[1], center[2], 1.0]
    #         cam_param.extrinsic = extrinsic
    #         ctrl.convert_from_pinhole_camera_parameters(cam_param, True)
    #
    #     else:
    #         print_error(f"enemy not instanced: {name}")

    def update_enemy(self):
        new_enemies = get_global_object("enemy").copy_members()
        his_names = [v.get_name() for v in self.history_dynamic_enemies.keys()]
        new_names = [v.get_name() for v in new_enemies]

        enemy_to_add = []
        enemy_to_remove = []

        for history_member, model in self.history_dynamic_enemies.items():
            if history_member.get_name() not in new_names:
                enemy_to_remove.append(history_member)
        for new in new_enemies:
            if new.get_name() not in his_names:
                enemy_to_add.append(new)

        for enemy in enemy_to_remove:
            self.history_dynamic_enemies[enemy].remove_mesh_to_visual(self.vis)
            self.history_dynamic_enemies.pop(enemy)

        for enemy in enemy_to_add:
            name = enemy.get_name()
            model_type = enemy.get_model_type()
            # location = enemy.get_location()
            if model_type in card_information_template.keys():
                information = card_information_template[model_type]
                m: DynamicModel = DynamicModel(information.color, information.obj_path, information.scale)
                # m.set_location(location.x, -location.y, location.z)
                self.history_dynamic_enemies[enemy] = m
                m.add_mesh_to_visual(self.vis)
            else:
                print_error(f"can not find model type: {model_type}")

        for enemy, model in self.history_dynamic_enemies.items():
            for new_enemy in new_enemies:
                if enemy.get_name() == new_enemy.get_name():
                    location = new_enemy.get_location()
                    model.set_location(location.x, -location.y, location.z)
                    model.update_mesh_to_visual(self.vis)
                    break

    def update_friend(self):
        new_friends = get_global_object("friend").copy_members()
        his_names = [v.get_name() for v in self.history_dynamic_friends.keys()]
        new_names = [v.get_name() for v in new_friends]

        friend_to_add = []
        friend_to_remove = []

        for history_member, model in self.history_dynamic_friends.items():
            if history_member.get_name() not in new_names:
                friend_to_remove.append(history_member)
        for new in new_friends:
            if new.get_name() not in his_names:
                friend_to_add.append(new)

        for friend in friend_to_remove:
            self.history_dynamic_friends[friend].remove_mesh_to_visual(self.vis)
            self.history_dynamic_friends.pop(friend)

        for friend in friend_to_add:
            name = friend.get_name()
            model_type = friend.get_model_type()
            # location = friend.get_location()
            if model_type in card_information_template.keys():
                information = card_information_template[model_type]
                m: DynamicModel = DynamicModel(information.color, information.obj_path, information.scale)
                # m.set_location(location.x, -location.y, location.z)
                self.history_dynamic_friends[friend] = m
                m.add_mesh_to_visual(self.vis)
            else:
                print_error(f"can not find model type: {model_type}")

        for friend, model in self.history_dynamic_friends.items():
            for new_friend in new_friends:
                if friend.get_name() == new_friend.get_name():
                    location = new_friend.get_location()
                    model.set_location(location.x, -location.y, location.z)
                    model.update_mesh_to_visual(self.vis)
                    break

        # friends = get_global_object("friend").copy_members()
        # for friend in friends:
        #     name = friend.get_name()
        #     model_type = friend.get_model_type()
        #     location = friend.get_location()
        #     if model_type in card_information_template.keys():
        #         information = card_information_template[model_type]
        #         m: DynamicModel = DynamicModel(information.color, information.obj_path, information.scale)
        #         m.set_location(location.x, -location.y, location.z)
        #         self.history_dynamic_friends[name] = m
        #         m.add_mesh_to_visual(self.vis)
        #     else:
        #         print_error(f"can not find model type: {model_type}")

    # def clear(self):
    #     for friend in self.history_dynamic_friends.values():
    #         friend.remove_mesh_to_visual(self.vis)
    #     self.history_dynamic_friends.clear()

    def convert_run(self):
        while self.convert_state == VirtualMapDataConvertState.RUNNING:
            terrain_data = get_global_object("terrain")
            while terrain_data.has_data() is True:
                depth_data = terrain_data.pop_depth_data()
                point_cloud_inst = PointCloudInstance()

                if self.convert is None:
                    sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
                    import depth2pd as d2p
                    self.convert = d2p.Depth2PointCloudConvert()
                    if self.convert.init(0, depth_data.width, depth_data.height, 4) is False:
                        print_error("converter init failed")

                # t1 = time.time()
                self.convert.convert(depth_data.view_matrix, depth_data.project_matrix, depth_data.data.ravel(), depth_data.color_data, 1.e+08)
                positions = self.convert.get_result_position().astype('float64')
                colors = self.convert.get_result_color().reshape(-1, 4)[:, 0: 3].ravel()
                point_cloud_inst.generate_pcd_directly(positions, colors)
                # t2 = time.time()
                # print('depth to point cloud time: %s ms' % ((t2 - t1) * 1000))

                # vertices = []
                # colors = []
                # t1 = time.time()
                # for x in range(depth_data.width):
                #     for y in range(depth_data.height):
                #         depth = depth_data.data[y][x]
                #         if depth == 1.e+08:
                #             continue
                #         pixel_x = float(x) / float(depth_data.width)
                #         pixel_y = float(y) / float(depth_data.height)
                #         point_location = back_project_pixel_to_world(pixel_x, pixel_y, depth_data.data, depth_data.width, depth_data.height, depth_data.view_matrix, depth_data.project_matrix)
                #         # invert y
                #         color_r = depth_data.color_data[(y * depth_data.color_width + x) * 4 + 0]
                #         color_g = depth_data.color_data[(y * depth_data.color_width + x) * 4 + 1]
                #         color_b = depth_data.color_data[(y * depth_data.color_width + x) * 4 + 2]
                #
                #         vertices.append(point_location[0])
                #         vertices.append(point_location[1])
                #         vertices.append(point_location[2])
                #
                #         colors.append(color_r)
                #         colors.append(color_g)
                #         colors.append(color_b)
                #
                # point_cloud_inst.generate_pcd_directly(np.array(vertices), np.array(colors))
                # t2 = time.time()
                # print('depth to point time: %s ms' % ((t2 - t1) * 1000))

                self.lock.acquire()
                self.point_clouds_to_add.append(point_cloud_inst)
                self.lock.release()
            time.sleep(0.02)

    def init(self):
        self.vis = open3d.visualization.Visualizer()
        self.vis.create_window(self.window_name)
        opt = self.vis.get_render_option()
        opt.background_color = np.asarray([0, 0, 0])

    def update(self):
        # while self.state == VirtualMapViewerState.RUNNING:
        if len(self.point_clouds_to_add) > 0:
            self.lock.acquire()
            point_cloud = self.point_clouds_to_add.pop(0)
            self.lock.release()

            adjust_view = len(self.point_clouds_exist) == 0
            self.vis.add_geometry(point_cloud.get_pcd(), adjust_view)
            self.point_clouds_exist.append(point_cloud)

        while len(self.point_clouds_exist) > 100:
            self.vis.remove_geometry(self.point_clouds_exist.pop(0).get_pcd(), False)
            print("pop o3d_geometries")

        self.vis.poll_events()
        self.vis.update_renderer()
        time.sleep(0.02)

    def stop(self):
        # self.state = VirtualMapViewerState.STOP
        # self.running_thread.join()
        # self.state = VirtualMapViewerState.IDLE

        self.convert_state = VirtualMapDataConvertState.STOP
        self.convert_running_thread.join()
        self.convert_state = VirtualMapDataConvertState.IDLE
