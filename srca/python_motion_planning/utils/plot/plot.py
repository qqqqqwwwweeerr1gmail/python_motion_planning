"""
Plot tools 2D
@author: huiming zhou
"""
from io import BytesIO

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from ..environment.env import Env, Grid, Map, Node
import os
from PIL import Image
from PIL import Image, ImageDraw
from pathlib import Path
import math


import imageio

save_gif = True
save_png = True

matplotlib.use("Agg")

class Plot:
    def __init__(self, start, goal, env: Env):
        self.start = Node(start, start, 0, 0)
        self.goal = Node(goal, goal, 0, 0)
        self.env = env
        self.markersize = 600 / math.sqrt(self.env.x_range * self.env.y_range)
        self.fig = plt.figure("planning", figsize=(12,12))
        self.ax = self.fig.add_subplot()

        self.frames = []  # Store frames for GIF

    def animation(self, path: list, name: str, cost: float = None, expand: list = None, history_pose: list = None,
                  predict_path: list = None, lookahead_pts: list = None, cost_curve: list = None,
                  ellipse: np.ndarray = None) -> None:
        name = name + "\ncost: " + str(cost) if cost else name
        self.plotEnv(name)
        if expand is not None:
            self.plotExpand(expand)
        if history_pose is not None:
            self.plotHistoryPose(history_pose, predict_path, lookahead_pts)
        if path is not None:
            self.plotPath(path)

        if cost_curve:
            plt.figure("cost curve")
            self.plotCostCurve(cost_curve, name)

        if ellipse is not None:
            self.plotEllipse(ellipse)


        # gif_path = rf'E:\GIT1\python_motion_planning\my_gif\20250616\\'+'aaa.gif'
        # gif_path1 = rf'E:\GIT1\python_motion_planning\my_gif\20250616\\'+'aaa1.gif'

        import os
        import sys

        file_abs = os.path.abspath(os.path.dirname(__file__))


        print(f"Project Root: {file_abs}")


        PROJECT_ROOT = Path(__file__)
        print(f"Project Root: {PROJECT_ROOT}")

        PROJECT_ROOT = PROJECT_ROOT.parents[4]  # Adjust number based on your structure
        print(f"Project Root: {PROJECT_ROOT}")

        output_abs = str(os.path.join(PROJECT_ROOT,'outputs\output.gif'))
        print(f"output_abs Root: {output_abs}")



        if save_gif:
            self._capture_frame()
        # if save_gif:
        #     plt.savefig(output_abs[:-4]+'.png')

        if isinstance(self.env, Grid):
            all_obs = self.env.obstacles
            ld_point = (0,0)
            ru_point = (10,10)
            for tp in all_obs:
                ld_point = (min(tp[0], ld_point[0]), min(tp[1], ld_point[1]))
                ru_point = (max(tp[0], ru_point[0]), max(tp[1], ru_point[1]))
        if isinstance(self.env, Map):
            all_obs = self.env.grid_map
            ld_point = (0,0)
            ru_point = (10,10)
            for tp in all_obs:
                ld_point = (min(tp[0], ld_point[0]), min(tp[1], ld_point[1]))
                ru_point = (max(tp[0], ru_point[0]), max(tp[1], ru_point[1]))


        # if save_gif: self._save_gif(gif_path)
        if save_gif: self._save_gif2(output_abs, point1=ld_point, point2=ru_point)

        # plt.show()

    def _capture_frame(self):
        '''Save current plot as a frame in memory (no temp files).'''
        buf = BytesIO()  # Create an in-memory binary buffer
        plt.savefig(buf, format='png')  # Save plot directly to buffer
        buf.seek(0)  # Move buffer position to start for reading
        img = Image.open(buf)  # Open the image from buffer
        self.frames.append(img.copy())  # Store the frame
        buf.close()  # Free memory (optional, Python usually handles this)


    def _save_gif(self, gif_path: str, loop_count: int = None):
        if not self.frames:
            print("No frames to save.")
            return

        processed_frames = []
        for frame in self.frames:
            if frame.mode == 'RGBA':
                background = Image.new('RGB', frame.size, (255, 255, 255))
                background.paste(frame, mask=frame.split()[3])
                processed_frames.append(background)
            else:
                processed_frames.append(frame.convert('RGB'))

        save_kwargs = {
            "save_all": True,
            "append_images": processed_frames[1:] if len(processed_frames) > 1 else [],
            "duration": 100, # Milliseconds per frame
            "optimize": True
        }

        if loop_count is not None:
            save_kwargs["loop"] = loop_count

        processed_frames[0].save(gif_path, **save_kwargs)



    def _save_gif2(self, gif_path: str, loop_count: int = None,
                   point1=(0,0), point2=(50, 30)):
        if not self.frames:
            print("No frames to save. Call draw_frame() multiple times first.")
            return

        self.fig.canvas.draw()
        fig_width_pixels, fig_height_pixels = self.fig.canvas.get_width_height()

        crop_data_xmin = min(point1[0], point2[0])
        crop_data_ymin = min(point1[1], point2[1])
        crop_data_xmax = max(point1[0], point2[0])
        crop_data_ymax = max(point1[1], point2[1])

        pixel_bl_data = self.ax.transData.transform((crop_data_xmin, crop_data_ymin))
        pixel_tr_data = self.ax.transData.transform((crop_data_xmax, crop_data_ymax))

        left = int(min(pixel_bl_data[0], pixel_tr_data[0]))
        right = int(max(pixel_bl_data[0], pixel_tr_data[0]))

        upper = int(fig_height_pixels - max(pixel_bl_data[1], pixel_tr_data[1]))
        lower = int(fig_height_pixels - min(pixel_bl_data[1], pixel_tr_data[1]))

        frame_width, frame_height = self.frames[0].size
        left = max(0, min(left, frame_width))
        right = max(0, min(right, frame_width))
        upper = max(0, min(upper, frame_height))
        lower = max(0, min(lower, frame_height))

        if left >= right or upper >= lower:
            print \
                (f"Warning: Calculated crop region resulted in zero or negative size: ({left}, {upper}, {right}, {lower}). Cannot crop.")
            return

        processed_frames = []
        for frame in self.frames:
            if frame.mode == 'RGBA':
                background = Image.new('RGB', frame.size, (255, 255, 255))
                background.paste(frame, mask=frame.split()[3])
                img = background
            else:
                img = frame.convert('RGB')

            cropped_img = img.crop((left, upper, right, lower))
            processed_frames.append(cropped_img)

        if not processed_frames:
            print("Error: No processed frames to save after cropping.")
            return

        save_kwargs = {
            "save_all": True,
            "append_images": processed_frames[1:] if len(processed_frames) > 1 else [],
            "duration": 100,
            "optimize": True
        }
        if loop_count is not None:
            save_kwargs["loop"] = loop_count

        processed_frames[0].save(gif_path, **save_kwargs)
        print(f"GIF saved to {gif_path}")


    def plotEnv(self, name: str) -> None:
        '''
        Plot environment with static obstacles.

        Parameters
        ----------
        name: Algorithm name or some other information
        '''


        if isinstance(self.env, Grid):
            obs_x = [x[0] for x in self.env.obstacles]
            obs_y = [x[1] for x in self.env.obstacles]
            # plt.plot(obs_x, obs_y, "^b")
            # plt.plot(obs_x, obs_y, "sk")
            plt.plot(obs_x, obs_y, color="#000000", marker='s', linestyle='',markersize=self.markersize)
            # plt.plot(obs_x, obs_y, color="#000000", marker='s')

            if self.start.x != -1:
                plt.plot(self.start.x, self.start.y, marker="s", color="#ff0000", markersize=self.markersize)
            if self.goal.x != -1:
                plt.plot(self.goal.x, self.goal.y, marker="s", color="#1155cc", markersize=self.markersize)

            if save_png:
                PROJECT_ROOT = Path(__file__)
                PROJECT_ROOT = PROJECT_ROOT.parents[4]  # Adjust number based on your structure
                output_abs = str(os.path.join(PROJECT_ROOT, 'outputs\output.png'))

                self._capture_frame()

                # plt.savefig(output_abs)

                point1 = (0,0)
                point2 = (self.env.x_range-1,self.env.y_range-1)
                # point2 = (101,101)


                fig_width_pixels, fig_height_pixels = self.fig.canvas.get_width_height()

                crop_data_xmin = min(point1[0], point2[0])
                crop_data_ymin = min(point1[1], point2[1])
                crop_data_xmax = max(point1[0], point2[0])
                crop_data_ymax = max(point1[1], point2[1])

                pixel_bl_data = self.ax.transData.transform((crop_data_xmin, crop_data_ymin))
                pixel_tr_data = self.ax.transData.transform((crop_data_xmax, crop_data_ymax))

                left = int(min(pixel_bl_data[0], pixel_tr_data[0]))
                right = int(max(pixel_bl_data[0], pixel_tr_data[0]))

                upper = int(fig_height_pixels - max(pixel_bl_data[1], pixel_tr_data[1]))
                lower = int(fig_height_pixels - min(pixel_bl_data[1], pixel_tr_data[1]))

                frame_width, frame_height = self.frames[0].size
                left = max(0, min(left, frame_width))
                right = max(0, min(right, frame_width))
                upper = max(0, min(upper, frame_height))
                lower = max(0, min(lower, frame_height))


                processed_frames = []
                for frame in self.frames:
                    if frame.mode == 'RGBA':
                        background = Image.new('RGB', frame.size, (255, 255, 255))
                        background.paste(frame, mask=frame.split()[3])
                        img = background
                    else:
                        img = frame.convert('RGB')

                    cropped_img = img.crop((left, upper, right, lower))
                cropped_img.save(output_abs)

        # plt.show(block=False)  # Non-blocking display



        if isinstance(self.env, Map):
            ax = self.fig.add_subplot()
            # boundary
            for (ox, oy, w, h) in self.env.boundary:
                ax.add_patch(patches.Rectangle(
                        (ox, oy), w, h,
                        edgecolor='blue',
                        facecolor='yellow',
                        fill=True
                    )
                )
            # rectangle obstacles
            for (ox, oy, w, h) in self.env.obs_rect:
                ax.add_patch(patches.Rectangle(
                        (ox, oy), w, h,
                        edgecolor='pink',
                        facecolor='gold',
                        fill=True
                    )
                )
            # circle obstacles
            for (ox, oy, r) in self.env.obs_circ:
                ax.add_patch(patches.Circle(
                        (ox, oy), r,
                        edgecolor='blue',
                        facecolor='purple',
                        fill=True
                    )
                )

        plt.title(name)
        plt.axis("equal")

        if save_png:
            img_binary_buffer = BytesIO()
            cropped_img.save(img_binary_buffer, format='PNG')  # or 'JPEG', etc.

            # Get raw bytes
            img_binary = img_binary_buffer.getvalue()
            return img_binary


    def plotExpand(self, expand: list) -> None:
        '''
        Plot expanded grids using in graph searching.

        Parameters
        ----------
        expand: Expanded grids during searching
        '''
        if self.start in expand:
            expand.remove(self.start)
        if self.goal in expand:
            expand.remove(self.goal)

        count = 0
        if isinstance(self.env, Grid):
            for x in expand:
                count += 1
                plt.plot(x.x, x.y, color="#afdfdf", marker='s' ,markersize=self.markersize)
                # plt.plot(x.x, x.y, color="#afdfdf", marker='s' ,markersize=2)
                plt.gcf().canvas.mpl_connect('key_release_event',
                                            lambda event: [exit(0) if event.key == 'escape' else None])
                if count < len(expand) / 3:         length = 20
                elif count < len(expand) * 2 / 3:   length = 30
                # else:                               length = 40
                else:                               length = 80
                if count % length == 0:
                    plt.pause(0.001)
                    if save_gif: self._capture_frame()
        
        if isinstance(self.env, Map):
            for x in expand:
                count += 1
                if x.parent:
                    plt.plot([x.parent[0], x.x], [x.parent[1], x.y], 
                        color="#eadddd", linestyle="-")
                    plt.gcf().canvas.mpl_connect('key_release_event',
                                                 lambda event:
                                                 [exit(0) if event.key == 'escape' else None])
                    if count % 10 == 0:
                        plt.pause(0.001)
                        if save_gif: self._capture_frame()


        plt.pause(0.001)
        if save_gif: self._capture_frame()


    def plotPath(self, path: list, path_color: str='#13ae00', path_style: str="-") -> None:
        '''
        Plot path in global planning.

        Parameters
        ----------
        path: Path found in global planning
        '''
        path_x = [path[i][0] for i in range(len(path))]
        path_y = [path[i][1] for i in range(len(path))]
        plt.plot(path_x, path_y, path_style, linewidth='2', color=path_color)
        plt.plot(self.start.x, self.start.y, marker="s", color="#ff0000")
        plt.plot(self.goal.x, self.goal.y, marker="s", color="#1155cc")

    def plotAgent(self, pose: tuple, radius: float=1) -> None:
        '''
        Plot agent with specifical pose.

        Parameters
        ----------
        pose: Pose of agent
        radius: Radius of agent
        '''
        x, y, theta = pose
        ref_vec = np.array([[radius / 2], [0]])
        rot_mat = np.array([[np.cos(theta), -np.sin(theta)],
                            [np.sin(theta),  np.cos(theta)]])
        end_pt = rot_mat @ ref_vec + np.array([[x], [y]])

        try:
            self.ax.artists.pop()
            for art in self.ax.get_children():
                if isinstance(art, matplotlib.patches.FancyArrow):
                    art.remove()
        except:
            pass

        self.ax.arrow(x, y, float(end_pt[0]) - x, float(end_pt[1]) - y,
                width=0.1, head_width=0.40, color="r")
        circle = plt.Circle((x, y), radius, color="r", fill=False)
        self.ax.add_artist(circle)

    def plotHistoryPose(self, history_pose, predict_path=None, lookahead_pts=None) -> None:
        lookahead_handler = None
        for i, pose in enumerate(history_pose):
            if i < len(history_pose) - 1:
                plt.plot([history_pose[i][0], history_pose[i + 1][0]],
                    [history_pose[i][1], history_pose[i + 1][1]], c="#13ae00")
                if predict_path is not None:
                    plt.plot(predict_path[i][:, 0], predict_path[i][:, 1], c="#ddd")
            i += 1

            # agent
            self.plotAgent(pose)

            # lookahead
            if lookahead_handler is not None:
                lookahead_handler.remove()
            if lookahead_pts is not None:
                try:
                    lookahead_handler = self.ax.scatter(lookahead_pts[i][0], lookahead_pts[i][1], c="b")
                except:
                    lookahead_handler = self.ax.scatter(lookahead_pts[-1][0], lookahead_pts[-1][1], c="b")

            plt.gcf().canvas.mpl_connect('key_release_event',
                                        lambda event: [exit(0) if event.key == 'escape' else None])
            if i % 5 == 0:
                plt.pause(0.01)
                if save_gif: self._capture_frame()

    def plotCostCurve(self, cost_list: list, name: str) -> None:
        '''
        Plot cost curve with epochs using in evolutionary searching.

        Parameters
        ----------
        cost_list: Cost with epochs
        name: Algorithm name or some other information
        '''
        plt.plot(cost_list, color="b")
        plt.xlabel("epochs")
        plt.ylabel("cost value")
        plt.title(name)
        plt.grid()

    def plotEllipse(self, ellipse: np.ndarray, color: str = 'darkorange', linestyle: str = '--', linewidth: float = 2):
        plt.plot(ellipse[0, :], ellipse[1, :], linestyle=linestyle, color=color, linewidth=linewidth)

    def connect(self, name: str, func) -> None:
        self.fig.canvas.mpl_connect(name, func)

    def clean(self):
        plt.cla()

    def update(self):
        self.fig.canvas.draw_idle()

    @staticmethod
    def plotArrow(x, y, theta, length, color):
        angle = np.deg2rad(30)
        d = 0.5 * length
        w = 2

        x_start, y_start = x, y
        x_end = x + length * np.cos(theta)
        y_end = y + length * np.sin(theta)

        theta_hat_L = theta + np.pi - angle
        theta_hat_R = theta + np.pi + angle

        x_hat_start = x_end
        x_hat_end_L = x_hat_start + d * np.cos(theta_hat_L)
        x_hat_end_R = x_hat_start + d * np.cos(theta_hat_R)

        y_hat_start = y_end
        y_hat_end_L = y_hat_start + d * np.sin(theta_hat_L)
        y_hat_end_R = y_hat_start + d * np.sin(theta_hat_R)

        plt.plot([x_start, x_end], [y_start, y_end], color=color, linewidth=w)
        plt.plot([x_hat_start, x_hat_end_L], [y_hat_start, y_hat_end_L], color=color, linewidth=w)
        plt.plot([x_hat_start, x_hat_end_R], [y_hat_start, y_hat_end_R], color=color, linewidth=w)

    @staticmethod
    def plotCar(x, y, theta, width, length, color):
        theta_B = np.pi + theta

        xB = x + length / 4 * np.cos(theta_B)
        yB = y + length / 4 * np.sin(theta_B)

        theta_BL = theta_B + np.pi / 2
        theta_BR = theta_B - np.pi / 2

        x_BL = xB + width / 2 * np.cos(theta_BL)        # Bottom-Left vertex
        y_BL = yB + width / 2 * np.sin(theta_BL)
        x_BR = xB + width / 2 * np.cos(theta_BR)        # Bottom-Right vertex
        y_BR = yB + width / 2 * np.sin(theta_BR)

        x_FL = x_BL + length * np.cos(theta)               # Front-Left vertex
        y_FL = y_BL + length * np.sin(theta)
        x_FR = x_BR + length * np.cos(theta)               # Front-Right vertex
        y_FR = y_BR + length * np.sin(theta)

        plt.plot([x_BL, x_BR, x_FR, x_FL, x_BL],
                 [y_BL, y_BR, y_FR, y_FL, y_BL],
                 linewidth=1, color=color)

        Plot.plotArrow(x, y, theta, length / 2, color)