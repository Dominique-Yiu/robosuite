# import h5py
# import imageio
# import pathlib
# import numpy as np
# import cv2
# from tqdm import tqdm

# # loading hdf5 file
# file_path = pathlib.Path(
#     "/home/shawn/Documents/robosuite/robosuite/models/assets/demonstrations/1703555673_5712008/image.hdf5"
# )
# saving_path = pathlib.Path("/home/shawn/Desktop/diffusion policy/simulation/lift/train")

# with h5py.File(file_path, "r") as file:
#     # Access the data in the file here
#     # print(file["data"]["demo_1"].keys())
#     mv_path = file_path.parent

#     writer = imageio.get_writer(mv_path / "demo_1.mp4", fps=30)

#     imgs_seq = file["data"]["demo_1"]["obs"]["robot0_eye_in_hand_image"]

#     for img in imgs_seq:
#         img = np.uint8(img)
#         writer.append_data(img)
#     writer.close()
# For example, you can print the keys of the datasets in the file
# with tqdm(range(100), desc="Saving...", leave=False) as pbar:
#     for idx in pbar:
#         demo_idx = f"demo_{idx + 1}"
#         img_seq = file["data"][demo_idx]["obs"]["agentview_image"]
#         # print(img_seq.shape)

#         mv_path = saving_path / f"{demo_idx}.mp4"
#         writer = imageio.get_writer(mv_path, fps=60)

#         for img in img_seq:
#             img = np.uint8(img)
#             # img_resized = cv2.resize(img, (96, 96))
#             writer.append_data(img)
#         writer.close()

# print(f"{demo_idx} is saved.")


import mujoco
import cv2
import numpy as np

# 加载 MuJoCo 模型
model = mujoco.MjModel.from_xml_path("/home/shawn/Documents/robosuite/robosuite/models/assets/arenas/table_arena.xml")
sim = mujoco.MjSim(model)

# 获取摄像头ID
camera_id = sim.model.camera_name2id("camera_name")

# 运行仿真的一步
sim.step()

# 渲染图像
width, height = 800, 600
image = sim.render(width, height, camera_name="camera_name")

# MuJoCo 返回的图像是RGB格式，OpenCV需要BGR格式
image = image[::-1, :, :]  # 上下翻转图像
image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

# 使用OpenCV显示图像
cv2.imshow("MuJoCo Camera Image", image)
cv2.waitKey(0)  # 等待按键后关闭窗口
cv2.destroyAllWindows()
