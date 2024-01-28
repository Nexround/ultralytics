from ultralytics import YOLO
from PIL import Image
from datetime import datetime
import os
import argparse


def init_parser():
    """
    Initialize the argument parser for the inference script.

    Returns:
        args (argparse.Namespace): Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--origin_dir', type=str, required=True)
    parser.add_argument(
        '--save_dir', type=str, default='./results/')
    parser.add_argument(
        '--model', type=str, required=True)
    args = parser.parse_args()
    return args


def get_formatted_time():
    current_time = datetime.now()
    formatted_time = current_time.strftime("%Y-%m-%d %H:%M:%S")
    return formatted_time


def create_folder_if_not_exists(folder_path):
    # 使用os.path.exists()检测文件夹是否存在
    if not os.path.exists(folder_path):
        # 如果不存在，就创建文件夹
        os.makedirs(folder_path)
        print(f"文件夹 '{folder_path}' 创建成功.")
    else:
        print(f"文件夹 '{folder_path}' 已经存在.")


def save_results(results, folder_path):
    """
    Save the results of predictions as images in the specified folder.

    Args:
        results (list): List of prediction results.
        folder_path (str): Path to the folder where the images will be saved.
    """
    for index, item in enumerate(results):
        im_array = item.plot()  # plot a BGR numpy array of predictions
        im = Image.fromarray(im_array[..., ::-1])  # RGB PIL image
        result_filename = folder_path + '/'+str(index)+'.jpg'
        print("writing " + result_filename)
        im.save(result_filename)  # save image


if __name__ == '__main__':
    args = init_parser()
    folder_path = args.save_dir + get_formatted_time()
    create_folder_if_not_exists(folder_path)

    model = YOLO(args.model)
    results = model(args.origin_dir)
    save_results(results, folder_path)
    
