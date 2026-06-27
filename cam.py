import argparse
import cv2
import os
import subprocess

def image_resize(image_path, size):
    image = cv2.imread(image_path)
    h, w = image.shape[:2]
    
    # Resize shorter side to size, maintaining aspect ratio
    if h < w:
        new_h = size
        new_w = int(w * size / h)
    else:
        new_w = size
        new_h = int(h * size / w)
    
    resized = cv2.resize(image, (new_w, new_h))
    
    # Center crop to size x size
    start_x = (new_w - size) // 2
    start_y = (new_h - size) // 2
    print(image.shape, resized.shape, start_x, start_y)
    cropped = resized[start_y:start_y + size, start_x:start_x + size]
    print(f'Resized image dimensions: {resized.shape}, Cropped image dimensions: {cropped.shape}')
    return cropped

def parse_args():
    parse_args = argparse.ArgumentParser(description='CAM')
    parse_args.add_argument('--model', type=str, help='model path')
    parse_args.add_argument('--config', type=str, help='model config path')
    # parse_args.add_argument('--output', type=str, help='output image path')

    return parse_args.parse_args()

def main():
    args = parse_args()

    img_dir = "./data/eye_area/"
    save_dir = "./heatmap/eye_area/"
    catagory = ["train/abnormal/", "train/normal/", "val/abnormal/", "val/normal/"]

    for cata in catagory:
        img_dir_cata = os.path.join(img_dir, cata)
        save_dir_cata = os.path.join(save_dir, cata)
        for img_name in os.listdir(img_dir_cata):
            img_path = os.path.join(img_dir_cata, img_name)
            img_name_pre = img_name.split('.')[0]

            resized_image = image_resize(img_path, 224)
            tmp_image_name = 'resized_output.jpg'
            cv2.imwrite(tmp_image_name, resized_image)
            save_path = os.path.join(save_dir_cata, f'{img_name_pre}_cam.jpg')

            try:
                subprocess.run([
                    'python', 'tools/visualization/vis_cam.py',
                    tmp_image_name,
                    args.config,
                    args.model,
                    '--num-extra-tokens', '0',
                    '--target-layers', 'backbone.stages.3.blocks.0.channel_block.norm2',
                    '--save-path', save_path,
                    '--vit-like'
                ], check=True)
            except KeyboardInterrupt:
                print("\nInterrupted by user. Exiting...")
                return

if __name__ == '__main__':
    main()