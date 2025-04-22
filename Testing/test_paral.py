import os
import torch
import sys
import argparse
import numpy as np
import imageio
import cv2
import timeit
from model import td4_psp18, td2_psp50, pspnet
from torch.utils.data import DataLoader
from dataloader_paral import cityscapesLoader

torch.backends.cudnn.benchmark = True
torch.cuda.cudnn_enabled = True

def convert_video_to_frames(input_video, output_dir):
    """Converts the input video to frames and saves them in the output directory."""
    os.makedirs(output_dir, exist_ok=True)
    # Use FFmpeg to extract frames at 1 frame per second (fps=1)
    os.system(f"ffmpeg -i {input_video} -vf 'fps=1' {output_dir}/frame_%04d.png")
    print(f"Frames extracted to {output_dir}")

def process_video(input_video, frame_output_dir, model, device, args):
    """Processes a single video: converts to frames and runs inference."""
    convert_video_to_frames(input_video, frame_output_dir)

    # Create dataset and dataloader
    dataset = cityscapesLoader(img_path=frame_output_dir, in_size=(769, 1537))
    data_loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)

    timer = 0.0
    i = 0

    with torch.no_grad():
        for i, (image, img_name, folder, ori_size) in enumerate(data_loader):
            image = image.to(device)

            # Timing the frame processing
            torch.cuda.synchronize()
            start_time = timeit.default_timer()

            # Run inference
            output = model(image, pos_id=i % 4)

            torch.cuda.synchronize()
            elapsed_time = timeit.default_timer() - start_time

            if i > 5:
                timer += elapsed_time
            else:
                print("Not enough frames processed for average latency.")

            # Post-processing (getting the prediction)
            pred = np.squeeze(output.data.max(1)[1].cpu().numpy(), axis=0)
            pred = pred.astype(np.int8)

            # Resize the output (pred) to match the original frame size
            pred = cv2.resize(pred, (ori_size[0] // 4, ori_size[1] // 4), interpolation=cv2.INTER_NEAREST)
            decoded = dataset.decode_segmap(pred)

            # Save the result
            save_dir = os.path.join(args.output_path, folder)
            res_path = os.path.join(save_dir, img_name)
            if not os.path.exists(save_dir):
                os.mkdir(save_dir)
            imageio.imwrite(res_path, decoded.astype(np.uint8))

            print(f"Frame {i + 1:2d}   RunningTime/Latency={elapsed_time:3.5f} s")

    print(f"Model: {args.model}")
    print(f"Average RunningTime/Latency={timer / (i - 5):3.5f} s")


def test(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model based on the selection
    if args.model == 'td4-psp18':
        path_num = 4
        model = td4_psp18.td4_psp18(nclass=19, path_num=path_num, model_path=args._td4_psp18_path)

    elif args.model == 'td2-psp50':
        path_num = 2
        model = td2_psp50.td2_psp50(nclass=19, path_num=path_num, model_path=args._td2_psp50_path)

    elif args.model == 'psp101':
        path_num = 1
        model = pspnet.pspnet(nclass=19, model_path=args._psp101_path)

    model.eval()
    model.to(device)

    # Loop through all videos in the input folder
    for video_name in os.listdir(args.input_video_dir):
        video_path = os.path.join(args.input_video_dir, video_name)
        
        if os.path.isdir(video_path):
            print(f"Processing video: {video_name}")

            # Directory to store frames for each video
            frame_output_dir = os.path.join(args.frame_output_dir, video_name)
            os.makedirs(frame_output_dir, exist_ok=True)

            # Process the current video
            process_video(video_path, frame_output_dir, model, device, args)

            print(f"Finished processing video: {video_name}")
        else:
            print(f"Skipping non-directory item: {video_name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Params")
    parser.add_argument("--input_video_dir", type=str, required=True, help="Directory containing videos")
    parser.add_argument("--frame_output_dir", type=str, required=True, help="Directory to save extracted frames")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save segmentation results")
    parser.add_argument("--_td4_psp18_path", type=str, default="./checkpoint/td4-psp18.pkl", help="Path to PSP Model")
    parser.add_argument("--_td2_psp50_path", type=str, default="./checkpoint/td2-psp50.pkl", help="Path to PSP Model")
    parser.add_argument("--_psp101_path", type=str, default="./checkpoint/psp101.pkl", help="Path to PSP Model")
    parser.add_argument("--gpu", type=str, default='0', help="gpu_id")
    parser.add_argument("--model", type=str, default='td4-psp18', help="model in [td4-psp18, td2_psp50, psp101]")
    
    args = parser.parse_args()

    test(args)

    # Example usage:
    # docker run --gpus all -v /path/to/videos:/videos -v /path/to/output:/output segmentation-container \
    # python3 test.py --input_video_dir /videos --frame_output_dir /output/frames --output_path /output/results
