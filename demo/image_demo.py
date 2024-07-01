# Copyright (c) Hikvision Research Institute. All rights reserved.
# Modified by Di Huang (di.huang@wustl.edu)

import asyncio
import os
from argparse import ArgumentParser

from NICUPose.apis import (async_inference_detector, inference_detector,
                        init_detector, show_result_pyplot)


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('img', help='Image file')
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument('--out-file', default=None, help='Path to output file')
    parser.add_argument('--pose_only', action='store_true', help='whether to only show pose')
    parser.add_argument('--write_img', action='store_true', help='whether to write images')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--palette',
        default='coco',
        choices=['coco', 'voc', 'citys', 'random'],
        help='Color palette used for visualization')
    parser.add_argument(
        '--score-thr', type=float, default=0.4, help='bbox score threshold')
    parser.add_argument(
        '--async-test',
        action='store_true',
        help='whether to set async options for async inference.')
    args = parser.parse_args()
    return args


def main(args):
    # build the model from a config file and a checkpoint file
    model = init_detector(args.config, args.checkpoint, device=args.device)
    # inference with a single image
    if args.img.endswith('.jpg') or args.img.endswith('.png'):
        args.img = [args.img]
    # inference with a image folder
    elif os.listdir(args.img):
        args.img = [os.path.join(args.img, img) for img in os.listdir(args.img)]
    for img in args.img:
        result = inference_detector(model, img)
        args.out_file = os.path.join('inference/', img.split('.')[0].split('/')[-1]+'_output.jpg')
        # show the results
        show_result_pyplot(
            model,
            img,
            result,
            palette=args.palette,
            score_thr=args.score_thr,
            out_file=args.out_file,
            pose_only=args.pose_only,
            write_img=args.write_img)


async def async_main(args):
    # build the model from a config file and a checkpoint file
    model = init_detector(args.config, args.checkpoint, device=args.device)
    # test a single image
    tasks = asyncio.create_task(async_inference_detector(model, args.img))
    result = await asyncio.gather(tasks)
    # show the results
    show_result_pyplot(
        model,
        args.img,
        result[0],
        palette=args.palette,
        score_thr=args.score_thr,
        out_file=args.out_file)


if __name__ == '__main__':
    args = parse_args()
    if args.async_test:
        asyncio.run(async_main(args))
    else:
        main(args)
