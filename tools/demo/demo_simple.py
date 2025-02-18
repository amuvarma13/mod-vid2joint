import torch
import numpy as np
import argparse
from pathlib import Path
from tqdm import tqdm
import hydra
from hydra import initialize_config_module, compose
from pytorch3d.transforms import quaternion_to_matrix

from hmr4d.utils.pylogger import Log
from hmr4d.configs import register_store_gvhmr
from hmr4d.utils.video_io_utils import get_video_lwh
from hmr4d.utils.preproc.tracker import Tracker
from hmr4d.utils.preproc.vitfeat_extractor import Extractor
from hmr4d.utils.preproc.vitpose import VitPoseExtractor
from hmr4d.utils.preproc.slam import SLAMModel
from hmr4d.utils.geo.hmr_cam import get_bbx_xys_from_xyxy, estimate_K, convert_K_to_K4
from hmr4d.utils.geo_transform import compute_cam_angvel
from hmr4d.model.gvhmr.gvhmr_pl_demo import DemoPL
from hmr4d.utils.net_utils import detach_to_cpu, to_cuda
from hmr4d.utils.smplx_utils import make_smplx

def parse_args_to_cfg():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, default="inputs/demo/tennis.mp4",
                        help="Path to input video")
    parser.add_argument("--static_cam", action="store_true", 
                        help="If true, skip SLAM (DPVO)")
    parser.add_argument("--verbose", action="store_true", 
                        help="If true, print intermediate logs")
    args = parser.parse_args()

    video_path = Path(args.video)
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")
    length, width, height = get_video_lwh(video_path)
    Log.info(f"[Input Video]: {video_path}")
    Log.info(f"(Frames, Width, Height) = ({length}, {width}, {height})")

    with initialize_config_module(version_base="1.3", config_module="hmr4d.configs"):
        overrides = [
            f"video_name={video_path.stem}",
            f"static_cam={args.static_cam}",
            f"verbose={args.verbose}",
        ]
        register_store_gvhmr()
        cfg = compose(config_name="demo", overrides=overrides)

    # Use the original video (no copying)
    cfg.video_path = str(video_path)
    return cfg

@torch.no_grad()
def run_preprocess(cfg):
    Log.info("[Preprocess] Starting full preprocessing pipeline...")
    video_path = cfg.video_path
    static_cam = cfg.static_cam

    # 1. Run YOLO-based Tracker to obtain bounding boxes
    tracker = Tracker()
    bbx_xyxy = tracker.get_one_track(video_path).float()  # (L, 4)
    bbx_xys = get_bbx_xys_from_xyxy(bbx_xyxy, base_enlarge=1.2).float()  # (L, 3)
    Log.info("[Preprocess] Tracker complete.")

    # 2. Run VitPose to extract 2D keypoints
    vitpose_extractor = VitPoseExtractor()
    vitpose = vitpose_extractor.extract(video_path, bbx_xys)
    Log.info("[Preprocess] VitPose complete.")

    # 3. Extract visual features (ViT features)
    extractor = Extractor()
    vit_features = extractor.extract_video_features(video_path, bbx_xys)
    Log.info("[Preprocess] ViT feature extraction complete.")

    # 4. Run SLAM (DPVO) to estimate camera motion if not using a static camera
    slam_results = None
    if not static_cam:
        length, width, height = get_video_lwh(video_path)
        K_fullimg = estimate_K(width, height)
        intrinsics = convert_K_to_K4(K_fullimg)
        slam = SLAMModel(video_path, width, height, intrinsics, buffer=4000, resize=0.5)
        bar = tqdm(total=length, desc="DPVO")
        while True:
            ret = slam.track()
            if ret:
                bar.update()
            else:
                break
        slam_results = slam.process()  # (L, 7), numpy array
        Log.info("[Preprocess] SLAM complete.")

    # Assemble data dictionary (all in memory; no files are saved)
    length, width, height = get_video_lwh(video_path)
    K_fullimg = estimate_K(width, height).repeat(length, 1, 1)
    if not static_cam:
        traj_quat = torch.from_numpy(slam_results[:, [6, 3, 4, 5]])
        R_w2c = quaternion_to_matrix(traj_quat).mT
    else:
        R_w2c = torch.eye(3).repeat(length, 1, 1)

    data = {
        "length": torch.tensor(length),
        "bbx_xys": bbx_xys,
        "kp2d": vitpose,
        "K_fullimg": K_fullimg,
        "cam_angvel": compute_cam_angvel(R_w2c),
        "f_imgseq": vit_features,
    }
    Log.info("[Preprocess] Full preprocessing complete.")
    return data

if __name__ == "__main__":
    cfg = parse_args_to_cfg()
    Log.info(f"[GPU]: {torch.cuda.get_device_name()}")
    Log.info(f"[GPU Properties]: {torch.cuda.get_device_properties('cuda')}")
    
    # Monkey-patch cfg to add a dummy merge_from_file if it doesn't exist
    if not hasattr(cfg, "merge_from_file"):
        def dummy_merge_from_file(file_path):
            Log.info(f"[Dummy] merge_from_file called with {file_path}; skipping.")
        cfg.merge_from_file = dummy_merge_from_file

    # Inject the patched configuration into the SLAM module
    import hmr4d.utils.preproc.slam as slam_module
    slam_module.cfg = cfg

    # Run full preprocessing (tracker, VitPose, ViT features, SLAM)
    data = run_preprocess(cfg)
    
    # Run HMR4D prediction
    Log.info("[HMR4D] Running prediction...")
    model: DemoPL = hydra.utils.instantiate(cfg.model, _recursive_=False)
    model.load_pretrained_model(cfg.ckpt_path)
    model = model.eval().cuda()
    pred = model.predict(data, static_cam=cfg.static_cam)
    pred = detach_to_cpu(pred)
    Log.info("[HMR4D] Prediction complete.")
    
    # Compute SMPL-X joints from global SMPL parameters
    smplx = make_smplx("supermotion").cuda()
    smplx_out = smplx(**to_cuda(pred["smpl_params_global"]))
    joints = smplx_out.joints
    print("SMPL-X joints tensor:")
    print(joints)
