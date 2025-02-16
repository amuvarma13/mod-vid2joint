import torch
import numpy as np
import argparse
import hydra
from hydra import initialize_config_module, compose
from pathlib import Path
from pytorch3d.transforms import quaternion_to_matrix
from tqdm import tqdm
import requests
import tempfile  # still imported in case you need it elsewhere
from uuid import uuid4  # for generating random IDs
from datasets import load_dataset

from hmr4d.utils.pylogger import Log
from hmr4d.configs import register_store_gvhmr
from hmr4d.utils.video_io_utils import get_video_lwh, get_video_reader
from hmr4d.utils.preproc import Tracker, Extractor, VitPoseExtractor, SLAMModel
from hmr4d.utils.geo.hmr_cam import get_bbx_xys_from_xyxy, estimate_K, convert_K_to_K4
from hmr4d.utils.geo_transform import compute_cam_angvel
from hmr4d.model.gvhmr.gvhmr_pl_demo import DemoPL
from hmr4d.utils.net_utils import detach_to_cpu, to_cuda
from hmr4d.utils.smplx_utils import make_smplx

CRF = 23  # quality parameter (unused in joint extraction)

# --- Default video URL variable ---
video_url = (
    "https://firebasestorage.googleapis.com/v0/b/humanview-d6bc8.appspot.com/"
    "o/cropped_video.mp4?alt=media&token=8e90ea8c-9e13-40af-a96b-690b218be515"
)

def parse_args_to_cfg(video_url_override=None):
    """Parse arguments and create the configuration."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--video",
        type=str,
        default=video_url,  # default is the provided remote video URL
        help="Path to a local video file or a video URL",
    )
    parser.add_argument("--output_root", type=str, default=None, help="by default to outputs/demo")
    parser.add_argument("-s", "--static_cam", action="store_true", help="If true, skip DPVO")
    parser.add_argument("--verbose", action="store_true", help="If true, show extra logs")
    args = parser.parse_args()

    # Override the video URL if a parameter is provided.
    if video_url_override is not None:
        args.video = video_url_override

    # Check if the video argument is a URL. If so, download it.
    if args.video.startswith("http"):
        Log.info(f"[Download] Downloading video from URL: {args.video}")
        response = requests.get(args.video, stream=True)
        response.raise_for_status()
        # Save the video to a randomly named file in the temps/ folder.
        temp_dir = Path("temps")
        temp_dir.mkdir(parents=True, exist_ok=True)
        random_filename = f"{uuid4().hex}.mp4"
        temp_vid_path = temp_dir / random_filename
        with temp_vid_path.open("wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        Log.info(f"[Download] Video downloaded to {temp_vid_path}")
        video_path = temp_vid_path
    else:
        video_path = Path(args.video)

    assert video_path.exists(), f"Video not found at {video_path}"
    length, width, height = get_video_lwh(video_path)
    Log.info(f"[Input]: {video_path}")
    Log.info(f"(L, W, H) = ({length}, {width}, {height})")

    with initialize_config_module(version_base="1.3", config_module="hmr4d.configs"):
        overrides = [
            f"video_name={video_path.stem}",
            f"static_cam={args.static_cam}",
            f"verbose={args.verbose}",
        ]
        if args.output_root is not None:
            overrides.append(f"output_root={args.output_root}")
        register_store_gvhmr()
        cfg = compose(config_name="demo", overrides=overrides)

    Log.info(f"[Output Dir]: {cfg.output_dir}")
    Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)
    Path(cfg.preprocess_dir).mkdir(parents=True, exist_ok=True)

    # Use the downloaded or original video path directly.
    cfg.video_path = str(video_path)
    return cfg


@torch.no_grad()
def run_preprocess(cfg):
    """Run preprocessing to generate necessary inputs for HMR4D."""
    Log.info("[Preprocess] Start!")
    tic = Log.time()
    video_path = cfg.video_path
    paths = cfg.paths
    static_cam = cfg.static_cam

    # --- Bounding Box Tracking ---
    if not Path(paths.bbx).exists():
        tracker = Tracker()
        bbx_xyxy = tracker.get_one_track(video_path).float()  # (L, 4)
        bbx_xys = get_bbx_xys_from_xyxy(bbx_xyxy, base_enlarge=1.2).float()  # (L, 3)
        torch.save({"bbx_xyxy": bbx_xyxy, "bbx_xys": bbx_xys}, paths.bbx)
        del tracker
    else:
        bbx_xys = torch.load(paths.bbx)["bbx_xys"]
        Log.info(f"[Preprocess] Loaded bbx (xyxy, xys) from {paths.bbx}")

    # --- 2D Keypoints Extraction via VitPose ---
    if not Path(paths.vitpose).exists():
        vitpose_extractor = VitPoseExtractor()
        vitpose = vitpose_extractor.extract(video_path, bbx_xys)
        torch.save(vitpose, paths.vitpose)
        del vitpose_extractor
    else:
        vitpose = torch.load(paths.vitpose)
        Log.info(f"[Preprocess] Loaded vitpose from {paths.vitpose}")

    # --- Vit Features Extraction ---
    if not Path(paths.vit_features).exists():
        extractor = Extractor()
        vit_features = extractor.extract_video_features(video_path, bbx_xys)
        torch.save(vit_features, paths.vit_features)
        del extractor
    else:
        Log.info(f"[Preprocess] Loaded vit_features from {paths.vit_features}")

    # --- SLAM/DPVO (for non-static cameras) ---
    if not static_cam:
        if not Path(paths.slam).exists():
            length, width, height = get_video_lwh(cfg.video_path)
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
            slam_results = slam.process()  # (L, 7) numpy array
            torch.save(slam_results, paths.slam)
        else:
            Log.info(f"[Preprocess] Loaded slam results from {paths.slam}")

    Log.info(f"[Preprocess] End. Time elapsed: {Log.time()-tic:.2f}s")


def load_data_dict(cfg):
    """Load data from preprocessed files to form the input for HMR4D."""
    paths = cfg.paths
    length, width, height = get_video_lwh(cfg.video_path)
    if cfg.static_cam:
        R_w2c = torch.eye(3).repeat(length, 1, 1)
    else:
        traj = torch.load(paths.slam)
        traj_quat = torch.from_numpy(traj[:, [6, 3, 4, 5]])
        R_w2c = quaternion_to_matrix(traj_quat).mT
    K_fullimg = estimate_K(width, height).repeat(length, 1, 1)

    data = {
        "length": torch.tensor(length),
        "bbx_xys": torch.load(paths.bbx)["bbx_xys"],
        "kp2d": torch.load(paths.vitpose),
        "K_fullimg": K_fullimg,
        "cam_angvel": compute_cam_angvel(R_w2c),
        "f_imgseq": torch.load(paths.vit_features),
    }
    return data


def load_models(cfg):
    """
    Load the HMR4D model and the SMPL-X model for joint extraction.
    """
    Log.info("[Model] Loading HMR4D model...")
    model: DemoPL = hydra.utils.instantiate(cfg.model, _recursive_=False)
    model.load_pretrained_model(cfg.ckpt_path)
    model = model.eval().cuda()

    Log.info("[Model] Loading SMPL-X model for joint extraction...")
    smplx_model = make_smplx("supermotion").cuda()

    return model, smplx_model


@torch.no_grad()
def process_video(cfg, model, smplx_model):
    """
    Orchestrate the video processing:
      1. Preprocess the video.
      2. Run HMR4D prediction.
      3. Extract and save joint positions.
    
    Returns:
        joints (torch.Tensor): The extracted joints.
    """
    # Run preprocessing
    run_preprocess(cfg)
    data = load_data_dict(cfg)
    paths = cfg.paths

    # Run HMR4D Prediction
    if not Path(paths.hmr4d_results).exists():
        Log.info("[HMR4D] Predicting...")
        tic = Log.sync_time()
        pred = model.predict(data, static_cam=cfg.static_cam)
        pred = detach_to_cpu(pred)
        data_time = data["length"] / 30
        Log.info(f"[HMR4D] Elapsed: {Log.sync_time() - tic:.2f}s for data-length={data_time:.1f}s")
        torch.save(pred, paths.hmr4d_results)
    else:
        pred = torch.load(paths.hmr4d_results)

    # Joint extraction using the SMPL-X model
    Log.info("[Joint Extraction] Extracting joints from SMPL parameters...")
    smplx_out = smplx_model(**to_cuda(pred["smpl_params_global"]))
    joints = detach_to_cpu(smplx_out.joints)

    # Save joints to file
    joints_path = Path(cfg.output_dir) / "joints.pt"
    torch.save(joints, joints_path)
    Log.info(f"Extracted joints saved to {joints_path}")

    return joints


def main_orchestration(video_url=None):
    """
    Main orchestration function that loads models and processes the video.
    
    Parameters:
        video_url (str, optional): A video URL to override the default video path.
    
    Returns:
        list of lists: The extracted joints.
    """
    cfg = parse_args_to_cfg(video_url_override=video_url)
    Log.info(f"[GPU]: {torch.cuda.get_device_name()}")
    Log.info(f"[GPU]: {torch.cuda.get_device_properties('cuda')}")
    model, smplx_model = load_models(cfg)
    joints_tensor = process_video(cfg, model, smplx_model)
    
    # Delete the downloaded video if it was saved in the temps/ folder.
    downloaded_video = Path(cfg.video_path)
    if downloaded_video.parent.name == "temps":
        try:
            downloaded_video.unlink()
            Log.info(f"Deleted downloaded video: {downloaded_video}")
        except Exception as e:
            Log.warning(f"Failed to delete downloaded video: {e}")

    # Convert the tensor to a list of lists before returning.
    return joints_tensor.tolist()


if __name__ == "__main__":
    # Pass the desired video URL here. If None, the default (or command-line) is used.

    dsn = "amuvarma/video_url_one_person-debug"
    ds = load_dataset(dsn, split="train")

    #map the dataset through 

    def map_fn(x):
        video_url = x["video_url"]
        joints = main_orchestration(video_url=video_url)
        x["joints"] = joints
        return x
    
    ds = ds.map(map_fn, num_proc=5)
    
    print(ds)
    ds.push_to_hub("amuvarma/video_url_one_person-debug-joints")
