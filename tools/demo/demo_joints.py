import os
import torch
import numpy as np
from pathlib import Path
from pytorch3d.transforms import quaternion_to_matrix
from tqdm import tqdm
from datasets import load_dataset
# Import necessary components from your hmr4d package.
from hmr4d.utils.video_io_utils import get_video_lwh, get_video_reader
from hmr4d.utils.preproc import Tracker, Extractor, VitPoseExtractor, SLAMModel
from hmr4d.utils.geo.hmr_cam import get_bbx_xys_from_xyxy, estimate_K, convert_K_to_K4
from hmr4d.utils.geo_transform import compute_cam_angvel
from hmr4d.model.gvhmr.gvhmr_pl_demo import DemoPL
from hmr4d.utils.net_utils import detach_to_cpu, to_cuda
from hmr4d.utils.smplx_utils import make_smplx
from hmr4d.utils.pylogger import Log

# Global models (to be loaded once)
global_model = None
global_smplx_model = None

def create_cfg_for_video(video_file, static_cam=False, output_root="outputs", verbose=False):
    """
    Create a minimal configuration object for processing a single video.
    This config mimics the fields needed by run_preprocess and load_data_dict.
    """
    class CFG:
        pass
    cfg = CFG()
    cfg.video_path = video_file
    cfg.static_cam = static_cam
    cfg.verbose = verbose
    video_stem = Path(video_file).stem
    base_output = Path(output_root) / video_stem
    cfg.output_dir = str(base_output)
    cfg.preprocess_dir = str(base_output / "preprocess")
    os.makedirs(cfg.output_dir, exist_ok=True)
    os.makedirs(cfg.preprocess_dir, exist_ok=True)
    
    # Create a simple "paths" object for saving intermediate results.
    class Paths:
        pass
    paths = Paths()
    paths.bbx = str(Path(cfg.preprocess_dir) / "bbx.pt")
    paths.vitpose = str(Path(cfg.preprocess_dir) / "vitpose.pt")
    paths.vit_features = str(Path(cfg.preprocess_dir) / "vit_features.pt")
    paths.slam = str(Path(cfg.preprocess_dir) / "slam.pt")
    paths.hmr4d_results = str(Path(cfg.preprocess_dir) / "hmr4d_results.pt")
    cfg.paths = paths
    return cfg

@torch.no_grad()
def run_preprocess(cfg):
    """
    Run preprocessing to generate the inputs required for HMR4D.
    This includes bounding box tracking, 2D keypoint extraction via VitPose,
    feature extraction, and (if necessary) SLAM/DPVO for camera motion.
    """
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
    """
    Load the preprocessed data from disk and form the input dictionary for HMR4D.
    """
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

def init_global_models():
    """
    Load the HMR4D and SMPL-X models once and store them as globals.
    Adjust the checkpoint path and instantiation as needed.
    """
    global global_model, global_smplx_model
    Log.info("[Model] Loading HMR4D model...")
    ckpt_path = "path/to/hmr4d_checkpoint.ckpt"  # <<=== Update this to your actual checkpoint path.
    global_model = DemoPL()  # Instantiate your model (adjust if your DemoPL requires config arguments)
    global_model.load_pretrained_model(ckpt_path)
    global_model = global_model.eval().cuda()

    Log.info("[Model] Loading SMPL-X model for joint extraction...")
    global_smplx_model = make_smplx("supermotion").cuda()

def process_video_row(row):
    """
    Process a single video row from the dataset and return a dictionary
    with key 'joints_array' containing a list-of-lists representing the joint positions.
    
    If any error occurs during processing, the function returns {"joints_array": None}.
    """
    try:
        # Assumes the dataset row has a "video" field with the video file path.
        video_file = row["video"]
        cfg = create_cfg_for_video(video_file, static_cam=False, output_root="outputs", verbose=False)
        run_preprocess(cfg)
        data = load_data_dict(cfg)
        paths = cfg.paths

        # If the HMR4D results have not been computed, run the prediction.
        if not Path(paths.hmr4d_results).exists():
            pred = global_model.predict(data, static_cam=cfg.static_cam)
            pred = detach_to_cpu(pred)
            torch.save(pred, paths.hmr4d_results)
        else:
            pred = torch.load(paths.hmr4d_results)

        # Use the SMPL-X model to extract joints.
        smplx_out = global_smplx_model(**to_cuda(pred["smpl_params_global"]))
        joints = detach_to_cpu(smplx_out.joints)
        joints_list = joints.tolist()  # Convert the tensor to a list of lists.
        return {"joints_array": joints_list}
    except Exception as e:
        print(f"Error processing video {row.get('video', None)}: {e}")
        return {"joints_array": None}

# =============================================================================
# Example usage: processing a Hugging Face dataset with a map function
# =============================================================================
if __name__ == "__main__":
    # Initialize global models once.
    init_global_models()
    
    # For example, load a Hugging Face dataset that has a "video" column.
    
    # Here we assume you have a CSV file (or another format) with a column "video" listing video file paths.
    dataset = load_dataset("amuvarma/video_url_one_person-debug", split="train")
    
    # Map our processing function over the dataset.
    # The new column "joints_array" will contain a list of lists (or None in case of error).
    processed_dataset = dataset.map(process_video_row)
    
    # Print out the processed dataset.
    print(processed_dataset)

    processed_dataset.push_to_hub("amuvarma/joints-on-video")
