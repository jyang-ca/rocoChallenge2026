#!/usr/bin/env python3
"""
RoCoChallenge HDF5 데이터셋을 LeRobotDataset 형식으로 변환합니다.

사용법:

0. huggingface_cli login을 통해 Hugging Face에 로그인합니다.

# 기본 사용 (전체 변환)
python to_lerobot_dataset.py \
    --input https://huggingface.co/datasets/rocochallenge2025/rocochallenge2025 \
    --output https://huggingface.co/datasets/yjsm1203/roco_2

# 배치 처리 예시 (192개 파일을 50개씩):

# 배치 1: 0-49
python to_lerobot_dataset.py \
    --input https://huggingface.co/datasets/rocochallenge2025/rocochallenge2025 \
    --output https://huggingface.co/datasets/yjsm1203/roco_2 \
    --start-episode 0 --end-episode 50 --clean-cache

# 배치 2: 50-99 (기존 데이터셋에 추가)
python to_lerobot_dataset.py \
    --input https://huggingface.co/datasets/rocochallenge2025/rocochallenge2025 \
    --output https://huggingface.co/datasets/yjsm1203/roco_2 \
    --start-episode 50 --end-episode 100 --resume --clean-cache

# 배치 3: 100-149
python to_lerobot_dataset.py \
    --input https://huggingface.co/datasets/rocochallenge2025/rocochallenge2025 \
    --output https://huggingface.co/datasets/yjsm1203/roco_2 \
    --start-episode 100 --end-episode 150 --resume --clean-cache

# 배치 4: 150-192 (마지막)
python to_lerobot_dataset.py \
    --input https://huggingface.co/datasets/rocochallenge2025/rocochallenge2025 \
    --output https://huggingface.co/datasets/yjsm1203/roco_2 \
    --start-episode 150 --resume --clean-cache

LeRobot Dataset Visualizer: https://huggingface.co/spaces/lerobot/visualize_dataset
"""

import argparse
import logging
import re
import shutil
from pathlib import Path
from typing import List, Dict

import h5py
import numpy as np
from huggingface_hub import HfApi, hf_hub_download

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
except ImportError:
    logger.error("lerobot이 설치되지 않았습니다. `pip install lerobot`으로 설치해주세요.")
    raise


def parse_args():
    parser = argparse.ArgumentParser(description="RoCo HDF5를 LeRobotDataset으로 변환합니다.")
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="로컬 HDF5 파일/디렉토리 경로 또는 Hugging Face repo Url",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="로컬 출력 디렉토리 경로 또는 업로드할 Hugging Face repo Url",
    )
    parser.add_argument(
        "--local-dir",
        type=str,
        default="data/cache",
        help="다운로드를 위한 로컬 캐시 디렉토리.",
    )
    parser.add_argument(
        "--robot-type",
        type=str,
        default="galaxea_r1",
        help="메타데이터에 사용할 로봇 타입 이름.",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=15,
        help="데이터셋의 초당 프레임 수 (FPS).",
    )
    parser.add_argument(
        "--start-episode",
        type=int,
        default=0,
        help="처리 시작 인덱스 (0-based, inclusive)",
    )
    parser.add_argument(
        "--end-episode",
        type=int,
        default=None,
        help="처리 종료 인덱스 (exclusive). None이면 끝까지.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="기존 원격 데이터셋에 이어서 추가 (Hub에서 다운로드 후 append)",
    )
    parser.add_argument(
        "--clean-cache",
        action="store_true",
        help="업로드 후 캐시 디렉토리 삭제",
    )

    return parser.parse_args()


def natural_sort_key(path: Path):
    """자연수 정렬 키 (1, 2, 10 순서로 정렬)"""
    return [int(c) if c.isdigit() else c.lower() for c in re.split(r'(\d+)', path.stem)]


def extract_repo_id(path_or_url: str) -> str:
    """Hugging Face URL에서 repo_id ('user/repo')를 추출하거나 원본 문자열을 반환합니다."""
    if path_or_url.startswith("http"):
        parts = path_or_url.rstrip("/").split("/")
        if "huggingface.co" in path_or_url:
            if "datasets" in parts:
                idx = parts.index("datasets")
                if idx + 2 < len(parts):
                    return f"{parts[idx+1]}/{parts[idx+2]}"
            elif len(parts) >= 2:
                return f"{parts[-2]}/{parts[-1]}"
    return path_or_url


def is_remote_repo(path_or_id: str) -> bool:
    """문자열이 Hugging Face repo ID 처럼 보이는지 확인합니다."""
    if Path(path_or_id).exists() or Path(path_or_id).parent.exists():
        return False
    if path_or_id.startswith("/") or path_or_id.startswith("./") or path_or_id.startswith("../"):
        return False
    if path_or_id.startswith("http://") or path_or_id.startswith("https://"):
        return True
    return "/" in path_or_id


def get_remote_hdf5_file_list(repo_id: str) -> List[str]:
    """원격 HF repo의 HDF5 파일 목록을 가져옵니다 (정렬됨)."""
    api = HfApi()
    files_in_repo = api.list_repo_files(repo_id=repo_id, repo_type="dataset")
    hdf5_files = [f for f in files_in_repo if f.endswith(".hdf5")]
    
    if not hdf5_files:
        raise ValueError(f"원격 repo에서 HDF5 파일을 찾을 수 없습니다: {repo_id}")
    
    # 자연수 정렬
    hdf5_files.sort(key=lambda x: natural_sort_key(Path(x)))
    return hdf5_files


def download_hdf5_file(repo_id: str, filename: str, local_dir: Path) -> Path:
    """단일 HDF5 파일을 다운로드합니다."""
    p = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        repo_type="dataset",
        local_dir=local_dir,
    )
    return Path(p)


def get_local_hdf5_paths(input_path: str, local_cache_dir: Path, 
                          start_idx: int = 0, end_idx: int = None) -> List[Path]:
    """입력 경로를 로컬 HDF5 파일 경로 리스트로 변환합니다."""
    input_path = extract_repo_id(input_path)
    path = Path(input_path)
    
    # Case 1: 로컬 파일인 경우
    if path.is_file() and path.suffix == ".hdf5":
        return [path]
    
    # Case 2: 로컬 디렉토리인 경우
    if path.is_dir():
        files = list(path.glob("**/*.hdf5"))
        if not files:
            raise ValueError(f"디렉토리에서 HDF5 파일을 찾을 수 없습니다: {input_path}")
        files.sort(key=natural_sort_key)
        
        # 범위 적용
        if end_idx is not None:
            files = files[start_idx:end_idx]
        else:
            files = files[start_idx:]
        return files

    # Case 3: 원격 HF Repo인 경우
    if is_remote_repo(input_path):
        logger.info(f"원격 HF repo 감지됨: {input_path}")
        
        # 전체 파일 목록 조회
        all_files = get_remote_hdf5_file_list(input_path)
        logger.info(f"원격 repo에 총 {len(all_files)}개의 HDF5 파일이 있습니다.")
        
        # 범위 적용
        if end_idx is not None:
            target_files = all_files[start_idx:end_idx]
        else:
            target_files = all_files[start_idx:]
        
        logger.info(f"다운로드할 파일: {len(target_files)}개 (인덱스 {start_idx} ~ {start_idx + len(target_files) - 1})")
        
        # 다운로드 디렉토리
        local_dir = local_cache_dir / input_path.replace("/", "_")
        
        downloaded_paths = []
        for i, filename in enumerate(target_files):
            logger.info(f"다운로드 중 ({i+1}/{len(target_files)}): {filename}")
            p = download_hdf5_file(input_path, filename, local_dir)
            downloaded_paths.append(p)
        
        return downloaded_paths

    raise ValueError(f"입력 '{input_path}'를 로컬에서 찾을 수 없으며 유효한 HF repo로 보이지 않습니다.")


def read_roco_hdf5(file_path: Path) -> Dict[str, np.ndarray]:
    """RoCo HDF5 파일을 읽어 LeRobot 형식으로 매핑합니다."""
    raw_data = {}
    
    def visitor(name, obj):
        if isinstance(obj, h5py.Dataset):
            raw_data[name] = np.array(obj)
            
    with h5py.File(file_path, "r") as f:
        f.visititems(visitor)

    data = {}
    
    def get_and_ensure_2d(key):
        val = raw_data.get(key)
        if val is not None:
            if val.ndim == 1:
                return val[:, None]
            return val
        return None

    # 1. 이미지 (RGB)
    image_map = {
        "observation.images.head": "observations/head_rgb",
        "observation.images.left_hand": "observations/left_hand_rgb",
        "observation.images.right_hand": "observations/right_hand_rgb",
    }
    
    for lerobot_key, hdf5_key in image_map.items():
        if hdf5_key in raw_data:
            val = raw_data[hdf5_key]
            if val.ndim == 3:
                val = val[..., None]
            data[lerobot_key] = val

    # 2. Depth 이미지
    depth_map = {
        "observation.depths.head": "observations/head_depth",
        "observation.depths.left_hand": "observations/left_hand_depth",
        "observation.depths.right_hand": "observations/right_hand_depth",
    }
    
    for lerobot_key, hdf5_key in depth_map.items():
        if hdf5_key in raw_data:
            val = raw_data[hdf5_key]
            # Depth는 보통 (T, H, W) 형태, channel 차원 추가 필요 시 처리
            if val.ndim == 3:
                val = val[..., None]  # (T, H, W) -> (T, H, W, 1)
            data[lerobot_key] = val

    # 3. 액션
    action_components = [
        "actions/left_arm_action",  
        "actions/left_gripper_action", 
        "actions/right_arm_action",
        "actions/right_gripper_action",
    ]
    
    collected_actions = []
    for k in action_components:
        val = get_and_ensure_2d(k)
        if val is not None:
            collected_actions.append(val)
    
    if collected_actions:
        data["action"] = np.concatenate(collected_actions, axis=1)

    # 4. 상태
    state_keys = [
        "observations/left_arm_joint_pos",
        "observations/left_arm_joint_vel",
        "observations/left_gripper_joint_pos",
        "observations/left_gripper_joint_vel",
        "observations/right_arm_joint_pos",
        "observations/right_arm_joint_vel",
        "observations/right_gripper_joint_pos",
        "observations/right_gripper_joint_vel",
    ]
    
    collected_states = []
    for k in state_keys:
        val = get_and_ensure_2d(k)
        if val is not None:
            collected_states.append(val)
             
    if collected_states:
        data["observation.state"] = np.concatenate(collected_states, axis=1)

    return data


def create_features_from_data(data: Dict[str, np.ndarray]) -> Dict:
    """데이터에서 features 정의를 생성합니다."""
    features = {}
    for key, val in data.items():
        if key.startswith("observation.images."):
            shape = val.shape
            features[key] = {
                "dtype": "video",
                "shape": shape[1:],
                "names": ["height", "width", "channel"],
            }
        elif key.startswith("observation.depths."):
            shape = val.shape
            features[key] = {
                "dtype": "video",
                "shape": shape[1:],
                "names": ["height", "width", "channel"],
            }
        elif key == "action":
            features[key] = {
                "dtype": "float32",
                "shape": (val.shape[1],),
                "names": [f"action_{i}" for i in range(val.shape[1])],
            }
        elif key == "observation.state":
            features[key] = {
                "dtype": "float32",
                "shape": (val.shape[1],),
                "names": [f"state_{i}" for i in range(val.shape[1])],
            }
    return features


def main():
    args = parse_args()
    local_cache_dir = Path(args.local_dir)
    local_cache_dir.mkdir(parents=True, exist_ok=True)

    is_remote_output = is_remote_repo(args.output)
    repo_id = extract_repo_id(args.output) if is_remote_output else None
    
    # 1. 입력 파일 목록 가져오기 (범위 적용)
    input_files = get_local_hdf5_paths(
        args.input, 
        local_cache_dir,
        start_idx=args.start_episode,
        end_idx=args.end_episode
    )
    logger.info(f"{len(input_files)}개의 처리할 HDF5 파일을 찾았습니다.")
    
    if not input_files:
        logger.warning("처리할 파일이 없습니다.")
        return

    # 2. 데이터셋 준비 (resume 또는 새로 생성)
    output_dir = local_cache_dir / "lerobot_build_temp"
    
    if args.resume and is_remote_output:
        logger.info(f"기존 데이터셋 다운로드 중: {repo_id}")
        
        # 기존 임시 디렉토리 삭제
        if output_dir.exists():
            shutil.rmtree(output_dir)
        
        try:
            # 기존 데이터셋 로드
            dataset = LeRobotDataset(
                repo_id=repo_id,
                root=output_dir,
            )
            logger.info(f"기존 데이터셋 로드됨: {dataset.num_episodes}개 에피소드")
        except Exception as e:
            logger.error(f"기존 데이터셋 로드 실패: {e}")
            logger.info("새 데이터셋으로 시작합니다.")
            args.resume = False
    
    if not args.resume:
        # 새 데이터셋 생성
        if output_dir.exists():
            shutil.rmtree(output_dir)
        
        first_data = read_roco_hdf5(input_files[0])
        features = create_features_from_data(first_data)
        
        create_repo_id = repo_id if is_remote_output else Path(args.output).name
        
        dataset = LeRobotDataset.create(
            repo_id=create_repo_id,
            fps=args.fps,
            robot_type=args.robot_type,
            features=features,
            root=output_dir if is_remote_output else Path(args.output),
        )
        logger.info("새 데이터셋 생성됨")

    # 3. 데이터 변환 및 추가
    for idx, file_path in enumerate(input_files):
        logger.info(f"처리 중 ({idx+1}/{len(input_files)}): {file_path.name}")
        data = read_roco_hdf5(file_path)
        
        key0 = next(iter(data))
        num_frames = len(data[key0])
        
        for i in range(num_frames):
            frame = {}
            for key, val in data.items():
                frame[key] = val[i]
            frame["task"] = "gearbox_assembly"
            dataset.add_frame(frame)
        
        dataset.save_episode()
        
        # 처리 완료한 파일 삭제 (디스크 공간 절약)
        if args.clean_cache and is_remote_repo(extract_repo_id(args.input)):
            try:
                file_path.unlink()
                logger.debug(f"캐시 파일 삭제: {file_path}")
            except Exception:
                pass
    
    # 4. Finalize
    dataset.finalize()
    logger.info(f"데이터셋 완료: 총 {dataset.num_episodes}개 에피소드")

    # 5. 업로드
    if is_remote_output:
        logger.info(f"Hugging Face Hub로 데이터셋 푸시 중: {repo_id}")
        dataset.push_to_hub()
        logger.info("업로드 완료!")
        
        # 캐시 정리
        if args.clean_cache and output_dir.exists():
            logger.info("임시 디렉토리 정리 중...")
            shutil.rmtree(output_dir)
    else:
        logger.info(f"데이터셋이 로컬에 저장되었습니다: {dataset.root}")


if __name__ == "__main__":
    main()
