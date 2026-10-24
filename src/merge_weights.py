import os
import torch

def merge_checkpoints(weight_paths: list, output_path: str):
    """여러 PC의 동일한 모델 체크포인트를 평균내 하나의 가중치로 저장합니다."""
    num_models = len(weight_paths)
    if num_models == 0:
        raise ValueError("❌ 합칠 가중치 파일이 없습니다.")

    print(f"🔄 총 {num_models}개 PC의 가중치 병합을 시작합니다...")
    
    # 첫 번째 체크포인트를 복사해 원본 파일을 변경하지 않습니다.
    avg_state_dict = {
        key: value.clone()
        for key, value in torch.load(weight_paths[0], map_location="cpu", weights_only=True).items()
    }

    # 나머지 PC의 가중치 누적 합산
    for path in weight_paths[1:]:
        state_dict = torch.load(path, map_location="cpu", weights_only=True)
        for key in avg_state_dict:
            if key not in state_dict or state_dict[key].shape != avg_state_dict[key].shape:
                raise ValueError(f"체크포인트 구조가 다릅니다: {path} ({key})")
            if avg_state_dict[key].is_floating_point():
                avg_state_dict[key] += state_dict[key]

    # 평균값 계산
    for key in avg_state_dict:
        if avg_state_dict[key].is_floating_point():
            avg_state_dict[key] /= num_models

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    torch.save(avg_state_dict, output_path)
    print(f"🎉 가중치 평균 병합 완료! -> {output_path}")

if __name__ == "__main__":
    # 병합할 파일을 실제 경로로 바꾼 뒤 실행합니다.
    pc_weights = []
    for f in os.listdir("src/merge"):
        if f.endswith(".pth"):
            pc_weights.append(os.path.join("src/merge", f))

    if len(pc_weights) < 2:
        raise ValueError("❌ 병합할 가중치 파일이 2개 이상 필요합니다.")
    
    merged_output = "src/merge/merged_model.pth"
    
    merge_checkpoints(pc_weights, merged_output)